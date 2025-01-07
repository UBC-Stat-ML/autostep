from abc import ABCMeta, abstractmethod

import jax
from jax import flatten_util
from jax import lax
from jax import numpy as jnp
from jax import random

from numpyro import infer
from numpyro import util

from autostep import utils
from autostep import statistics

class AutoStep(infer.mcmc.MCMCKernel, metaclass=ABCMeta):

    def init_alter_step_size_loop_funs(self):
        self.shrink_step_size_cond_fun = utils.gen_alter_step_size_cond_fun(
            self.selector.should_shrink)
        self.shrink_step_size_body_fun = utils.gen_alter_step_size_body_fun(self, -1)
        self.grow_step_size_cond_fun = utils.gen_alter_step_size_cond_fun(
            self.selector.should_grow)
        self.grow_step_size_body_fun = utils.gen_alter_step_size_body_fun(self, 1)
    
    @staticmethod
    @abstractmethod
    def init_state(initial_params, sample_field_flat_shape, rng_key):
        """
        Initialize the state of the sampler.

        :param initial_params: Initial values for the latent parameters.
        :param sample_field_flat_shape: Shape of the flattened latent variables 
        :param rng_key: The PRNG key that the sampler should use for simulation.
        :return: The initial state of the sampler.
        """
        raise NotImplementedError

    # note: this is called by the enclosing numpyro.infer.MCMC object
    def init(self, rng_key, num_warmup, initial_params, model_args, model_kwargs):
        # determine number of adaptation rounds
        self.adapt_rounds = utils.num_warmup_to_adapt_rounds(num_warmup)

        # initialize loop helpers
        self.init_alter_step_size_loop_funs()

        # initialize model if it exists, and if so, use it to get initial parameters
        if self._model is not None:
            rng_key, rng_key_init = random.split(rng_key)
            initial_params, self._potential_fn, self._postprocess_fn = utils.init_model(
                self._model, rng_key_init, model_args, model_kwargs
            )
        
        # initialize the state of the autostep sampler
        initial_state = self.init_state(initial_params, rng_key)

        return jax.device_put(initial_state)

    @property
    def model(self):
        return self._model
    
    def get_diagnostics_str(self, state):
        return "base_step_size {:.2e}. mean_acc_prob={:.2f}".format(
            state.base_step_size, state.stats.adapt_stats.mean_acc_prob
        )

    def postprocess_fn(self, model_args, model_kwargs):
        if self._postprocess_fn is None:
            return util.identity
        return self._postprocess_fn(*model_args, **model_kwargs)
    
    @abstractmethod
    def update_log_joint(self, state):
        """
        Compute the log joint density for all variables, i.e., including auxiliary.
        This should also update the gradient of the log joint density whenever the
        these are part of the state of the underlying involutive sampler.

        :param state: Current state.
        :return: Updated state.
        """
        raise NotImplementedError

    @abstractmethod
    def refresh_aux_vars(self, rng_key, state):
        """
        Refresh auxiliary variables required by the underlying involutive MCMC method.

        :param rng_key: Random number generator key.
        :param state: Current state.
        :return: State with updated auxiliary variables.
        """
        raise NotImplementedError
        
    @staticmethod
    @abstractmethod
    def involution_main(step_size, state, diag_precond):
        """
        Apply the main part of the involution. This is usually the part that 
        modifies the variables of interests.

        :param step_size: Step size to use in the involutive transformation.
        :param state: Current state.
        :param diag_precond: A vector representing a diagonal preconditioning matrix.
        :return: Updated state.
        """
        raise NotImplementedError
    
    @staticmethod
    @abstractmethod
    def involution_aux(state):
        """
        Apply the auxiliary part of the involution. This is usually the part that
        is not necessary to implement for the respective involutive MCMC algorithm
        to work correctly (e.g., momentum flip in HMC).
        Note: it is assumed that the augmented target is invariant to this transformation.

        :param state: Current state.
        :return: Updated state.
        """
        raise NotImplementedError

    def sample(self, state, model_args, model_kwargs):
        # refresh auxiliary variables (e.g., momentum), update the log joint 
        # density, and finally check if the latter is finite
        state = self.update_log_joint(self.refresh_aux_vars(state))
        utils.checkified_is_finite(state.log_joint)

        # build a (possibly randomized) diagonal preconditioner
        rng_key, precond_key, selector_key = random.split(state.rng_key, 3)
        state = state._replace(rng_key = rng_key) # always update the state with the modified key!
        diag_precond = self.preconditioner.build_diag_precond(
            state.estimated_std_devs, precond_key)

        # draw selector parameters
        selector_params = self.selector.draw_parameters(selector_key)

        # forward step size search
        # jax.debug.print("fwd autostep: init_log_joint={i}", ordered=True, i=state.log_joint)
        state, fwd_exponent = self.auto_step_size(
            state, selector_params, diag_precond)
        fwd_step_size = utils.step_size(state.base_step_size, fwd_exponent)
        proposed_state = self.update_log_joint(self.involution_main(
            fwd_step_size, state, diag_precond))
        # jax.debug.print(
        #     "fwd done: step_size={s}, init_log_joint={l}, next_log_joint={ln}, log_joint_diff={ld}",
        #     ordered=True, s=fwd_step_size, l=state.log_joint, ln=proposed_state.log_joint, 
        #     ld=proposed_state.log_joint-state.log_joint)

        # backward step size search
        prop_state_flip = self.involution_aux(proposed_state) # don't recompute log_joint because we assume inv_aux leaves it invariant
        # jax.debug.print("bwd begin", ordered=True)
        prop_state_flip, bwd_exponent = self.auto_step_size(
            prop_state_flip, selector_params, diag_precond)
        reversibility_passed = fwd_exponent == bwd_exponent

        # Metropolis-Hastings step
        log_joint_diff = proposed_state.log_joint - state.log_joint
        acc_prob = lax.clamp(0., reversibility_passed * lax.exp(log_joint_diff), 1.)
        # jax.debug.print(
        #     "bwd done: reversibility_passed={r}, acc_prob={a}", ordered=True,
        #     r=reversibility_passed, a=acc_prob)
        rng_key, accept_key = random.split(prop_state_flip.rng_key) # note: this is the state with the "freshest" rng_key

        # build the next state depending on the MH outcome
        next_state = lax.cond(
            random.bernoulli(accept_key, acc_prob),
            utils.next_state_accepted,
            utils.next_state_rejected,
            (state, proposed_state, prop_state_flip, rng_key)
        )

        # collect statistics
        bwd_step_size = utils.step_size(state.base_step_size, bwd_exponent)
        avg_fwd_bwd_step_size = 0.5 * (fwd_step_size + bwd_step_size)
        new_stats = statistics.record_post_sample_stats(
            next_state.stats, avg_fwd_bwd_step_size, acc_prob,
            jax.flatten_util.ravel_pytree(getattr(next_state, self.sample_field))[0]
        )
        next_state = next_state._replace(stats = new_stats)

        # maybe adapt
        next_state = self.adapt(next_state)

        return next_state

    def auto_step_size(self, state, selector_params, diag_precond):
        init_log_joint = state.log_joint # Note: assumes the log joint value is up to date!
        next_state = self.update_log_joint(self.involution_main(
            state.base_step_size, state, diag_precond))
        next_log_joint = next_state.log_joint
        state = utils.copy_state_extras(next_state, state) # update state's stats and rng_key

        # try shrinking (no-op if selector decides not to shrink)
        # note: we call the output of this `state` because it should equal the 
        # initial state except for extra fields -- stats, rng_key -- which we
        # want to update
        state, shrink_exponent = self.shrink_step_size(
            state, selector_params, next_log_joint, init_log_joint, diag_precond)

        # try growing (no-op if selector decides not to grow)
        state, grow_exponent = self.grow_step_size(
            state, selector_params, next_log_joint, init_log_joint, diag_precond)

        # check only one route was taken
        utils.checkified_is_zero(shrink_exponent * grow_exponent)

        return state, shrink_exponent + grow_exponent
    
    def shrink_step_size(self, state, selector_params, next_log_joint, init_log_joint, diag_precond):
        exponent = 0
        (state, exponent, *extra,) = lax.while_loop(
            self.shrink_step_size_cond_fun,
            self.shrink_step_size_body_fun,
            (state, exponent, next_log_joint, init_log_joint, 
             selector_params, diag_precond)
        )
        return state, exponent
    
    def grow_step_size(self, state, selector_params, next_log_joint, init_log_joint, diag_precond):
        exponent = 0        
        (state, exponent, *extra,) = lax.while_loop(
            self.grow_step_size_cond_fun,
            self.grow_step_size_body_fun,
            (state, exponent, next_log_joint, init_log_joint, 
             selector_params, diag_precond)
        )

        # deduct 1 step to avoid cliffs, but only if we actually entered the loop
        exponent = lax.cond(exponent > 0, lambda e: e-1, util.identity, exponent)
        return state, exponent
    
    def adapt(self, state):
        """
        Round-based adaptation, as described in Biron-Lattes et al. (2024).

        Currently this updates `base_step_size` and `estimated_std_devs`.
        At the end, it builds a fresh recorder `state.stats`.

        :param state: Current state.
        :return: 
        """
        stats = state.stats
        round = utils.current_round(stats.n_samples)
        new_base_step_size, new_estimated_std_devs, new_adapt_stats = lax.cond(
            jnp.logical_and(
                round <= self.adapt_rounds,                              # are we still adapting?
                stats.n_samples == utils.last_sample_idx_in_round(round) # are we at the end of a round?
            ),
            lambda t: (
                t[2].mean_step_size, 
                jnp.where(
                    t[2].vars_flat > 0,
                    lax.sqrt(t[2].vars_flat),
                    t[1] # use the old estimated std dev in case of 0 (which is initialized as ones)
                ),
                statistics.empty_adapt_stats_recorder(t[2])
            ),
            util.identity,
            (state.base_step_size, state.estimated_std_devs, stats.adapt_stats)
        )
        new_stats = stats._replace(adapt_stats = new_adapt_stats)
        state = state._replace(
            base_step_size = new_base_step_size, estimated_std_devs = new_estimated_std_devs,
            stats = new_stats
        )
        jax.debug.print(
            "n_samples={n},round={r},sample_idx={s},base_step_size={b}, estimated_std_devs={e}", ordered=True,
            n=stats.n_samples,r=round,s=stats.adapt_stats.sample_idx,
            b=state.base_step_size, e=state.estimated_std_devs)
        return state

