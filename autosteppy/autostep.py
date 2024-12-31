from abc import ABCMeta, abstractmethod

import jax
from jax import lax
from jax import random
from numpyro import infer
from numpyro import util

from autosteppy import utils
from autosteppy import statistics

class AutoStep(infer.mcmc.MCMCKernel, metaclass=ABCMeta):

    def init_alter_step_size_loop_funs(self):
        self.shrink_step_size_cond_fun = utils.gen_alter_step_size_cond_fun(
            self.selector.should_shrink)
        self.shrink_step_size_body_fun = utils.gen_alter_step_size_body_fun(self, -1)
        self.grow_step_size_cond_fun = utils.gen_alter_step_size_cond_fun(
            self.selector.should_grow)
        self.grow_step_size_body_fun = utils.gen_alter_step_size_body_fun(self, 1)

    @property
    def model(self):
        return self._model
    
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
    def involution_main(step_size, state):
        """
        Apply the main part of the involution. This is usually the part that 
        modifies the variables of interests.

        :param state: Current state.
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

        # draw selector parameters
        rng_key, selector_key = random.split(state.rng_key)
        state = state._replace(rng_key = rng_key) # always update the state with the modified key
        selector_params = self.selector.draw_parameters(selector_key)

        # forward step size search
        # jax.debug.print("fwd autostep: init_log_joint={i}", ordered=True, i=state.log_joint)
        state, fwd_exponent = self.auto_step_size(state, selector_params)
        fwd_step_size = utils.step_size(self._base_step_size, fwd_exponent)
        proposed_state = self.update_log_joint(self.involution_main(
            fwd_step_size, state))
        # jax.debug.print(
        #     "fwd done: step_size={s}, init_log_joint={l}, next_log_joint={ln}, log_joint_diff={ld}",
        #     ordered=True, s=fwd_step_size, l=state.log_joint, ln=proposed_state.log_joint, 
        #     ld=proposed_state.log_joint-state.log_joint)

        # backward step size search
        prop_state_flip = self.involution_aux(proposed_state) # don't recompute log_joint because we assume inv_aux leaves it invariant
        # jax.debug.print("bwd begin", ordered=True)
        prop_state_flip, bwd_exponent = self.auto_step_size(prop_state_flip, selector_params)
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

        # collect statistics and return
        bwd_step_size = utils.step_size(self._base_step_size, bwd_exponent)
        avg_fwd_bwd_step_size = 0.5 * (fwd_step_size + bwd_step_size)
        new_stats = statistics.record_post_sample_stats(
            next_state.stats, avg_fwd_bwd_step_size, acc_prob
        )
        return next_state._replace(stats = new_stats)

    def auto_step_size(self, state, selector_params):
        init_log_joint = state.log_joint # Note: assumes the log joint value is up to date!
        next_state = self.update_log_joint(self.involution_main(self._base_step_size, state))
        next_log_joint = next_state.log_joint
        state = utils.copy_state_extras(next_state, state) # update state's stats and rng_key

        # try shrinking (no-op if selector decides not to shrink)
        # note: we call the output of this `state` because it should equal the 
        # initial state except for extra fields -- stats, rng_key -- which we
        # want to update
        state, shrink_exponent = self.shrink_step_size(state, selector_params, next_log_joint, init_log_joint)

        # try growing (no-op if selector decides not to grow)
        state, grow_exponent = self.grow_step_size(state, selector_params, next_log_joint, init_log_joint)

        # check only one route was taken
        utils.checkified_is_zero(shrink_exponent * grow_exponent)

        return state, shrink_exponent + grow_exponent
    
    def shrink_step_size(self, state, selector_params, next_log_joint, init_log_joint):
        exponent = 0
        (state, exponent, *extra,) = lax.while_loop(
            self.shrink_step_size_cond_fun,
            self.shrink_step_size_body_fun,
            (state, exponent, next_log_joint, init_log_joint, selector_params, self._base_step_size)
        )
        return state, exponent
    
    def grow_step_size(self, state, selector_params, next_log_joint, init_log_joint):
        exponent = 0        
        (state, exponent, *extra,) = lax.while_loop(
            self.grow_step_size_cond_fun,
            self.grow_step_size_body_fun,
            (state, exponent, next_log_joint, init_log_joint, selector_params, self._base_step_size)
        )

        # deduct 1 step to avoid cliffs, but only if we actually entered the loop
        exponent = lax.cond(exponent > 0, lambda e: e-1, util.identity, exponent)
        return state, exponent

