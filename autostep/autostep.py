from abc import ABCMeta, abstractmethod
from collections import namedtuple
from functools import partial

import jax
from jax import flatten_util
from jax import lax
from jax import numpy as jnp
from jax import random

from numpyro import infer
from numpyro import util

from autostep import utils
from autostep import statistics
from autostep import preconditioning

AutoStepState = namedtuple(
    "AutoStepState",
    [
        "x",
        "v_flat",
        "log_joint",
        "rng_key",
        "stats",
        "base_step_size",
        "sqrt_var"
    ],
)
"""
A :func:`~collections.namedtuple` defining the state of autoStep kernels.
It consists of the fields:

 - **x** - the sample field.
 - **v_flat** - flattened velocity vector.
 - **log_joint** - joint log density of ``(x,v_flat)``.
 - **rng_key** - random number generator key.
 - **stats** - an ``AutoStepStats`` object.
 - **base_step_size** - the initial step size. Fixed within a round but updated
   at the end of each adaptation round.
 - **sqrt_var** - the current best guess of the square-root of the (co)variance
   of the flattened sample field. It can be either a vector (diagonal 
   preconditioning) or a matrix (dense preconditioner). Fixed within a round 
   but updated at the end of each adaptation round.
"""

class AutoStep(infer.mcmc.MCMCKernel, metaclass=ABCMeta):

    def init_alter_step_size_loop_funs(self):
        self.shrink_step_size_cond_fun = utils.gen_alter_step_size_cond_fun(
            self.selector.should_shrink)
        self.shrink_step_size_body_fun = utils.gen_alter_step_size_body_fun(self, -1)
        self.grow_step_size_cond_fun = utils.gen_alter_step_size_cond_fun(
            self.selector.should_grow)
        self.grow_step_size_body_fun = utils.gen_alter_step_size_body_fun(self, 1)
    
    def init_state(self, initial_params, rng_key):
        """
        Initialize the state of the sampler.

        :param initial_params: Initial values for the latent parameters.
        :param rng_key: The PRNG key that the sampler should use for simulation.
        :return: The initial state of the sampler.
        """
        sample_field_flat_shape = jnp.shape(flatten_util.ravel_pytree(initial_params)[0])
        return AutoStepState(
            initial_params,
            jnp.zeros(sample_field_flat_shape),
            0., # Note: not the actual log joint value; needs to be updated 
            rng_key,
            statistics.make_stats_recorder(
                sample_field_flat_shape, self.preconditioner
            ),
            1.0,
            utils.init_sqrt_var(sample_field_flat_shape, self.preconditioner)
        )
    
    def init_extras(self, state):
        """
        Carry out additional initializations not required by all autoStep kernels.
        """
        return state

    # note: this is called by the enclosing numpyro.infer.MCMC object
    def init(self, rng_key, num_warmup, initial_params, model_args, model_kwargs):
        # determine number of adaptation rounds
        self.adapt_rounds = utils.n_warmup_to_adapt_rounds(num_warmup)

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

        # carry out any other initialization required by an autoStep kernel
        initial_state = self.init_extras(initial_state)

        return initial_state

    @property
    def sample_field(self):
        return "x"

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
        
    # @staticmethod
    @abstractmethod
    def involution_main(self, step_size, state, precond_array):
        """
        Apply the main part of the involution. This is usually the part that 
        modifies the variables of interests.

        :param step_size: Step size to use in the involutive transformation.
        :param state: Current state.
        :param precond_array: A preconditioning array.
        :return: Updated state.
        """
        raise NotImplementedError
    
    @abstractmethod
    def involution_aux(self, step_size, state, precond_array):
        """
        Apply the auxiliary part of the involution. This is usually the part that
        is not necessary to implement for the respective involutive MCMC algorithm
        to work correctly (e.g., momentum flip in HMC).
        Note: it is assumed that the augmented target is invariant to this transformation.

        :param step_size: Step size to use in the involutive transformation.
        :param state: Current state.
        :param precond_array: A preconditioning array.
        :return: Updated state.
        """
        raise NotImplementedError

    def sample(self, state, model_args, model_kwargs):
        # refresh auxiliary variables (e.g., momentum), update the log joint 
        # density, and finally check if the latter is finite
        state = self.update_log_joint(self.refresh_aux_vars(state))
        utils.checkified_is_finite(state.log_joint)

        # build a (possibly randomized) preconditioner
        rng_key, precond_key, selector_key = random.split(state.rng_key, 3)
        state = state._replace(rng_key = rng_key) # always update the state with the modified key!
        precond_array = self.preconditioner.build_precond(
            state.sqrt_var, precond_key
        )

        # draw selector parameters
        selector_params = self.selector.draw_parameters(selector_key)

        # forward step size search
        # jax.debug.print("fwd autostep: init_log_joint={i}, init_x={x}, init_v={v}", ordered=True, i=state.log_joint, x=state.x, v=state.v_flat)
        state, fwd_exponent = self.auto_step_size(
            state, selector_params, precond_array
        )
        fwd_step_size = utils.step_size(state.base_step_size, fwd_exponent)
        proposed_state = self.update_log_joint(
            self.involution_main(fwd_step_size, state, precond_array)
        )
        # jax.debug.print(
        #     "fwd done: step_size={s}, init_log_joint={l}, next_log_joint={ln}, log_joint_diff={ld}, prop_x={x}, prop_v={v}",
        #     ordered=True, s=fwd_step_size, l=state.log_joint, ln=proposed_state.log_joint, 
        #     ld=proposed_state.log_joint-state.log_joint, x=proposed_state.x, v=proposed_state.v_flat)

        # backward step size search
        # don't recompute log_joint for flipped state because we assume inv_aux 
        # leaves it invariant
        prop_state_flip = self.involution_aux(
            fwd_step_size, proposed_state, precond_array
        )
        # jax.debug.print("bwd begin", ordered=True)
        prop_state_flip, bwd_exponent = self.auto_step_size(
            prop_state_flip, selector_params, precond_array
        )
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

    def auto_step_size(self, state, selector_params, precond_array):
        init_log_joint = state.log_joint # Note: assumes the log joint value is up to date!
        next_state = self.update_log_joint(
            self.involution_main(state.base_step_size, state, precond_array)
        )
        next_log_joint = next_state.log_joint
        state = utils.copy_state_extras(next_state, state) # update state's stats and rng_key

        # try shrinking (no-op if selector decides not to shrink)
        # note: we call the output of this `state` because it should equal the 
        # initial state except for extra fields -- stats, rng_key -- which we
        # want to update
        state, shrink_exponent = self.shrink_step_size(
            state, selector_params, next_log_joint, init_log_joint, precond_array
        )

        # try growing (no-op if selector decides not to grow)
        state, grow_exponent = self.grow_step_size(
            state, selector_params, next_log_joint, init_log_joint, precond_array
        )

        # check only one route was taken
        utils.checkified_is_zero(shrink_exponent * grow_exponent)

        return state, shrink_exponent + grow_exponent
    
    def shrink_step_size(
            self, 
            state, 
            selector_params, 
            next_log_joint, 
            init_log_joint, 
            precond_array
        ):
        exponent = 0
        state, exponent, *_ = lax.while_loop(
            self.shrink_step_size_cond_fun,
            self.shrink_step_size_body_fun,
            (state, exponent, next_log_joint, init_log_joint, 
             selector_params, precond_array)
        )
        return state, exponent
    
    def grow_step_size(self, state, selector_params, next_log_joint, init_log_joint, precond_array):
        exponent = 0        
        state, exponent, *_ = lax.while_loop(
            self.grow_step_size_cond_fun,
            self.grow_step_size_body_fun,
            (state, exponent, next_log_joint, init_log_joint, 
             selector_params, precond_array)
        )

        # deduct 1 step to avoid cliffs, but only if we actually entered the loop
        exponent = lax.cond(exponent > 0, lambda e: e-1, util.identity, exponent)
        return state, exponent
    
    def adapt(self, state, force=False):
        """
        Round-based adaptation, as described in Biron-Lattes et al. (2024).

        Currently, this updates `base_step_size` and `sqrt_var`.
        At the end, it empties the `AutoStepAdaptStats` recorder.

        :param state: Current state.
        :param force: Should adaptation be forced regardless of round status?
        :return: Possibly updated state.
        """
        stats = state.stats
        round = utils.current_round(stats.n_samples)
        new_base_step_size, new_sqrt_var, new_adapt_stats = lax.cond(
            force or jnp.logical_and(
                round <= self.adapt_rounds,              # are we still adapting?
                stats.adapt_stats.sample_idx == 2**round # are we at the end of a round?
            ),
            partial(update_sampler_params, self.preconditioner),
            util.identity,
            (state.base_step_size, state.sqrt_var, stats.adapt_stats)
        )
        new_stats = stats._replace(adapt_stats = new_adapt_stats)
        state = state._replace(
            base_step_size = new_base_step_size, sqrt_var = new_sqrt_var,
            stats = new_stats
        )
        # jax.debug.print(
        #     """
        #     n_samples={n}, round={r}, sample_idx={s},
        #     base_step_size={b}, sqrt_var={e},
        #     mean_step_size={ms}, mean_acc_prob={ma},
        #     means_flat={m}, vars_flat={v}""", ordered=True,
        #     n=stats.n_samples,r=round,s=stats.adapt_stats.sample_idx,
        #     b=state.base_step_size, e=state.sqrt_var,
        #     ms=new_stats.adapt_stats.mean_step_size, ma=new_stats.adapt_stats.mean_acc_prob,
        #     m=new_stats.adapt_stats.means_flat,v=new_stats.adapt_stats.vars_flat)
        return state


def update_sampler_params(preconditioner, args):
    base_step_size, sqrt_var, adapt_stats = args
    new_base_step_size = adapt_stats.mean_step_size
    if preconditioning.is_dense(preconditioner):
        # add exponentially decaying regularization to prevent numerical issues
        # this avoids having to check eigenvalues which would be too costly
        eps = lax.max(
            jax.numpy.finfo(sqrt_var.dtype).eps, 
            jnp.exp(-adapt_stats.sample_idx)
        )
        new_sqrt_var = lax.linalg.cholesky(
            adapt_stats.vars_flat + eps*jnp.eye(*sqrt_var.shape)
        )
    else:
        new_sqrt_var = jnp.where(
            adapt_stats.vars_flat > 0,
            lax.sqrt(adapt_stats.vars_flat),
            sqrt_var # use the old sqrt_var in case of 0 (which is initialized as ones)
        )
    new_adapt_stats = statistics.empty_adapt_stats_recorder(adapt_stats)
    return (new_base_step_size, new_sqrt_var, new_adapt_stats)