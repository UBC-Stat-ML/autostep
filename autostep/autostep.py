from abc import ABCMeta, abstractmethod
from collections import namedtuple
from functools import partial

import jax
from jax.experimental import checkify
from jax import flatten_util
from jax import lax
from jax import numpy as jnp
from jax import random

from numpyro import infer
from numpyro import util

from autostep import initialization
from autostep import preconditioning
from autostep import selectors
from autostep import statistics
from autostep import tempering
from autostep import utils

AutoStepState = namedtuple(
    "AutoStepState",
    [
        "x",
        "p_flat",
        "log_lik",
        "log_posterior",
        "log_joint",
        "rng_key",
        "stats",
        "base_step_size",
        "base_precond_state",
        "inv_temp"
    ],
)
"""
A :func:`~collections.namedtuple` defining the state of autoStep kernels.
It consists of the fields:

 - **x** - the sample field, corresponding to a pytree representing a value in
   unconstrained space.
 - **p_flat** - flattened momentum/velocity vector.
 - **log_lik** - log-likelihood of ``x``.
 - **log_posterior** - log-posterior of ``x``. The log-prior part includes 
   possible log-abs-det-jac values associated with transformations to 
   unconstrained space.
 - **log_joint** - log-posterior plus kinetic energy of ``p_flat``.
 - **rng_key** - random number generator key.
 - **stats** - an ``AutoStepStats`` object.
 - **base_step_size** - the initial step size. Fixed within a round but updated
   at the end of it.
 - **base_precond_state** - an instance of ``PreconditionerState``. Fixed within a 
   round but updated at the end of it.
 - **inv_temp**: optional inverse temperature parameter to target an annealed
   version of a model's distribution.
"""

class AutoStep(infer.mcmc.MCMCKernel, metaclass=ABCMeta):

    # surprisingly, you *can* have an __init__ method in an ABC class; see
    # https://stackoverflow.com/q/72924810/5443023
    def __init__(
        self,
        model=None,
        potential_fn=None,
        logprior_and_loglik = None,
        init_base_step_size = 1.0,
        selector = selectors.SymmetricSelector(),
        preconditioner = preconditioning.FixedDiagonalPreconditioner(),
        init_inv_temp = None,
        initialization_settings = None
    ):
        self._model = model
        self._potential_fn = potential_fn
        self.logprior_and_loglik = logprior_and_loglik
        self._postprocess_fn = None
        self.init_base_step_size = init_base_step_size
        self.selector = selector
        self.preconditioner = preconditioner
        self.init_inv_temp = (
            None if init_inv_temp is None else jnp.array(init_inv_temp)
        )
        self.initialization_settings = initialization_settings

    def init_alter_step_size_loop_funs(self):
        self.shrink_step_size_cond_fun = utils.gen_alter_step_size_cond_fun(
            self.selector.should_shrink, self.selector.max_n_iter
        )
        self.shrink_step_size_body_fun = utils.gen_alter_step_size_body_fun(
            self, -1
        )
        self.grow_step_size_cond_fun = utils.gen_alter_step_size_cond_fun(
            self.selector.should_grow, self.selector.max_n_iter
        )
        self.grow_step_size_body_fun = utils.gen_alter_step_size_body_fun(
            self, 1
        )
    
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
            jnp.array(0.), # Note: not the actual loglik; needs to be updated 
            jnp.array(0.), # Note: not the actual logposterior; needs to be updated 
            jnp.array(0.), # Note: not the actual log joint value; needs to be updated 
            rng_key,
            statistics.make_stats_recorder(
                sample_field_flat_shape, self.preconditioner
            ),
            jnp.array(self.init_base_step_size),
            preconditioning.init_base_precond_state(
                sample_field_flat_shape, self.preconditioner
            ),
            None if self.init_inv_temp is None else jnp.array(self.init_inv_temp)
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
        if self.logprior_and_loglik is None:
            if self._model is not None:
                rng_key, rng_key_init = random.split(rng_key)
                (
                    new_params, 
                    self._potential_fn, 
                    self._postprocess_fn
                ) = initialization.init_model(
                    self._model, rng_key_init, model_args, model_kwargs
                )
                if initial_params is None:
                    initial_params = new_params
                
                # initialize the logprior_and_loglik function
                self.logprior_and_loglik = partial(
                    tempering.model_logprior_and_loglik, 
                    self._model, 
                    model_args, 
                    model_kwargs
                )
            elif self._potential_fn is not None:
                self.logprior_and_loglik = lambda x: (-self._potential_fn(x), 0.)
            else:
                raise(ValueError(
                    "You need to provide a model, a logprior_and_loglik, or a " \
                    "potential function"
                ))
        
        # maybe optimize the initial parameters
        if self.initialization_settings is not None:
            initial_params = initialization.optimize_init_params(
                self.logprior_and_loglik, 
                initial_params, 
                self.init_inv_temp,
                self.initialization_settings
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
        return "base_step {:.2e}, rev_rate={:.2f}, acc_prob={:.2f}".format(
            state.base_step_size,
            state.stats.adapt_stats.rev_rate, 
            state.stats.adapt_stats.mean_acc_prob
        )

    def postprocess_fn(self, model_args, model_kwargs):
        if self._postprocess_fn is None:
            return util.identity
        return self._postprocess_fn(*model_args, **model_kwargs)
    
    def kinetic_energy(self, state, precond_state):
        """
        Computes the potential energy for any augmented variables whose density
        is not invariant to the involution. The default implementation assumes
        that the distribution of all auxiliary variables is invariant under the
        involution (this is true for autoRWMH, for example).

        :param state: Current state.
        :param precond_state: Preconditioner state.
        :return: Kinetic energy.
        """
        return jnp.zeros_like(state.p_flat[0])

    def update_log_joint(self, state, precond_state):
        """
        Update the log-likelihood, log-posterior, and log-joint.

        :param state: Current state.
        :param precond_state: Preconditioner state.
        :return: Updated state.
        """
        # get new kinetic energy
        new_kinetic_energy = self.kinetic_energy(state, precond_state)

        # get logprior, loglik and tempered potential
        new_log_prior, new_log_lik = self.logprior_and_loglik(state.x)
        new_temp_pot = tempering.tempered_potential_from_logprior_and_loglik(
            new_log_prior, new_log_lik, state.inv_temp
        )
        
        # replace state with updated values and return
        return state._replace(
            log_lik = new_log_lik,
            log_posterior = new_log_prior + new_log_lik,
            log_joint = -(new_temp_pot + new_kinetic_energy)
        )

    @abstractmethod
    def refresh_aux_vars(self, rng_key, state, precond_state):
        """
        Refresh auxiliary variables required by the underlying involutive MCMC method.

        :param rng_key: Random number generator key.
        :param state: Current state.
        :param precond_state: Preconditioner state.
        :return: State with updated auxiliary variables.
        """
        raise NotImplementedError
        
    @abstractmethod
    def involution_main(self, step_size, state, precond_state):
        """
        Apply the main part of the involution. This is usually the part that 
        modifies the variables of interests.

        :param step_size: Step size to use in the involutive transformation.
        :param state: Current state.
        :param precond_state: Preconditioner state.
        :return: Updated state.
        """
        raise NotImplementedError
    
    @abstractmethod
    def involution_aux(self, step_size, state, precond_state):
        """
        Apply the auxiliary part of the involution. This is usually the part that
        is not necessary to implement for the respective involutive MCMC algorithm
        to work correctly (e.g., momentum flip in HMC).
        Note: it is assumed that the augmented target is invariant to this transformation.

        :param step_size: Step size to use in the involutive transformation.
        :param state: Current state.
        :param precond_state: Preconditioner state.
        :return: Updated state.
        """
        raise NotImplementedError
    
    def sample(self, state, model_args, model_kwargs):
        # build a (possibly randomized) preconditioner
        rng_key, precond_key, selector_key = random.split(state.rng_key, 3)
        state = state._replace(rng_key = rng_key) # always update the state with the modified key!
        precond_state = self.preconditioner.maybe_alter_precond_state(
            state.base_precond_state, precond_key
        )
        # jax.debug.print(
        #     "precond_state: var={v}   chol_tril={c}   inv_fac={i}", 
        #     ordered=True, 
        #     v=precond_state.var,
        #     c=precond_state.var_chol_tril,
        #     i=precond_state.inv_var_triu_factor
        # )

        # refresh auxiliary variables (e.g., momentum), update the log joint 
        # density, and finally check if the latter is finite
        # Checker needs checkifying twice for some reason
        state = self.update_log_joint(
            self.refresh_aux_vars(state, precond_state), precond_state
        )
        checkify.checkify(utils.checkified_is_finite)(state.log_joint)[0].throw()

        # draw selector parameters
        selector_params = self.selector.draw_parameters(selector_key)

        # forward step size search
        # jax.debug.print("fwd autostep: init_log_joint={i}, init_x={x}, init_v={v}", ordered=True, i=state.log_joint, x=state.x, v=state.p_flat)
        state, fwd_exponent = self.auto_step_size(
            state, selector_params, precond_state
        )
        fwd_step_size = utils.step_size(state.base_step_size, fwd_exponent)
        proposed_state = self.update_log_joint(
            self.involution_main(fwd_step_size, state, precond_state),
            precond_state
        )
        # jax.debug.print(
        #     "fwd done: step_size={s}, init_log_joint={l}, next_log_joint={ln}, log_joint_diff={ld}, prop_x={x}, prop_v={v}",
        #     ordered=True, s=fwd_step_size, l=state.log_joint, ln=proposed_state.log_joint, 
        #     ld=proposed_state.log_joint-state.log_joint, x=proposed_state.x, v=proposed_state.p_flat)

        # backward step size search
        # don't recompute log_joint for flipped state because we assume inv_aux 
        # leaves it invariant
        prop_state_flip = self.involution_aux(
            fwd_step_size, proposed_state, precond_state
        )
        # jax.debug.print("bwd begin", ordered=True)
        prop_state_flip, bwd_exponent = self.auto_step_size(
            prop_state_flip, selector_params, precond_state
        )
        reversibility_passed = fwd_exponent == bwd_exponent

        # Metropolis-Hastings step
        # note: when the magnitude of log_joint is ~ 1e8, the difference in
        # Float32 precision of two floats next to each other can be >> 1.
        # For this reason, we consider 2 consecutive floats to be equal.
        log_joint_diff = utils.numerically_safe_diff(
            state.log_joint, proposed_state.log_joint
        )
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
            next_state.stats, avg_fwd_bwd_step_size, acc_prob, reversibility_passed,
            jax.flatten_util.ravel_pytree(getattr(next_state, self.sample_field))[0]
        )
        next_state = next_state._replace(stats = new_stats)

        # maybe adapt
        next_state = self.adapt(next_state)

        return next_state

    def auto_step_size(self, state, selector_params, precond_state):
        init_log_joint = state.log_joint # Note: assumes the log joint value is up to date!
        next_state = self.update_log_joint(
            self.involution_main(state.base_step_size, state, precond_state),
            precond_state
        )
        next_log_joint = next_state.log_joint
        state = utils.copy_state_extras(next_state, state) # update state's stats and rng_key

        # try shrinking (no-op if selector decides not to shrink)
        # note: we call the output of this `state` because it should equal the 
        # initial state except for extra fields -- stats, rng_key -- which we
        # want to update
        state, shrink_exponent = self.shrink_step_size(
            state, selector_params, next_log_joint, init_log_joint, precond_state
        )

        # try growing (no-op if selector decides not to grow)
        state, grow_exponent = self.grow_step_size(
            state, selector_params, next_log_joint, init_log_joint, precond_state
        )

        # check only one route was taken
        # Needs checkifying twice for some reason
        checkify.checkify(utils.checkified_is_zero)(
            shrink_exponent * grow_exponent
        )[0].throw()

        return state, shrink_exponent + grow_exponent
    
    def shrink_step_size(
            self, 
            state, 
            selector_params, 
            next_log_joint, 
            init_log_joint, 
            precond_state
        ):
        exponent = 0
        state, exponent, *_ = lax.while_loop(
            self.shrink_step_size_cond_fun,
            self.shrink_step_size_body_fun,
            (state, exponent, next_log_joint, init_log_joint, 
             selector_params, precond_state)
        )
        return state, exponent
    
    def grow_step_size(self, state, selector_params, next_log_joint, init_log_joint, precond_state):
        exponent = 0        
        state, exponent, *_ = lax.while_loop(
            self.grow_step_size_cond_fun,
            self.grow_step_size_body_fun,
            (state, exponent, next_log_joint, init_log_joint, 
             selector_params, precond_state)
        )

        # deduct 1 step to avoid cliffs, but only if we actually entered the loop
        exponent = lax.cond(exponent > 0, lambda e: e-1, util.identity, exponent)
        return state, exponent
    
    def adapt(self, state, force=False):
        """
        Round-based adaptation, as described in Biron-Lattes et al. (2024).

        Currently, this updates `base_step_size` and `base_precond_state`.
        At the end, it empties the `AutoStepAdaptStats` recorder.

        :param state: Current state.
        :param force: Should adaptation be forced regardless of round status?
        :return: Possibly updated state.
        """
        stats = state.stats
        round = utils.current_round(stats.n_samples)
        new_base_step_size, new_base_precond_state, new_adapt_stats = lax.cond(
            force or jnp.logical_and(
                round <= self.adapt_rounds,              # are we still adapting?
                stats.adapt_stats.sample_idx == 2**round # are we at the end of a round?
            ),
            partial(statistics.update_sampler_params, self.selector),
            util.identity,
            (state.base_step_size, state.base_precond_state, stats.adapt_stats)
        )
        new_stats = stats._replace(adapt_stats = new_adapt_stats)
        state = state._replace(
            base_step_size = new_base_step_size, 
            base_precond_state = new_base_precond_state,
            stats = new_stats
        )
        # jax.debug.print(
        #     """
        #     n_samples={n}, round={r}, sample_idx={s},
        #     base_step_size={b}, base_precond_state={e},
        #     mean_step_size={ms}, mean_acc_prob={ma},
        #     sample_mean={m}, sample_var={v}""", ordered=True,
        #     n=stats.n_samples,r=round,s=stats.adapt_stats.sample_idx,
        #     b=state.base_step_size, e=state.base_precond_state,
        #     ms=new_stats.adapt_stats.mean_step_size, ma=new_stats.adapt_stats.mean_acc_prob,
        #     m=new_stats.adapt_stats.sample_mean,v=new_stats.adapt_stats.sample_var)
        return state
