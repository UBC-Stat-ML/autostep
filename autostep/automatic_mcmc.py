from abc import ABCMeta, abstractmethod
from collections import namedtuple
from functools import partial

from jax import flatten_util
from jax import numpy as jnp
from jax import random

from numpyro import infer
from numpyro import util

from autostep import initialization
from autostep import preconditioning
from autostep import statistics
from autostep import tempering
from autostep import utils

AutoMCMCState = namedtuple(
    "AutoMCMCState",
    [
        "x",
        "p_flat",
        "log_prior",
        "log_lik",
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
 - **log_prior** - log-prior density of ``x``. Includes possible log-abs-det-jac 
   values associated with transformations to unconstrained space.
 - **log_lik** - log-likelihood of ``x``.
 - **log_joint** - log-prior plus log-likelihood plus log density of ``p_flat``.
 - **rng_key** - random number generator key.
 - **stats** - an ``AutoStepStats`` object.
 - **base_step_size** - the initial step size. Fixed within a round but updated
   at the end of it.
 - **base_precond_state** - an instance of ``PreconditionerState``. Fixed within a 
   round but updated at the end of it.
 - **inv_temp**: optional inverse temperature parameter to target an annealed
   version of a model's distribution.
"""

class AutomaticMCMC(infer.mcmc.MCMCKernel, metaclass=ABCMeta):
    """
    Interface for MCMC samplers that can be understood as automatically
    tuning their key parameters at each step. Any other hyperparameter is 
    assumed to be tunable via round-based adaptation.

    Additionally, we build native support for tempered (or annealed) sampling
    of target densities by including the (inverse) temperature parameter into
    the state of the sampler. This permits adjusting the parameter as the
    sampling progresses, enabling further implementation of tempering-based
    ensemble MCMC algorithms.
    """

    # surprisingly, you *can* have an __init__ method in an ABC class; see
    # https://stackoverflow.com/q/72924810/5443023
    def __init__(
        self,
        model=None,
        potential_fn=None,
        logprior_and_loglik = None,
        init_base_step_size = 1.0,
        preconditioner = preconditioning.FixedDiagonalPreconditioner(),
        init_inv_temp = None,
        initialization_settings = None
    ):
        self._model = model
        self._potential_fn = potential_fn
        self.logprior_and_loglik = logprior_and_loglik
        self._postprocess_fn = None
        self.init_base_step_size = init_base_step_size
        self.preconditioner = preconditioner
        self.init_inv_temp = (
            None if init_inv_temp is None else jnp.array(init_inv_temp)
        )
        self.initialization_settings = initialization_settings
    
    def init_state(self, initial_params, rng_key):
        """
        Initialize the state of the sampler.

        :param initial_params: Initial values for the latent parameters.
        :param rng_key: The PRNG key that the sampler should use for simulation.
        :return: The initial state of the sampler.
        """
        sample_field_flat_shape = jnp.shape(flatten_util.ravel_pytree(initial_params)[0])
        return AutoMCMCState(
            initial_params,
            jnp.zeros(sample_field_flat_shape),
            jnp.array(0.), # Note: not the actual log-prior; needs to be updated 
            jnp.array(0.), # Note: not the actual log-lik; needs to be updated 
            jnp.array(0.), # Note: not the actual log-joint; needs to be updated 
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
        Carry out additional initializations not required by all 
        :class:`AutomaticMCMC` kernels.
        """
        return state

    # note: this is called by the enclosing numpyro.infer.MCMC object
    def init(self, rng_key, num_warmup, initial_params, model_args, model_kwargs):
        # determine number of adaptation rounds
        self.adapt_rounds = utils.n_warmup_to_adapt_rounds(num_warmup)

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

        # initialize the state of the sampler
        initial_state = self.init_state(initial_params, rng_key)

        # carry out any other initialization required by the kernel
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
        Update the log-prior, log-likelihood, and log-joint.

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
            log_prior = new_log_prior,
            log_lik = new_log_lik,            
            log_joint = -(new_temp_pot + new_kinetic_energy)
        )
    
    def is_time_to_adapt(self, stats):
        round = utils.current_round(stats.n_samples)
        last_step = utils.n_steps_in_round(round)
        return jnp.logical_and(
            round <= self.adapt_rounds,               # are we still adapting?
            stats.adapt_stats.sample_idx == last_step # are we at the end of a round?
        )

    @abstractmethod
    def adapt(self, state, force=False):
        """
        Round-based adaptation.

        :param state: Current state.
        :param force: Should adaptation be forced regardless of round status?
        :return: Possibly updated state.
        """
        raise NotImplementedError
    
