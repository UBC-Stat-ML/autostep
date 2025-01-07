from collections import namedtuple

import jax

from jax import flatten_util
from jax import numpy as jnp
from jax import random

from autostep import autostep
from autostep import preconditioning
from autostep import selectors
from autostep import statistics
from autostep import utils

AutoRWMHState = namedtuple(
    "AutoRWMHState",
    [
        "x",
        "v_flat",
        "log_joint",
        "rng_key",
        "stats",
        "base_step_size",
        "estimated_std_devs"
    ],
)
"""
A :func:`~collections.namedtuple` defining the state of the ``AutoRWMH`` kernel.
It consists of the fields:

 - **x** - the sample field.
 - **v_flat** - flattened velocity vector.
 - **log_joint** - joint log density of ``(x,v_flat)``.
 - **rng_key** - random number generator key.
 - **stats** - an ``AutoStepStats`` object.
 - **base_step_size** - the initial step size. Fixed within a round but updated
   at the end of each adaptation round.
 - **estimated_std_devs** - the current best guess of the standard deviations of the
   flattened sample field. Fixed within a round but updated at the end of each 
   adaptation round.
"""

class AutoRWMH(autostep.AutoStep):

    def __init__(
        self,
        model=None,
        potential_fn=None,
        selector = selectors.SymmetricSelector(),
        preconditioner = preconditioning.MixDiagonalPreconditioner()
    ):
        self._model = model
        self._potential_fn = potential_fn
        self._postprocess_fn = None
        self.selector = selector
        self.preconditioner = preconditioner

    @staticmethod
    def init_state(initial_params, rng_key):
        sample_field_flat_shape = jnp.shape(flatten_util.ravel_pytree(initial_params)[0])
        return AutoRWMHState(
            initial_params,
            jnp.zeros(sample_field_flat_shape),
            0., # Note: not the actual log joint value; needs to be updated 
            rng_key,
            statistics.make_stats_recorder(sample_field_flat_shape),
            1.0,
            jnp.ones(sample_field_flat_shape)
        )

    @property
    def sample_field(self):
        return "x"

    def update_log_joint(self, state):
        x, v_flat, *extra = state
        new_log_joint = -self._potential_fn(x) - utils.std_normal_potential(v_flat)
        new_stats = statistics.increase_n_pot_evals_by_one(state.stats)
        return state._replace(log_joint = new_log_joint, stats = new_stats)
    
    def refresh_aux_vars(self, state):
        rng_key, v_key = random.split(state.rng_key)
        v_flat = random.normal(v_key, jnp.shape(state.v_flat))
        return state._replace(v_flat = v_flat, rng_key = rng_key)
    
    @staticmethod
    def involution_main(step_size, state, diag_precond):
        x_flat, unravel_fn = flatten_util.ravel_pytree(state.x)
        x_new = unravel_fn(x_flat + step_size * diag_precond * state.v_flat)
        return state._replace(x = x_new)
    
    @staticmethod
    def involution_aux(state):
        return state._replace(v_flat = -state.v_flat)

