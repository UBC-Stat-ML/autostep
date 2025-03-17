from jax import flatten_util
from jax import numpy as jnp
from jax import random

from autostep import autostep
from autostep import preconditioning
from autostep import selectors
from autostep import statistics
from autostep import utils

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

    def update_log_joint(self, state):
        x, v_flat, *extra = state
        new_log_joint = -self._potential_fn(x) - utils.std_normal_potential(v_flat)
        new_stats = statistics.increase_n_pot_evals_by_one(state.stats)
        return state._replace(log_joint = new_log_joint, stats = new_stats)
    
    def refresh_aux_vars(self, state):
        rng_key, v_key = random.split(state.rng_key)
        v_flat = random.normal(v_key, jnp.shape(state.v_flat))
        return state._replace(v_flat = v_flat, rng_key = rng_key)
    
    def involution_main(self, step_size, state, precond_array):
        x_flat, unravel_fn = flatten_util.ravel_pytree(state.x)
        prec_v_flat = utils.apply_precond(precond_array, state.v_flat)
        x_new = unravel_fn(x_flat + step_size * prec_v_flat)
        return state._replace(x = x_new)
    
    def involution_aux(self, step_size, state, precond_array):
        return state._replace(v_flat = -state.v_flat)

