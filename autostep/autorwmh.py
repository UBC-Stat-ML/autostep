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
        tempered_potential = None,
        init_base_step_size = 1.0,
        selector = selectors.SymmetricSelector(),
        preconditioner = preconditioning.MixDiagonalPreconditioner(),
        init_inv_temp = None
    ):
        self._model = model
        self._potential_fn = potential_fn
        self.tempered_potential = tempered_potential
        self._postprocess_fn = None
        self.init_base_step_size = init_base_step_size
        self.selector = selector
        self.preconditioner = preconditioner
        self.init_inv_temp = (
            None if init_inv_temp is None else jnp.array(init_inv_temp)
        )

    def update_log_joint(self, state):
        x, v_flat, *_ = state
        new_temp_pot = self.tempered_potential(x, state.inv_temp)
        new_log_joint = -new_temp_pot - utils.std_normal_potential(v_flat)
        new_stats = statistics.increase_n_pot_evals_by_one(state.stats)
        return state._replace(log_joint = new_log_joint, stats = new_stats)
    
    def refresh_aux_vars(self, state):
        rng_key, v_key = random.split(state.rng_key)
        v_flat = random.normal(v_key, jnp.shape(state.v_flat))
        return state._replace(v_flat = v_flat, rng_key = rng_key)
    
    def involution_main(self, step_size, state, precond_state):
        x_flat, unravel_fn = flatten_util.ravel_pytree(state.x)
        if jnp.ndim(precond_state.var_chol_tril) == 2:
            prec_v_flat = precond_state.var_chol_tril @ state.v_flat
        else:
            prec_v_flat = precond_state.var_chol_tril * state.v_flat
        x_new = unravel_fn(x_flat + step_size * prec_v_flat)
        return state._replace(x = x_new)
    
    def involution_aux(self, step_size, state, precond_state):
        return state._replace(v_flat = -state.v_flat)

