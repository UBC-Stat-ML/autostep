from jax import flatten_util
from jax import numpy as jnp
from jax import random

from autostep import autostep
from autostep import preconditioning
from autostep import selectors

class AutoRWMH(autostep.AutoStep):

    def __init__(
        self,
        model=None,
        potential_fn=None,
        logprior_and_loglik = None,
        init_base_step_size = 1.0,
        selector = selectors.SymmetricSelector(),
        preconditioner = preconditioning.FixedDiagonalPreconditioner(),
        init_inv_temp = None,
        n_iter_opt_init_params = 0
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
        self.n_iter_opt_init_params = n_iter_opt_init_params
   
    def refresh_aux_vars(self, state, precond_state):
        rng_key, v_key = random.split(state.rng_key)
        p_flat = random.normal(v_key, jnp.shape(state.p_flat))
        return state._replace(p_flat = p_flat, rng_key = rng_key)
    
    def involution_main(self, step_size, state, precond_state):
        x_flat, unravel_fn = flatten_util.ravel_pytree(state.x)
        if jnp.ndim(precond_state.var_chol_tril) == 2:
            prec_p_flat = precond_state.var_chol_tril @ state.p_flat
        else:
            prec_p_flat = precond_state.var_chol_tril * state.p_flat
        x_new = unravel_fn(x_flat + step_size * prec_p_flat)
        return state._replace(x = x_new)
    
    def involution_aux(self, step_size, state, precond_state):
        return state._replace(p_flat = -state.p_flat)

