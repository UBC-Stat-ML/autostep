from jax import flatten_util
from jax import numpy as jnp
from jax import random

from autostep import autostep

class AutoPCN(autostep.AutoStep):
    """
    Involutive implementation of a finite-dimensional Preconditioned 
    Crank-Nicolson sampler a la [1].

    [1] Cotter, S. L., Roberts, G. O., Stuart, A. M., & White, D. (2013). 
    MCMC methods for functions: Modifying old algorithms to make them faster.
    *Statistical Science*, 424-446.
    """
    # compute kinetic energy for p under N(0, S). Recall that we have U s.t.
    #   UU^T = S^{-1}
    # So the kinetic energy is
    #   0.5 p^T S^{-1} p = 0.5 p^T(UU^T)p = 0.5 v^Tv
    # where v:=U^Tp (the corresponding N(0,I) variable since U=chol(S)^{-T})
    def kinetic_energy(self, state, precond_state):
        p_flat = state.p_flat
        U = precond_state.inv_var_triu_factor
        v_flat = U.T @ p_flat if jnp.ndim(U) == 2 else U * p_flat
        return 0.5*jnp.dot(v_flat, v_flat)
    
    # sample p ~ N(0,S), where S is approx the posterior covariance
    # equivalent to v~N(0,I) and p = Lv, with LL^T = S.
    def refresh_aux_vars(self, state, precond_state):
        rng_key, v_key = random.split(state.rng_key)
        v_flat = random.normal(v_key, jnp.shape(state.p_flat))
        L = precond_state.var_chol_tril
        p_flat = L @ v_flat if jnp.ndim(L) == 2 else L * v_flat
        return state._replace(p_flat = p_flat, rng_key = rng_key)
    
    # pCN as rotation
    # note: for implementation purposes, we assume `step_size` to be abs value 
    # of the angle, while the sign is stored in `base_step_size` for convenience
    def involution_main(self, step_size, state, precond_state):
        x_flat, unravel_fn = flatten_util.ravel_pytree(state.x)
        p_flat = state.p_flat
        theta = step_size * jnp.sign(state.base_step_size)
        sin_theta, cos_theta = jnp.sin(theta), jnp.cos(theta)
        x_flat_new =  cos_theta*x_flat + sin_theta*p_flat
        p_flat_new = -sin_theta*x_flat + cos_theta*p_flat
        return state._replace(x = unravel_fn(x_flat_new), p_flat = p_flat_new)
    
    # to achieve the sign flip in theta, we flip base_step_size
    # it is not kosher but gets the job done since the result of this function
    # is never used beyond its log joint value
    def involution_aux(self, step_size, state, precond_state):
        return state._replace(base_step_size = -state.base_step_size)
    
    # custom formula for turning an exponent into an angle in [0,pi/2], such that
    #   1) exponent = 0 => step_size = abs_base_theta
    #   2) exponent -> -inf => step_size -> 0
    #   3) exponent ->  inf => step_size -> pi/2
    def step_size(self, base_step_size, exponent):
        abs_base_theta = jnp.abs(base_step_size)
        return jnp.arctan(jnp.tan(abs_base_theta) * (2.0 ** exponent))

