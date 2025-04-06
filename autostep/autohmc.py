import jax
from jax import flatten_util
from jax import lax
from jax import numpy as jnp
from jax import random

from autostep import autostep
from autostep import preconditioning
from autostep import selectors
from autostep import statistics
from autostep.utils import std_normal_potential, apply_precond

class AutoHMC(autostep.AutoStep):

    def __init__(
        self,
        model=None,
        potential_fn=None,
        tempered_potential = None,
        n_leapfrog_steps = 1,
        init_base_step_size = 1.0,
        selector = selectors.SymmetricSelector(),
        preconditioner = preconditioning.MixDiagonalPreconditioner(),
        init_inv_temp = None
    ):
        self._model = model
        self._potential_fn = potential_fn
        self.tempered_potential = tempered_potential
        self._postprocess_fn = None
        self.n_leapfrog_steps = n_leapfrog_steps
        self.init_base_step_size = init_base_step_size
        self.selector = selector
        self.preconditioner = preconditioner
        self.init_inv_temp = (
            None if init_inv_temp is None else jnp.array(init_inv_temp)
        )
    
    def init_extras(self, initial_state):
        self.integrator = gen_integrator(self.tempered_potential, initial_state)
        return initial_state
    
    def update_log_joint(self, state):
        x, p_flat, *_ = state
        new_temp_pot = self.tempered_potential(x, state.inv_temp)
        new_log_joint = -new_temp_pot - std_normal_potential(p_flat)
        new_stats = statistics.increase_n_pot_evals_by_one(state.stats)
        return state._replace(log_joint = new_log_joint, stats = new_stats)
    
    def refresh_aux_vars(self, state):
        rng_key, v_key = random.split(state.rng_key)
        p_flat = random.normal(v_key, jnp.shape(state.p_flat))
        return state._replace(p_flat = p_flat, rng_key = rng_key)
    
    def involution_main(self, step_size, state, precond_state):
        return self.integrator(
            step_size, state, precond_state, self.n_leapfrog_steps
        )
    
    def involution_aux(self, step_size, state, precond_state):
        return state._replace(p_flat = -state.p_flat)


##############################################################################
# leapfrog (velocity Verlet) integrator
# The update is
# 	p* = p  - (eps/2)grad(U)(x)
# 	x' = x  + eps M^{-1}p*
# 	p' = p* - (eps/2)grad(U)(x')
# Let L lower triangular such that
#   M = LL^T
# If y ~ N(0,I), then
#   p = Ly ~ N(0,M)
# Working in this space requires
#   - Getting L=chol(M) and M^{-1}= [L^{-1}]^T [L^{-1}]
#   - At each step we need 3 matrix ops:
#       1) Sampling p = Ly
#       2) kinetic energy eval: [L^{-1}p]^T [L^{-1}p]
#       3) For each leapfrog, a single matop (M^{-1}p)
# The alternative is to work with y:=L^{-1}p, which gives the updates 
# 	y* = y  - (eps/2)L^{-1}grad(U)(x)
# 	x' = x  + eps (L^T)^{-1}y
# 	y' = y* - (eps/2)L^{-1}grad(U)(x')
# This means we need 
#   - To get L^{-1} and (L^{-1})^T
#   - No matrix ops outside leapfrog
#   - ...but 3 O(d^2) ops for every LF step!
# So this is too expensive.
#
# Note: when M = S^{-1}, with S=Cov(x). Clearly, M^{-1}=S. But also,
#   S = LL^T <=> M = S^{-1} = (L^T)^{-1}L^{-1} = UU^T
# where U := (L^T)^{-1}. This solve is O(d^2) because L^T is upper triangular,
# which also means that U is also upper triangular. Then we have
#   p = Uy ~ N(0,M)
# Hence, we need S (no-op as this is directly estimated) and U.
##############################################################################

def velocity_step(p_flat, step_size, precond_state, grad_flat):
    return p_flat - step_size * apply_precond(precond_state, grad_flat)

def position_step(x_flat, step_size, precond_state, p_flat):
    return x_flat + step_size * apply_precond(precond_state, p_flat)

# helper function to define the leapfrog integrator
def gen_integrator(tempered_potential, initial_state):
    unravel_fn = flatten_util.ravel_pytree(initial_state.x)[1]

    def grad_flat_x_flat(x_flat, inv_temp):
        """
        Flattened gradient of the potential evaluated at a flattened state.
        """
        return flatten_util.ravel_pytree(
            # note: by default, `grad` takes gradient w.r.t. the first arg only
            jax.grad(tempered_potential)(unravel_fn(x_flat), inv_temp)
        )[0]

    # full position and velocity updates
    def integrator_scan_body_fn(carry, _):
        x_flat, p_flat, step_size, precond_state, inv_temp = carry
        x_flat    = position_step(x_flat, step_size, precond_state, p_flat)
        grad_flat = grad_flat_x_flat(x_flat, inv_temp)
        p_flat    = velocity_step(p_flat, step_size, precond_state, grad_flat)
        return (x_flat, p_flat, step_size, precond_state, inv_temp), None
    
    # leapfrog integrator using Neal (2011, Fig. 2) trick to use only 
    # (n_steps+1) grad evals
    # IMPORTANT: contrary to HMC convention, `precond_state` is on the scale of
    # Sigma^{1/2}, where Sigma=Cov(x_flat)
    def integrator(step_size, state, precond_state, n_steps):
        # jax.debug.print("start: step_size={s}, precond_state={d}", ordered=True, s=step_size, d=precond_state)

        # first velocity half-step
        x, p_flat, *_, inv_temp = state
        x_flat    = flatten_util.ravel_pytree(x)[0]
        grad_flat = grad_flat_x_flat(x_flat, inv_temp)
        # jax.debug.print("pre 1st momentum half-step: x={x}, x_flat={xf}, grad={g}, p_flat={v}", ordered=True, x=x, xf=x_flat, g=grad_flat, v=p_flat)
        p_flat    = velocity_step(p_flat, step_size/2, precond_state, grad_flat)
        # jax.debug.print("post: p_flat={v}", ordered=True, v=p_flat)

        # loop full position and velocity leapfrog steps
        # note: slight modification from Neal's to avoid the "if" inside the loop
        # In particular, MALA (n_steps=1) doesn't enter the loop
        x_flat, p_flat, *_ = lax.scan(
            integrator_scan_body_fn,
            (x_flat, p_flat, step_size, precond_state, inv_temp),
            None,
            n_steps-1
        )[0]
        # jax.debug.print("post loop: x_flat={xf}, p_flat={v}", ordered=True, xf=x_flat, v=p_flat)

        # final full position step plus half velocity step
        x_flat    = position_step(x_flat, step_size, precond_state, p_flat)
        grad_flat = grad_flat_x_flat(x_flat, inv_temp)
        p_flat    = velocity_step(p_flat, step_size/2, precond_state, grad_flat)
        
        # unravel, update state, and return it
        x_new = unravel_fn(x_flat)
        # jax.debug.print("final: x_new={x}, x_flat={xf}, grad={g}, p_flat={v}", ordered=True, x=x_new, xf=x_flat, g=grad_flat, v=p_flat)

        return state._replace(x = x_new, p_flat = p_flat)
    
    return integrator


## autoMALA alias
def AutoMALA(*args, **kwargs):
    return AutoHMC(n_leapfrog_steps=1, *args, **kwargs)
