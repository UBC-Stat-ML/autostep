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
        x, v_flat, *_ = state
        new_temp_pot = self.tempered_potential(x, state.inv_temp)
        new_log_joint = -new_temp_pot - std_normal_potential(v_flat)
        new_stats = statistics.increase_n_pot_evals_by_one(state.stats)
        return state._replace(log_joint = new_log_joint, stats = new_stats)
    
    def refresh_aux_vars(self, state):
        rng_key, v_key = random.split(state.rng_key)
        v_flat = random.normal(v_key, jnp.shape(state.v_flat))
        return state._replace(v_flat = v_flat, rng_key = rng_key)
    
    def involution_main(self, step_size, state, precond_array):
        return self.integrator(
            step_size, state, precond_array, self.n_leapfrog_steps
        )
    
    def involution_aux(self, step_size, state, precond_array):
        return state._replace(v_flat = -state.v_flat)


#######################################
# leapfrog (velocity Verlet) integrator
#######################################

def velocity_step(v_flat, step_size, precond_array, grad_flat):
    return v_flat - step_size * apply_precond(precond_array, grad_flat)

def position_step(x_flat, step_size, precond_array, v_flat):
    return x_flat + step_size * apply_precond(precond_array, v_flat)

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
        x_flat, v_flat, step_size, precond_array, inv_temp = carry
        x_flat    = position_step(x_flat, step_size, precond_array, v_flat)
        grad_flat = grad_flat_x_flat(x_flat, inv_temp)
        v_flat    = velocity_step(v_flat, step_size, precond_array, grad_flat)
        return (x_flat, v_flat, step_size, precond_array, inv_temp), None
    
    # leapfrog integrator using Neal (2011, Fig. 2) trick to use only 
    # (n_steps+1) grad evals
    # IMPORTANT: `precond_array` is on the scale of Sigma^{1/2}, where Sigma=Cov(x)
    def integrator(step_size, state, precond_array, n_steps):
        # jax.debug.print("start: step_size={s}, precond_array={d}", ordered=True, s=step_size, d=precond_array)

        # first velocity half-step
        x, v_flat, *_, inv_temp = state
        x_flat    = flatten_util.ravel_pytree(x)[0]
        grad_flat = grad_flat_x_flat(x_flat, inv_temp)
        # jax.debug.print("pre 1st momentum half-step: x={x}, x_flat={xf}, grad={g}, v_flat={v}", ordered=True, x=x, xf=x_flat, g=grad_flat, v=v_flat)
        v_flat    = velocity_step(v_flat, step_size/2, precond_array, grad_flat)
        # jax.debug.print("post: v_flat={v}", ordered=True, v=v_flat)

        # loop full position and velocity leapfrog steps
        # note: slight modification from Neal's to avoid the "if" inside the loop
        # In particular, MALA (n_steps=1) doesn't enter the loop
        x_flat, v_flat, *_ = lax.scan(
            integrator_scan_body_fn,
            (x_flat, v_flat, step_size, precond_array, inv_temp),
            None,
            n_steps-1
        )[0]
        # jax.debug.print("post loop: x_flat={xf}, v_flat={v}", ordered=True, xf=x_flat, v=v_flat)

        # final full position step plus half velocity step
        x_flat    = position_step(x_flat, step_size, precond_array, v_flat)
        grad_flat = grad_flat_x_flat(x_flat, inv_temp)
        v_flat    = velocity_step(v_flat, step_size/2, precond_array, grad_flat)
        
        # unravel, update state, and return it
        x_new = unravel_fn(x_flat)
        # jax.debug.print("final: x_new={x}, x_flat={xf}, grad={g}, v_flat={v}", ordered=True, x=x_new, xf=x_flat, g=grad_flat, v=v_flat)

        return state._replace(x = x_new, v_flat = v_flat)
    
    return integrator


## autoMALA alias
def AutoMALA(*args, **kwargs):
    return AutoHMC(n_leapfrog_steps=1, *args, **kwargs)
