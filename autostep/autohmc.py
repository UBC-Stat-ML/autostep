import jax
from jax import flatten_util
from jax import lax
from jax import numpy as jnp
from jax import random

from autostep import autostep
from autostep import preconditioning
from autostep import selectors
from autostep import statistics
from autostep import utils

class AutoHMC(autostep.AutoStep):

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
    
    def init_extras(self, initial_state):
        self.integrator = gen_integrator(self._potential_fn, initial_state)
    
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


def gen_integrator(potential_fn, initial_state):
    unravel_fn = flatten_util.ravel_pytree(initial_state.x)[1]

    def grad_flat_x_flat(x_flat):
        return flatten_util.ravel_pytree(
            jax.grad(potential_fn)(unravel_fn(x_flat))
        )[0]

    # full position and velocity updates
    def integrator_loop_body(args):
        i, x_flat, v_flat, step_size, diag_precond = args
        x_flat = x_flat + step_size * v_flat
        grad_flat = grad_flat_x_flat(x_flat)
        v_flat = v_flat - step_size * (diag_precond*grad_flat)
        return (x_flat, v_flat, step_size, diag_precond)
    
    # leapfrog integrator using Neal (2011, Fig. 2) trick to use only (n_steps+1) grad evals
    def integrator(step_size, state, diag_precond, n_steps):

        # first velocity half-step
        x, v_flat, *_ = state
        x_flat = flatten_util.ravel_pytree(x)[0]
        grad_flat = grad_flat_x_flat(x_flat)
        v_flat = v_flat - (step_size/2) * (diag_precond*grad_flat)

        # loop full position and velocity leapfrog steps
        # note: slight modification from Neal's to avoid the "if" inside the loop
        # In particular, MALA (n_steps=1) doesn't enter the loop
        x_flat, v_flat, *_ = lax.fori_loop(
            0,
            n_steps-1, 
            integrator_loop_body,
            (x_flat, v_flat, step_size, diag_precond)
        )

        # final full position step plus half velocity step
        x_flat = x_flat + step_size * v_flat
        grad_flat = grad_flat_x_flat(x_flat)
        v_flat = v_flat - (step_size/2) * (diag_precond*grad_flat)
        
        # unravel, update state, and return it
        x_new = unravel_fn(x_flat)
        return state._replace(x = x_new, v_flat = v_flat)
    
    return integrator
