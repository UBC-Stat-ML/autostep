from copy import deepcopy

import math

import numpy as np

import jax
from jax import numpy as jnp

import optax
import tqdm

from automcmc import utils

# L-BFGS
def make_lbfgs_solver(target_fun, solver_params, verbose):
    if verbose:
        print(f'L-BFGS optimization loop.')

    # can reuse stuff because target is deterministic
    # see https://optax.readthedocs.io/en/stable/api/optimizers.html#lbfgs
    cached_value_and_grad = optax.value_and_grad_from_state(target_fun)

    # build solver and loop function
    solver = optax.lbfgs(**solver_params)
    @jax.jit
    def step_fn(params, opt_state):
        value, grad = cached_value_and_grad(params, state=opt_state)
        updates, opt_state = solver.update(
            grad, opt_state, params, value=value, grad=grad, value_fn=target_fun
        )
        params = optax.apply_updates(params, updates)
        grad_norm = utils.pytree_norm(grad, ord=jnp.inf) # sup norm
        return params, opt_state, value, grad_norm
    
    return solver, step_fn

def optimization_loop(
        target_fun,
        step_fn,
        params,
        opt_state,
        n_iter, 
        tol,
        max_consecutive, 
        verbose
    ):
    value = old_value = target_fun(params)
    verbose and print(f'Initial energy: {value:.1e}')
    old_params = params
    grad_norm = value_abs_diff = params_diff_norm = jnp.full_like(value, 10*tol)
    n_consecutive = np.zeros((3,), np.int32) # one counter for each termination criterion
    n = 0
    with tqdm.tqdm(total=n_iter, disable=(not verbose)) as t:
        while (n < n_iter and np.all(n_consecutive < max_consecutive)):
            # take one optim step
            params, opt_state, value, grad_norm = step_fn(params, opt_state)

            # update termination indicators
            value_abs_diff = jnp.abs(value-old_value)
            old_value = value
            params_diff_norm = utils.pytree_norm(
                jax.tree.map(lambda a,b: a-b, params, old_params),
                ord=jnp.inf # sup norm
            )
            old_params = params

            # update termination counters
            for (i,eps) in enumerate((grad_norm,value_abs_diff,params_diff_norm)):
                n_consecutive[i] = n_consecutive[i]+1 if eps < tol else 0

            # update progress bar and iteration counter
            diag_str = "f={:.1e}, Δf={:.0e}, |g|={:.0e}, |Δx|={:.0e}" \
                .format(value, value_abs_diff, grad_norm, params_diff_norm)
            t.set_postfix_str(diag_str, refresh=False) # will refresh with `update`
            t.update()
            n += 1
    return params, opt_state, value

# optimization loop
def optimize_fun(
        target_fun, 
        init_params, 
        settings,
        verbose = True,
        tol = None,
    ):
    settings = deepcopy(settings) # safer since we `pop` stuff from it
    
    if tol is None:
        # default to sqrt of machine tol of the float type used in the first leaf
        tol = 10*jnp.finfo(jax.tree.leaves(init_params)[0].dtype).eps

    # select solver
    solver_params = settings['solver_params']
    if settings['strategy'] == "L-BFGS":
        solver, step_fn = make_lbfgs_solver(target_fun, solver_params, verbose)
    else:
        raise ValueError(
            f"Unknown strategy '{settings['strategy']}'"
        )
    
    opt_state = solver.init(init_params)
    params, opt_state, final_value = optimization_loop(
        target_fun,
        step_fn,
        params,
        opt_state,
        n_iter, 
        tol,
        max_consecutive, 
        verbose
    )

    if verbose:
        print(f'Final energy: {final_value:.1e}')
              
    return params, opt_state

