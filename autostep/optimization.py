import jax
from jax import lax

import optax
import tqdm

# simple scan-based optimization loop
def optimize_fun(
        target_fun, 
        init_params, 
        settings,
        verbose = True
    ):

    # select solver
    solver = settings['strategy']
    solver_params = settings['params']
    n_iter = solver_params.pop('n_iter')
    if settings['strategy'] == "L-BFGS":
        solver, step_fn = make_lbfgs_solver(target_fun, solver_params, verbose)
    else:
        raise ValueError(
            f"Unknown strategy '{settings['strategy']}'"
        )
    
    # optimization loop
    value = target_fun(init_params)
    verbose and print(f'Initial energy: {value:.1e}')
    params, opt_state = init_params, solver.init(init_params)
    with tqdm.trange(n_iter, disable=(not verbose)) as t:
        for _ in t:
            params, opt_state, value = step_fn(params, opt_state)
            diag_str = "fn={:.1e}".format(value)
            t.set_postfix_str(diag_str, refresh=False)

    if verbose:
        print(f'Final energy: {target_fun(params):.1e}')
    return params, opt_state

# L-BFGS
def make_lbfgs_solver(target_fun, solver_params, verbose):
    if verbose:
        print(f'L-BFGS optimization loop.')

    # can reuse stuff because target is deterministic
    # see https://optax.readthedocs.io/en/stable/api/optimizers.html#lbfgs
    cached_value_and_grad = optax.value_and_grad_from_state(target_fun)

    # build solver and loop function
    solver = optax.lbfgs(**solver_params)
    def step_fn(params, opt_state):
        value, grad = cached_value_and_grad(params, state=opt_state)
        updates, opt_state = solver.update(
            grad, opt_state, params, value=value, grad=grad, value_fn=target_fun
        )
        params = optax.apply_updates(params, updates)
        return params, opt_state, value
    
    return solver, jax.jit(step_fn)

