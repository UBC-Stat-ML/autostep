from functools import partial

import jax
from jax import lax

from numpyro import infer

import optax

from autostep import tempering

# initialize a NumPyro model
# get starting point, potential fun, and postprocess fun
# Taken without much changes from
# https://github.com/pyro-ppl/numpyro/blob/master/numpyro/infer/barker.py
def init_model(model, rng_key, model_args, model_kwargs):
    (
        params_info,
        potential_fn_gen,
        postprocess_fn,
        model_trace,
    ) = infer.util.initialize_model(
        rng_key,
        model,
        dynamic_args=True,
        model_args=model_args,
        model_kwargs=model_kwargs,
    )
    init_params = params_info[0]
    model_kwargs = {} if model_kwargs is None else model_kwargs
    potential_fn = potential_fn_gen(*model_args, **model_kwargs)
    return init_params, potential_fn, postprocess_fn

###############################################################################
# find a better starting point using optimization
###############################################################################

def optimize_init_params(
        logprior_and_loglik, 
        init_params, 
        inv_temp, 
        initialization_settings
    ):
    # define tempered posterior as target function
    target_fun = partial(
        tempering.tempered_potential,
        logprior_and_loglik,
        inv_temp=inv_temp
    )

    # select solver
    if initialization_settings['strategy'] == "L-BFGS":
        solver, scan_fn = make_lbfgs_solver(target_fun)
    elif initialization_settings['strategy'] == "ADAM":
        solver, scan_fn = make_adam_solver(target_fun)
    else:
        raise ValueError(
            f"Unknown strategy '{initialization_settings['strategy']}'"
        )

    # can reuse stuff because target is deterministic
    # see https://optax.readthedocs.io/en/stable/api/optimizers.html#lbfgs
    cached_value_and_grad = optax.value_and_grad_from_state(target_fun)

    # use L-BFGS to search for better initial point
    solver = optax.lbfgs()
    def scan_fn(carry, _):
        params, opt_state = carry
        value, grad = cached_value_and_grad(params, state=opt_state)
        updates, opt_state = solver.update(
            grad, opt_state, params, value=value, grad=grad, value_fn=target_fun
        )
        params = optax.apply_updates(params, updates)
        return (params, opt_state), None
    print(f'Initial energy: {target_fun(init_params):.1e}')
    opt_params, opt_state = lax.scan(
        scan_fn, 
        (init_params, solver.init(init_params)),
        length = initialization_settings['params']['n_iter']
    )[0]
    print(f'Final energy: {target_fun(opt_params):.1e}')
    
    return opt_params

# L-BFGS
def make_lbfgs_solver(target_fun):
    print(f'Using L-BFGS to improve initial state.')

    # can reuse stuff because target is deterministic
    # see https://optax.readthedocs.io/en/stable/api/optimizers.html#lbfgs
    cached_value_and_grad = optax.value_and_grad_from_state(target_fun)

    # build solver and loop function
    solver = optax.lbfgs()
    def scan_fn(carry, _):
        params, opt_state = carry
        value, grad = cached_value_and_grad(params, state=opt_state)
        updates, opt_state = solver.update(
            grad, opt_state, params, value=value, grad=grad, value_fn=target_fun
        )
        params = optax.apply_updates(params, updates)
        return (params, opt_state), None
    
    return solver, scan_fn

# ADAM
def make_adam_solver(target_fun):
    print(f'Using ADAM to improve initial state.')
    solver = optax.adam(learning_rate=0.003) # LR is from the example in docs
    def scan_fn(carry, _):
        params, opt_state = carry
        grad = jax.grad(target_fun)(params)
        updates, opt_state = solver.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), None
    
    return solver, scan_fn
