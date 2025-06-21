from functools import partial

import jax
from jax import lax

from numpyro import infer

import optax

from autostep import tempering

# initialize state, potential fun, and postprocess fun for a model
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

# find a better starting point using optimization
def optimize_init_params(logprior_and_loglik, init_params, inv_temp, n_iter):
    print(f'Running {n_iter} ADAM iterations to improve initial state.')

    # define target function to minimize
    target_fun = partial(
        tempering.tempered_potential,
        logprior_and_loglik,
        inv_temp=inv_temp
    )

    # use ADAM to search for better initial point
    solver = optax.adam(learning_rate=0.003)
    def scan_fn(carry, _):
        params, opt_state = carry
        grad = jax.grad(target_fun)(params)
        updates, opt_state = solver.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), None
    print(f'Initial tempered potential: {target_fun(init_params):.1e}')
    opt_params, opt_state = lax.scan(
        scan_fn, 
        (init_params, solver.init(init_params)),
        length = n_iter
    )[0]
    print(f'Final tempered potential: {target_fun(opt_params):.1e}')
    
    return opt_params