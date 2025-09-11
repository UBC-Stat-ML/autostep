from functools import partial
from copy import deepcopy

from numpyro import infer

from autostep import tempering, optimization

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
# optimization-based initialization utils
###############################################################################

# standalone (i.e., no kernel required) MAP optimization
def MAP(
        model,
        rng_key,
        model_args,
        model_kwargs,
        options = {'strategy': "L-BFGS", 'params': {'n_iter': 64}}
    ):
    init_params, potential_fn, *_ = init_model(
        model, rng_key, model_args, model_kwargs
    )
    return optimization.optimize_fun(
        potential_fn, 
        init_params, 
        options,
        verbose = True
    )

# called within kernel initialization
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

    # TODO: maybe use the final optimizer state to tune samplers (e.g. preconditioner)
    return optimization.optimize_fun(
        target_fun, 
        init_params, 
        deepcopy(initialization_settings) # safer since we `pop` stuff from it
    )[0]
