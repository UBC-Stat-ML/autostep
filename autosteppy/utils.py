from jax import lax
from jax.experimental import checkify
from numpyro import infer

def proto_checkified_is_finite(x):
  checkify.check(lax.is_finite(x), f"Found non-finite value x = {x}")
  return

checkified_is_finite = checkify.checkify(proto_checkified_is_finite)


def std_normal_potential(v):
    return lax.dot(v,v)

# Taken without much changes from
# https://github.com/pyro-ppl/numpyro/blob/master/numpyro/infer/barker.py
def init_state_and_model(model, rng_key, model_args, model_kwargs, init_params):
    if model is not None:
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
