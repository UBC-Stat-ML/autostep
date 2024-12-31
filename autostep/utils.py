import jax
from jax import lax
from jax.experimental import checkify
from numpyro import infer

def proto_checkified_is_finite(x):
  checkify.check(lax.is_finite(x), f"Found non-finite value x = {x}")
  return

checkified_is_finite = checkify.checkify(proto_checkified_is_finite)

def proto_checkified_is_zero(x):
  checkify.check(x==0, f"Expected zero but x = {x}")
  return

checkified_is_zero = checkify.checkify(proto_checkified_is_zero)

def std_normal_potential(v):
    return lax.dot(v,v)

# Taken without much changes from
# https://github.com/pyro-ppl/numpyro/blob/master/numpyro/infer/barker.py
def init_state_and_model(model, rng_key, model_args, model_kwargs, init_params):
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


# functions used withing lax.cond to create the output state for `sample` 
def next_state_accepted(args):
    _, proposed_state, bwd_state, rng_key = args
    # keep everything from proposed_state except for stats (use bwd) and rng_key
    return proposed_state._replace(stats = bwd_state.stats, rng_key = rng_key)

def next_state_rejected(args):
    init_state, _, bwd_state, rng_key = args
    # keep everything from init_state except for stats (use bwd) and rng_key
    return init_state._replace(stats = bwd_state.stats, rng_key = rng_key)
    
def step_size(base_step_size, exponent):
    return base_step_size * (2. ** exponent)

def copy_state_extras(source, dest):
    return dest._replace(stats = source.stats, rng_key = source.rng_key)

def gen_alter_step_size_cond_fun(pred_fun):
    def alter_step_size_cond_fun(args):
        state, exponent, next_log_joint, init_log_joint, selector_params, *extra = args
        log_diff = next_log_joint - init_log_joint
        decision = pred_fun(selector_params, log_diff)

        # jax.debug.print("{f}? Log-diff: {l} + bounds: ({a},{b}) => Decision: {d}", 
        #         ordered=True, f=pred_fun.__name__, l=log_diff, a=selector_params[0], 
        #         b=selector_params[1], d=decision)

        return decision
    return alter_step_size_cond_fun

def gen_alter_step_size_body_fun(kernel, direction):
    def alter_step_size_body_fun(args):
        state, exponent, _, *extra, base_step_size = args
        exponent = exponent + direction
        eps = step_size(base_step_size, exponent)
        next_state = kernel.update_log_joint(kernel.involution_main(eps, state))
        next_log_joint = next_state.log_joint
        state = copy_state_extras(next_state, state)

        # jax.debug.print(
        #     "Direction: {d}, Step size: {eps}, n_pot_evals: {n}",
        #     ordered=True, d=direction, eps=eps, n=state.stats.n_pot_evals)

        return (state, exponent, next_log_joint, *extra, base_step_size,)

    return alter_step_size_body_fun
