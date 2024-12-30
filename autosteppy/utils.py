import jax
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

def reset_exponent(state):
    return state._replace(exponent = 0)
    
def step_size(base_step_size, exponent):
    return base_step_size * (2. ** exponent)

def gen_shrink_step_size_cond_fun(selector):
    def shrink_step_size_cond_fun(args):
        state, selector_params, init_log_joint, _ = args
        log_diff = mod_step_size_cond_common(state, selector_params, init_log_joint)
        return selector.should_shrink(selector_params, log_diff)
    return shrink_step_size_cond_fun

def gen_grow_step_size_cond_fun(selector):
    def grow_step_size_cond_fun(args):
        state, selector_params, init_log_joint, _ = args
        log_diff = mod_step_size_cond_common(state, selector_params, init_log_joint)
        return selector.should_shrink(selector_params, log_diff)
    return grow_step_size_cond_fun

def mod_step_size_cond_common(state, selector_params, init_log_joint):
    log_diff = state.log_joint - init_log_joint
    jax.debug.print(f"Log-diff: {log_diff}\tBounds: {selector_params}")
    return log_diff

def gen_mod_step_size_body_fun(stepper, direction):
    def mod_step_size_body_fun(args):
        state, *extra, base_step_size = args
        state = state._replace(exponent = state.exponent + direction)
        eps = step_size(base_step_size, state.exponent)
        jax.debug.print(f"Direction: {direction}\tStep size: {eps}\tn_pot_evals: {state.stats.n_pot_evals}")
        new_state = stepper.update_log_joint(stepper.involution_main(eps, state))
        return (new_state, *extra, base_step_size,)
    return mod_step_size_body_fun
