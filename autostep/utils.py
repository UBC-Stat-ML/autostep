import jax
from jax import lax
from jax import numpy as jnp
from jax.experimental import checkify
from numpyro import infer

###############################################################################
# basic utilities
###############################################################################

def proto_checkified_is_finite(x):
  checkify.check(lax.is_finite(x), f"Found non-finite value x = {x}")
  return

checkified_is_finite = checkify.checkify(proto_checkified_is_finite)

def proto_checkified_is_zero(x):
  checkify.check(x==0, f"Expected zero but x = {x}")
  return

checkified_is_zero = checkify.checkify(proto_checkified_is_zero)

def std_normal_potential(v):
    return (v*v).sum()/2

def ceil_log2(x):
    """
    Ceiling of log2(x). Guaranteed to be an integer.
    """
    n_bits = jax.lax.clz(jnp.zeros_like(x))
    return n_bits - jax.lax.clz(x) - (jax.lax.population_count(x)==1)

def apply_precond(precond_array, vec):
    return (
        precond_array * vec 
        if len(jnp.shape(precond_array)) == 1 
        else precond_array @ vec
    )

###############################################################################
# Rounds-based sampling arithmetic
#
# Running a rounds-based sampler for "n_r" rounds means we take a total of
#   n_samples = 2 + 4 + 8 + ... + 2^{n_r} = 2^(n_r+1) - 2 = [2^{n_r} - 2] + [2^{n_r}]
# samples. The decomposition in the RHS shows that this corresponds to
#   - A warmup phase of n_r-1 rounds, with a total of 2^{n_r} - 2 samples.
#     We call this quantity "n_warmup".
#   - A main sampling phase comprised of a final round with 2^{n_r} steps.
#     We call this quantity "n_keep".
# We use the name "n_samples" for the sum of "n_warmup" and "n_keep". Finally,
# we call "sample_idx" the current step within a round, which resets at the end
# of every round.
###############################################################################

def split_n_rounds(n_rounds):
    n_keep = 2 ** n_rounds
    return (n_keep-2, n_keep)

def current_round(n_samples):
    return ceil_log2(n_samples + 2) - 1

def n_warmup_to_adapt_rounds(n_warmup):
    return ceil_log2(n_warmup + 2) - 1

###############################################################################
# kernel initialization
###############################################################################

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
# functions used withing lax.cond to create the output state for `sample`
###############################################################################

def next_state_accepted(args):
    _, proposed_state, bwd_state, rng_key = args
    # keep everything from proposed_state except for stats (use bwd) and rng_key
    return proposed_state._replace(stats = bwd_state.stats, rng_key = rng_key)

def next_state_rejected(args):
    init_state, _, bwd_state, rng_key = args
    # keep everything from init_state except for stats (use bwd) and rng_key
    return init_state._replace(stats = bwd_state.stats, rng_key = rng_key)

###############################################################################
# functions that control the step-size growing and shrinking loops
###############################################################################

def step_size(base_step_size, exponent):
    return base_step_size * (2.0 ** exponent)

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
        state, exponent, _, *extra, precond_array = args
        exponent = exponent + direction
        eps = step_size(state.base_step_size, exponent)
        next_state = kernel.update_log_joint(
            kernel.involution_main(eps, state, precond_array)
        )
        next_log_joint = next_state.log_joint
        state = copy_state_extras(next_state, state)

        # jax.debug.print(
        #     "Direction: {d}, Step size: {eps}, n_pot_evals: {n}",
        #     ordered=True, d=direction, eps=eps, n=state.stats.n_pot_evals)

        return (state, exponent, next_log_joint, *extra, precond_array)

    return alter_step_size_body_fun
