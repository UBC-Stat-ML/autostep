import jax
from jax import lax
from jax import numpy as jnp
from jax.experimental import checkify
from numpyro import infer

from autostep.preconditioning import is_dense

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

@jax.jit
def numerically_safe_diff(x0, x1):
    """Return `x1-x0` if x1 is not the next float after x0, and 0 otherwise."""
    return jax.lax.cond(
        jax.lax.nextafter(x0, x1) == x1,
        lambda t: jnp.zeros_like(t[0]),
        lambda t: t[1]-t[0],
        (x0,x1)
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

def init_sqrt_var(sample_field_flat_shape, preconditioner):
    if is_dense(preconditioner):
        return jnp.eye(*(2*sample_field_flat_shape))
    else: 
        return jnp.ones(sample_field_flat_shape)

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

def gen_alter_step_size_cond_fun(pred_fun, max_n_iter):
    def alter_step_size_cond_fun(args):
        (
            state, 
            exponent, 
            next_log_joint, 
            init_log_joint,
            selector_params, 
            precond_array
        ) = args
        log_diff = numerically_safe_diff(init_log_joint,next_log_joint)
        decision = jnp.logical_and(
            lax.abs(exponent) < max_n_iter,     # bail if max number of iterations reached
            pred_fun(selector_params, log_diff)
        )

        return decision
    return alter_step_size_cond_fun

def gen_alter_step_size_body_fun(kernel, direction):
    def alter_step_size_body_fun(args):
        (
            state, 
            exponent, 
            next_log_joint, 
            init_log_joint,
            selector_params, 
            precond_array
        ) = args
        exponent = exponent + direction
        eps = step_size(state.base_step_size, exponent)
        next_state = kernel.update_log_joint(
            kernel.involution_main(eps, state, precond_array)
        )
        next_log_joint = next_state.log_joint
        state = copy_state_extras(next_state, state)

        # debug
        jax.debug.print(
            "dir: {d}: + exp: {e} + eps: {s:.8f} + (L0, L1, DL, NDL): ({l0:.2f},{l1:.2f},{dl:.2f},{ndl:.2f}) + bounds: ({a},{b})", 
            ordered=True,
            d=direction, 
            e=exponent,
            s=eps,
            l0=init_log_joint,
            l1=next_log_joint,
            dl=next_log_joint-init_log_joint,
            ndl=numerically_safe_diff(init_log_joint,next_log_joint),
            a=selector_params[0],
            b=selector_params[1]
        )

        return (
            state, 
            exponent, 
            next_log_joint, 
            init_log_joint,
            selector_params, 
            precond_array
        )

    return alter_step_size_body_fun
