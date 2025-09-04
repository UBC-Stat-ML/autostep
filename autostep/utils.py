import jax
from jax import lax
from jax import numpy as jnp
from jax.experimental import checkify

###############################################################################
# basic utilities
###############################################################################

@checkify.checkify
def checkified_is_finite(x):
  checkify.check(lax.is_finite(x), f"Found non-finite value x = {x}")
  return True

@checkify.checkify
def checkified_is_zero(x):
  checkify.check(x==0, f"Expected zero but x = {x}")
  return True

def ceil_log2(x):
    """
    Ceiling of log2(x). Guaranteed to be an integer.
    """
    n_bits = jax.lax.clz(jnp.zeros_like(x))
    return n_bits - jax.lax.clz(x) - (jax.lax.population_count(x)==1)

def numerically_safe_diff(x0, x1):
    """
    Return `x1-x0` if x1 is not the next float after x0, and 0 otherwise.
    """
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

DEBUG_ALTER_STEP_SIZE = None # anything other than None will print during step size loop

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
            precond_state
        ) = args

        # `numerically_safe_diff` is used to avoid corner cases where the step
        # size is 0 already but, because of extreme nonlinearities, the 
        # potential at this fictituous "next" point gives a log_joint that is
        # exactly equal to the next float of `init_log_joint`
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
            precond_state
        ) = args
        exponent = exponent + direction
        eps = kernel.step_size(state.base_step_size, exponent)
        next_state = kernel.update_log_joint(
            kernel.involution_main(eps, state, precond_state),
            precond_state
        )
        next_log_joint = next_state.log_joint
        state = copy_state_extras(next_state, state)

        # maybe print debug info
        if DEBUG_ALTER_STEP_SIZE is not None:
            jax.debug.print(
                "dir: {d: d}: base: {bs:.8f} + exp: {e: d} = eps: {s:.8f} | (L0, L1, DL, NDL): ({l0: .2f},{l1: .2f},{dl: .2f},{ndl: .2f}) | bounds: ({a:.3f},{b:.3f})", 
                ordered=True,
                d=direction, 
                bs=state.base_step_size,
                e=exponent,
                s=eps,
                l0=init_log_joint,
                l1=next_log_joint,
                dl=next_log_joint-init_log_joint,
                ndl=numerically_safe_diff(init_log_joint,next_log_joint),
                a=selector_params[0],
                b=selector_params[1]
            )
            # jax.debug.print(
            #     "{v} | {c} | {i}",
            #     v=precond_state.var,
            #     c=precond_state.var_tril_factor,
            #     i=precond_state.inv_var_triu_factor
            # )

        return (
            state, 
            exponent, 
            next_log_joint, 
            init_log_joint,
            selector_params, 
            precond_state
        )

    return alter_step_size_body_fun
