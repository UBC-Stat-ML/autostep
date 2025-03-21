from collections import namedtuple
import jax
from jax import numpy as jnp
from autostep.preconditioning import is_dense

AutoStepAdaptStats = namedtuple(
    "AutoStepAdaptStats",
    [
        "sample_idx",
        "mean_step_size",
        "mean_acc_prob",
        "means_flat",
        "vars_flat"
    ],
    defaults=(0, 0.0, 0.0, None, None)
)
"""
A :func:`~collections.namedtuple` used for round-based adaptation. Its contents
are cleared at the end of each round. Its fields are:

 - **sample_idx** - current sample in the current round.
 - **mean_step_size** - online mean step size for the round.
 - **mean_acc_prob** - online mean acceptance probability for the round.
 - **means_flat** - online mean of the flattened sample field.
 - **vars_flat** - online variance estimate of the flattened sample field.
"""

def make_adapt_stats_recorder(
        sample_field_flat_shape, 
        vars_flat_shape = None,
        preconditioner = None
    ):
    if vars_flat_shape is None:
        vars_flat_shape = (
            2*sample_field_flat_shape
            if is_dense(preconditioner) 
            else sample_field_flat_shape
        )
    return AutoStepAdaptStats()._replace(
        means_flat = jnp.zeros(sample_field_flat_shape),
        vars_flat = jnp.zeros(vars_flat_shape)
    )

def empty_adapt_stats_recorder(adapt_stats):
    return make_adapt_stats_recorder(
        jnp.shape(adapt_stats.means_flat),
        vars_flat_shape=jnp.shape(adapt_stats.vars_flat)
    )

AutoStepStats = namedtuple(
    "AutoStepStats",
    [
        "n_pot_evals",
        "n_samples",
        "adapt_stats"
    ],
    defaults=(0, 0, AutoStepAdaptStats())
)
"""
A :func:`~collections.namedtuple` consisting of the following fields:

 - **n_pot_evals** - total number of potential evaluations (including warmup).
 - **n_samples** - total number of calls to ``sample`` so far. At the end of
   a run, this should be equal to `num_warmup+num_samples`.
 - **adapt_stats** - an ``AutoStepAdaptStats`` namedtuple which contains adaptation
   information pertaining to the current round.
"""

def make_stats_recorder(sample_field_flat_shape, preconditioner):
    return AutoStepStats()._replace(
        adapt_stats = make_adapt_stats_recorder(
            sample_field_flat_shape, preconditioner=preconditioner
        )
    )

def increase_n_pot_evals_by_one(stats):
    return stats._replace(n_pot_evals = stats.n_pot_evals + 1)

# update sample statistics
# use the "tracking-error" formula for updating means, based on the identity
#     m_n+1 = (n*m_n + x_{n+1}) / (n+1) = ((n+1)*m_n + (x_{n+1}-m_n)) / (n+1) 
#           = m_n + (x_{n+1}-m_n)/(n+1)
# use Welford's_online_algorithm to update variances
#     v_n+1 = [n*v_n + (x_n+1 - m_n+1)(x_n+1 - m_n)]/(n+1)
@jax.jit
def record_post_sample_stats(stats, avg_fwd_bwd_step_size, acc_prob, x_flat):
    # update whole-run statistics
    n_pot_evals, n_samples, adapt_stats = stats
    n_samples = n_samples + 1

    # update round statistics
    sample_idx, mean_step_size, mean_acc_prob, means_flat, vars_flat = adapt_stats
    sample_idx = sample_idx + 1
    new_mean_step_size = mean_step_size + (avg_fwd_bwd_step_size-mean_step_size)/sample_idx
    new_mean_acc_prob = mean_acc_prob + (acc_prob-mean_acc_prob)/sample_idx
    new_means_flat = means_flat + (x_flat - means_flat)/sample_idx
    dvars = delta_vars(vars_flat, x_flat - means_flat, x_flat - new_means_flat)
    new_vars_flat = ((sample_idx-1)*vars_flat + dvars) / sample_idx
    return AutoStepStats(
        n_pot_evals, 
        n_samples, 
        AutoStepAdaptStats(
            sample_idx, 
            new_mean_step_size, 
            new_mean_acc_prob, 
            new_means_flat,
            new_vars_flat
        )
    )

def delta_vars(vars_flat, dx1, dx2):
    if len(jnp.shape(vars_flat))==1:
        return dx1 * dx2
    else:
        return jnp.outer(dx1, dx2)

