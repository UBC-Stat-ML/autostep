from collections import namedtuple
from jax import numpy as jnp

AutoStepStats = namedtuple(
    "AutoStepStats",
    [
        "n_pot_evals",
        "n_samples",
        "mean_step_size",
        "mean_acc_prob",
        "means_flat",
        "vars_flat"
    ],
    defaults=(0, 0, 0.0, 0.0, None, None)
)

def make_recorder(sample_field_flat_shape):
    return AutoStepStats(
        0, 0, 0.0, 0.0,
        jnp.zeros(sample_field_flat_shape), jnp.zeros(sample_field_flat_shape)
    )

def increase_n_pot_evals_by_one(stats):
    return stats._replace(n_pot_evals = stats.n_pot_evals + 1)

# update sample statistics
# use the "tracking-error" formula for updating means, based on the identity
# m_n+1 = (n*m_n + x_{n+1}) / (n+1) = ((n+1)*m_n + (x_{n+1}-m_n)) / (n+1) 
# = m_n + (x_{n+1}-m_n)/(n+1)
# use Welford's_online_algorithm to update variances
# v_n+1 = [n*v_n + (x_n+1 - m_n+1)(x_n+1 - m_n)]/(n+1)
def record_post_sample_stats(stats, avg_fwd_bwd_step_size, acc_prob, x_flat):
    n_pot_evals, n_samples, mean_step_size, mean_acc_prob, means_flat, vars_flat = stats
    n_samples = n_samples + 1
    new_mean_step_size = mean_step_size + (avg_fwd_bwd_step_size-mean_step_size)/n_samples
    new_mean_acc_prob = mean_acc_prob + (acc_prob-mean_acc_prob)/n_samples
    new_means_flat = means_flat + (x_flat - means_flat)/n_samples
    new_vars_flat = ((n_samples-1)*vars_flat + (x_flat - means_flat) * (x_flat - new_means_flat)) / n_samples
    return AutoStepStats(
        n_pot_evals, n_samples, new_mean_step_size, new_mean_acc_prob,
        new_means_flat, new_vars_flat
    )
