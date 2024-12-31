from collections import namedtuple
from jax import numpy as jnp

AutoStepStats = namedtuple(
    "AutoStepStats",
    [
        "n_pot_evals",
        "n_samples",
        "mean_step_size",
        "mean_acc_prob"
    ],
    defaults=(jnp.int32(0), jnp.int32(0), jnp.float32(0.0), jnp.float32(0.0))
)

def increase_n_pot_evals_by_one(stats):
    return stats._replace(n_pot_evals = stats.n_pot_evals + 1)

# update sample statistics
# use the "tracking-error" formula for updating means, based on the identity
# m_n+1 = (n*m_n + x_{n+1}) / (n+1) = ((n+1)*m_n + (x_{n+1}-m_n)) / (n+1) 
# = m_n + (x_{n+1}-m_n)/(n+1)
def record_post_sample_stats(stats, avg_fwd_bwd_step_size, acc_prob):
    n_pot_evals, n_samples, mean_step_size, mean_acc_prob = stats
    n_samples = n_samples + 1
    new_mean_step_size = mean_step_size + (avg_fwd_bwd_step_size-mean_step_size)/n_samples
    new_mean_acc_prob = mean_acc_prob + (acc_prob-mean_acc_prob)/n_samples
    return AutoStepStats(n_pot_evals, n_samples, new_mean_step_size, new_mean_acc_prob)
