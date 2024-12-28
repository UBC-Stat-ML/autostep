from collections import namedtuple

AutoStepStats = namedtuple(
    "AutoStepStats",
    [
        "n_pot_evals",
        "n_samples",
        "mean_step_size",
        "mean_acc_prob"
    ],
    defaults=(0, 0, 0., 0.)
)

def increase_n_pot_evals_by_one(stats: AutoStepStats) -> AutoStepStats:
    n_pot_evals, _ = stats
    return stats._replace(n_pot_evals=n_pot_evals+1)

# update sample statistics
# use the "tracking-error" formula for updating means, based on the identity
# m_n+1 = (n*m_n + x_{n+1}) / (n+1) = ((n+1)*m_n + (x_{n+1}-m_n)) / (n+1) 
# = m_n + (x_{n+1}-m_n)/(n+1)
def record_post_sample_stats(stats: AutoStepStats, avg_fwd_bwd_step_size, acc_prob) -> AutoStepStats:
    n_pot_evals, n_samples, mean_step_size, mean_acc_prob = stats
    n_samples = n_samples + 1
    new_mean_step_size = mean_step_size + (avg_fwd_bwd_step_size-mean_step_size)/n_samples
    new_mean_acc_prob = mean_acc_prob + (acc_prob-mean_acc_prob)/n_samples
    return AutoStepStats(n_pot_evals, n_samples, new_mean_step_size, new_mean_acc_prob)
