from collections import namedtuple

AutoStepStats = namedtuple(
    "AutoStepStats",
    [
        "n_pot_evals",
        "n_samples",
        "mean_step_size"
    ],
    defaults=(0, 0, 0.)
)

def record_post_sample_stats(stats: AutoStepStats, avg_fwd_bwd_step_size) -> AutoStepStats:
    n_pot_evals, n_samples, mean_step_size = stats
    new_mean_step_size = (n_samples*mean_step_size + avg_fwd_bwd_step_size) / (n_samples + 1)
    return AutoStepStats(n_pot_evals, n_samples + 1, new_mean_step_size)
