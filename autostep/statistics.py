from collections import namedtuple
from jax import numpy as jnp

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
 - **rng_key** - random number generator seed used for generating proposals, etc.
"""

def make_adapt_stats_recorder(sample_field_flat_shape):
    return AutoStepAdaptStats()._replace(
        means_flat = jnp.zeros(sample_field_flat_shape),
        vars_flat = jnp.zeros(sample_field_flat_shape))

def empty_adapt_stats_recorder(adapt_stats):
    return make_adapt_stats_recorder(jnp.shape(adapt_stats.means_flat))

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
 - **n_samples** - total number of calls to `kernel.sample` (including warmup).
 - **adapt_stats** - An ``AutoStepAdaptStats`` namedtuple which contains adaptation
   information pertaining to the current round.
- **z** - Python collection representing values (unconstrained samples from
   the posterior) at latent sites.
 - **potential_energy** - Potential energy computed at the given value of ``z``.
 - **z_grad** - Gradient of potential energy w.r.t. latent sample sites.
 - **accept_prob** - Acceptance probability of the proposal. Note that ``z``
   does not correspond to the proposal if it is rejected.
 - **mean_accept_prob** - Mean acceptance probability until current iteration
   during warmup adaptation or sampling (for diagnostics).
 
   + **step_size** - Step size to be used by the integrator in the next iteration.
   + **inverse_mass_matrix** - The inverse mass matrix to be used for the next
     iteration.
   + **mass_matrix_sqrt** - The square root of mass matrix to be used for the next
     iteration. In case of dense mass, this is the Cholesky factorization of the
     mass matrix.

 - **rng_key** - random number generator seed used for generating proposals, etc.
"""

def make_stats_recorder(sample_field_flat_shape):
    return AutoStepStats()._replace(
        adapt_stats = make_adapt_stats_recorder(sample_field_flat_shape))

def increase_n_pot_evals_by_one(stats):
    return stats._replace(n_pot_evals = stats.n_pot_evals + 1)

# update sample statistics
# use the "tracking-error" formula for updating means, based on the identity
#     m_n+1 = (n*m_n + x_{n+1}) / (n+1) = ((n+1)*m_n + (x_{n+1}-m_n)) / (n+1) 
#           = m_n + (x_{n+1}-m_n)/(n+1)
# use Welford's_online_algorithm to update variances
#     v_n+1 = [n*v_n + (x_n+1 - m_n+1)(x_n+1 - m_n)]/(n+1)
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
    new_vars_flat = ((sample_idx-1)*vars_flat + (x_flat - means_flat) * (x_flat - new_means_flat)) / sample_idx
    return AutoStepStats(
        n_pot_evals, n_samples, AutoStepAdaptStats(
            sample_idx, new_mean_step_size, new_mean_acc_prob, new_means_flat,
            new_vars_flat)
        )
