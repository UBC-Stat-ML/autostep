import numpy
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

def gaussian_potential(x):
    return ((x - 2) ** 2).sum()

def make_const_off_diag_corr_mat(dim, rho):
    return jnp.full((dim,dim), rho) + (1-rho)*jnp.eye(dim, dim)

def make_correlated_Gaussian_potential(S=None, dim=None, rho=None):
    S = make_const_off_diag_corr_mat(dim, rho) if S is None else S
    P = jnp.linalg.inv(S)
    @jax.jit
    def pot_fn(x):
        return 0.5*jnp.dot(x.T, jnp.dot(P, x))
    return pot_fn

def toy_unid(n_flips, n_heads=None):
    p1 = numpyro.sample('p1', dist.Uniform())
    p2 = numpyro.sample('p2', dist.Uniform())
    p = p1*p2
    numpyro.sample('n_heads', dist.Binomial(n_flips, p), obs=n_heads)


def extremal_diagnostics(mcmc):
    """
    Compute maximum of the Gelman--Rubin diagnostic across dimensions, and the
    minimum ESS across dimensions.

    :param mcmc: An instance of `numpyro.infer.MCMC`.
    :return: Worst-case Gelman--Rubin (max) and ESS (min) across dimensions.
    """
    grouped_samples = mcmc.get_samples(group_by_chain=True)
    diags = numpyro.diagnostics.summary(grouped_samples)
    max_grd = next(numpy.max(v["r_hat"].max() for v in diags.values()))
    min_ess = next(numpy.min(v["n_eff"].min() for v in diags.values()))
    return (max_grd, min_ess)
