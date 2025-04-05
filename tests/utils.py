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
    def pot_fn(x):
        return 0.5*jnp.dot(x.T, jnp.dot(P, x))
    return pot_fn

def toy_unid(n_flips, n_heads=None):
    p1 = numpyro.sample('p1', dist.Uniform())
    p2 = numpyro.sample('p2', dist.Uniform())
    p = p1*p2
    numpyro.sample('n_heads', dist.Binomial(n_flips, p), obs=n_heads)

# Toy conjugate Gaussian example, admits closed form tempering path
# For d in ints, m in R, sigma0 >0, and beta>=0,
#   x ~ pi_beta = N_d(mu(beta), v(beta))
# where
#   mu(beta) := beta m v(beta)
#   v(beta)  := (beta + sigma0^{-2})^{-1}
#
# Ref: Biron-Lattes, Campbell, & Bouchard-Côté (2024)
def toy_conjugate_normal(
        d = jnp.int32(3), 
        m = jnp.float32(2.), 
        sigma0 = jnp.float32(2.)
    ):
    def model(sigma0, y):
        with numpyro.plate('dim', len(y)):
            x = numpyro.sample('x', dist.Normal(scale=sigma0))
            numpyro.sample('obs', dist.Normal(x), obs=y)

    # inputs
    y = jnp.full((d,), m)
    model_args = (sigma0, y)
    model_kwargs = {}
    return model, model_args, model_kwargs


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
