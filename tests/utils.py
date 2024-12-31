import numpyro
import numpyro.distributions as dist

def toy_unid(n_flips, n_heads=None):
    p1 = numpyro.sample('p1', dist.Uniform())
    p2 = numpyro.sample('p2', dist.Uniform())
    p = p1*p2
    numpyro.sample('n_heads', dist.Binomial(n_flips, p), obs=n_heads)