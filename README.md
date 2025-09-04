[![Build Status](https://github.com/UBC-Stat-ML/autostep/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/UBC-Stat-ML/autostep/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/UBC-Stat-ML/autostep/branch/main/graph/badge.svg)](https://codecov.io/gh/UBC-Stat-ML/autostep)

# `autostep`

***A NumPyro-compatible JAX implementation of AutoStep samplers***

## Installation

```bash
pip install "autostep @ git+https://github.com/UBC-Stat-ML/autostep.git"
```

## Eight-schools example

```python
from jax import random
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC
from autostep import preconditioning
from autostep.autohmc import AutoMALA, AutoHMC
from autostep import utils

# define model
y = jnp.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
sigma = jnp.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])

def eight_schools(sigma, y=None):
    mu = numpyro.sample('mu', dist.Normal(0, 5))
    tau = numpyro.sample('tau', dist.HalfCauchy(5))
    with numpyro.plate('J', len(sigma)):
        theta = numpyro.sample('theta', dist.Normal(mu, tau))
        numpyro.sample('obs', dist.Normal(theta, sigma), obs=y)

# instantiate sampler and run
n_rounds = 14
n_warmup, n_keep = utils.split_n_rounds(n_rounds) # translate rounds to warmup/keep
kernel = AutoHMC(
    eight_schools,
    n_leapfrog_steps=32,
    preconditioner = preconditioning.FixedDensePreconditioner()
)
mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_keep)
mcmc.run(random.key(9), sigma, y=y)
mcmc.print_summary()
```
```
sample: 100%|███| 32766/32766 [00:58<00:00, 561.27it/s, base_step 1.50e-01, rev_rate=0.89, acc_prob=0.81]

                mean       std    median      5.0%     95.0%     n_eff     r_hat
        mu      4.35      3.34      4.40     -1.06      9.85  11948.87      1.00
       tau      3.67      3.34      2.82      0.02      7.83    561.33      1.00
  theta[0]      6.26      5.66      5.69     -2.43     15.03   3694.65      1.00
  theta[1]      4.85      4.72      4.77     -2.67     12.64  12237.45      1.00
  theta[2]      3.92      5.39      4.15     -4.70     12.35  10668.33      1.00
  theta[3]      4.74      4.82      4.72     -3.21     12.15  14035.57      1.00
  theta[4]      3.59      4.74      3.85     -3.84     11.44   6886.19      1.00
  theta[5]      4.00      4.86      4.17     -3.73     11.73  10520.05      1.00
  theta[6]      6.34      5.12      5.86     -1.64     14.61   3060.37      1.00
  theta[7]      4.83      5.47      4.75     -3.86     12.89  12227.15      1.00
```

## TODO

- autoHMC with randomized number of steps (RHMC)
- Re-implement the `MixDiagonalPreconditioner` in the new framework

## References

Biron-Lattes, M., Surjanovic, N., Syed, S., Campbell, T., & Bouchard-Côté, A.. (2024). 
[autoMALA: Locally adaptive Metropolis-adjusted Langevin algorithm](https://proceedings.mlr.press/v238/biron-lattes24a.html). 
*Proceedings of The 27th International Conference on Artificial Intelligence and Statistics*, 
in *Proceedings of Machine Learning Research* 238:4600-4608.

Liu, T., Surjanovic, N., Biron-Lattes, M., Bouchard-Côté, A., & Campbell, T. (2024). 
[AutoStep: Locally adaptive involutive MCMC](https://openreview.net/forum?id=QUOwuRtf61). ICML 2025.
