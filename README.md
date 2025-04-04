[![Build Status](https://github.com/UBC-Stat-ML/autostep/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/UBC-Stat-ML/autostep/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/UBC-Stat-ML/autostep/branch/main/graph/badge.svg)](https://codecov.io/gh/UBC-Stat-ML/autostep)

# `autostep`

***A numpyro implementation of autoStep methods***

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
    preconditioner = preconditioning.MixDensePreconditioner()
)
mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_keep)
mcmc.run(random.key(9), sigma, y=y)
mcmc.print_summary()
```
```
                mean       std    median      5.0%     95.0%     n_eff     r_hat
        mu      4.57      3.19      4.52     -0.42     10.03    960.15      1.00
       tau      3.63      3.43      2.68      0.02      8.06    453.54      1.00
  theta[0]      6.52      5.61      5.80     -2.72     14.60   1082.41      1.00
  theta[1]      5.13      4.62      4.89     -2.76     12.17   1440.08      1.00
  theta[2]      4.03      5.31      4.30     -3.41     12.96   1352.01      1.00
  theta[3]      5.02      4.73      4.86     -2.68     12.57   1561.84      1.00
  theta[4]      3.76      4.70      4.03     -3.38     11.84   1335.61      1.00
  theta[5]      4.23      4.72      4.39     -3.40     11.49   1364.35      1.00
  theta[6]      6.55      5.03      5.95     -1.24     14.48   1019.30      1.00
  theta[7]      5.14      5.48      4.90     -3.11     13.42   1334.00      1.00
```

## TODO

- autoHMC with randomized number of steps (RHMC)

## References

Biron-Lattes, M., Surjanovic, N., Syed, S., Campbell, T., & Bouchard-Côté, A.. (2024). 
[autoMALA: Locally adaptive Metropolis-adjusted Langevin algorithm](https://proceedings.mlr.press/v238/biron-lattes24a.html). 
*Proceedings of The 27th International Conference on Artificial Intelligence and Statistics*, 
in *Proceedings of Machine Learning Research* 238:4600-4608.

Liu, T., Surjanovic, N., Biron-Lattes, M., Bouchard-Côté, A., & Campbell, T. (2024). 
[AutoStep: Locally adaptive involutive MCMC](https://arxiv.org/abs/2410.18929). arXiv preprint arXiv:2410.18929.
