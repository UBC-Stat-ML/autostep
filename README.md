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
from autostep.autohmc import AutoMALA
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
n_rounds = 16
n_warmup, n_keep = utils.split_n_rounds(n_rounds) # translate rounds to warmup/keep
kernel = AutoMALA(eight_schools) # default: symmetric selector, (log-)random mix preconditioner
mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_keep)
mcmc.run(random.key(9), sigma, y=y)
mcmc.print_summary()
```
```
                mean       std    median      5.0%     95.0%     n_eff     r_hat
        mu      4.35      3.46      4.51     -1.43      9.61     79.59      1.01
       tau      3.85      3.39      2.92      0.07      8.52    160.44      1.01
  theta[0]      6.24      5.50      5.87     -1.99     15.00    110.23      1.01
  theta[1]      4.87      4.70      4.95     -2.80     12.16    156.55      1.00
  theta[2]      3.70      5.76      4.09     -4.88     12.74    168.55      1.00
  theta[3]      4.65      5.50      4.66     -3.40     13.72    135.82      1.01
  theta[4]      3.58      4.81      3.80     -3.98     11.16    170.71      1.00
  theta[5]      3.86      4.82      4.17     -4.14     11.62    158.95      1.00
  theta[6]      6.60      5.64      6.14     -2.01     15.13    101.10      1.01
  theta[7]      5.14      5.75      5.02     -4.09     13.66     94.06      1.01
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
