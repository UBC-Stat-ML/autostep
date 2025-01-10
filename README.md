[![Build Status](https://github.com/UBC-Stat-ML/autostep/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/UBC-Stat-ML/autostep/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/UBC-Stat-ML/autostep/branch/main/graph/badge.svg)](https://codecov.io/gh/UBC-Stat-ML/autostep)

# `autostep`

***A numpyro implementation of autoStep methods***

## Installation

```bash
pip install "autostep @ git+ssh://git@github.com/UBC-Stat-ML/autostep.git"
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
J = 8
y = jnp.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
sigma = jnp.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])

def eight_schools(J, sigma, y=None):
    mu = numpyro.sample('mu', dist.Normal(0, 5))
    tau = numpyro.sample('tau', dist.HalfCauchy(5))
    with numpyro.plate('J', J):
        theta = numpyro.sample('theta', dist.Normal(mu, tau))
        numpyro.sample('obs', dist.Normal(theta, sigma), obs=y)

# instantiate sampler and run
n_rounds = 16
n_warmup, n_keep = utils.split_n_rounds(n_rounds) # translate rounds to warmup/keep
kernel = AutoMALA(eight_schools) # default: symmetric selector, (log-)random mix preconditioner
mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_keep)
mcmc.run(random.key(9), J, sigma, y=y)
mcmc.print_summary()
```
```
                mean       std    median      5.0%     95.0%     n_eff     r_hat
        mu      6.55      3.00      6.61      2.08     12.07     13.91      1.12
       tau      2.95      2.51      2.31      0.04      6.30     37.44      1.03
  theta[0]      7.56      4.39      7.44     -0.89     14.19     14.30      1.20
  theta[1]      7.28      3.96      7.12      0.34     13.56     33.12      1.12
  theta[2]      6.71      3.93      6.47      0.07     13.04     45.15      1.03
  theta[3]      5.83      4.72      6.52     -1.56     12.87     20.07      1.14
  theta[4]      6.16      4.03      6.46     -0.71     12.64     30.49      1.05
  theta[5]      7.23      4.42      6.93     -1.63     13.64     25.33      1.06
  theta[6]      7.63      3.65      7.45      1.46     13.59     23.19      1.13
  theta[7]      7.28      3.86      7.23      0.60     12.96     34.00      1.09
```

## TODO

- Jittered step sizes

## References

Biron-Lattes, M., Surjanovic, N., Syed, S., Campbell, T., & Bouchard-Côté, A.. (2024). 
[autoMALA: Locally adaptive Metropolis-adjusted Langevin algorithm](https://proceedings.mlr.press/v238/biron-lattes24a.html). 
*Proceedings of The 27th International Conference on Artificial Intelligence and Statistics*, 
in *Proceedings of Machine Learning Research* 238:4600-4608.

Liu, T., Surjanovic, N., Biron-Lattes, M., Bouchard-Côté, A., & Campbell, T. (2024). 
[AutoStep: Locally adaptive involutive MCMC](https://arxiv.org/abs/2410.18929). arXiv preprint arXiv:2410.18929.
