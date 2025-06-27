[![Build Status](https://github.com/UBC-Stat-ML/autostep/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/UBC-Stat-ML/autostep/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/UBC-Stat-ML/autostep/branch/main/graph/badge.svg)](https://codecov.io/gh/UBC-Stat-ML/autostep)

# `autostep`

***A NumPyro-compatible JAX implementation of autoStep samplers***

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
sample: 100%|███| 32766/32766 [00:56<00:00, 577.20it/s, base_step_size 1.59e-01. mean_acc_prob=0.81]

                mean       std    median      5.0%     95.0%     n_eff     r_hat
        mu      4.44      3.29      4.46     -0.93      9.91   8377.35      1.00
       tau      3.71      3.34      2.84      0.02      8.09    451.99      1.00
  theta[0]      6.31      5.66      5.68     -3.13     14.37   3559.68      1.00
  theta[1]      4.97      4.73      4.94     -2.51     12.62  10851.66      1.00
  theta[2]      3.96      5.39      4.23     -4.20     12.80   8986.34      1.00
  theta[3]      4.80      4.76      4.80     -3.03     12.16  10271.22      1.00
  theta[4]      3.62      4.64      3.89     -3.77     11.00   6365.64      1.00
  theta[5]      4.03      4.84      4.17     -3.62     12.08   9346.41      1.00
  theta[6]      6.44      5.22      5.91     -1.97     14.45   3388.24      1.00
  theta[7]      4.95      5.43      4.84     -3.48     13.50  11422.42      1.00
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
[AutoStep: Locally adaptive involutive MCMC](https://arxiv.org/abs/2410.18929). arXiv preprint arXiv:2410.18929.
