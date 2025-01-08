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
from autostep.autorwmh import AutoRWMH
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
kernel = AutoRWMH(eight_schools) # default: symmetric selector, (log-)random mix preconditioner
mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_keep)
mcmc.run(random.key(9), J, sigma, y=y)
mcmc.print_summary()
```
```
                mean       std    median      5.0%     95.0%     n_eff     r_hat
        mu      4.04      2.85      3.96     -0.98      8.50     59.73      1.04
       tau      3.37      3.21      2.36      0.07      7.85     38.93      1.05
  theta[0]      6.08      5.76      4.99     -2.88     14.66     35.34      1.10
  theta[1]      4.69      4.15      4.28     -1.73     11.11    108.97      1.05
  theta[2]      3.36      5.06      3.52     -4.60     11.09    132.02      1.01
  theta[3]      4.01      4.09      3.81     -3.05     10.43    106.56      1.02
  theta[4]      3.34      4.10      3.48     -3.06      9.98    131.32      1.00
  theta[5]      3.65      4.08      3.69     -2.85     10.56    151.94      1.02
  theta[6]      6.03      4.95      5.25     -1.65     14.15     29.04      1.06
  theta[7]      3.94      4.56      3.88     -2.66     11.59    123.57      1.01
```

## TODO

- Jittered step sizes
- autoMALA
- Github CI

## References

Biron-Lattes, M., Surjanovic, N., Syed, S., Campbell, T., & Bouchard-Côté, A.. (2024). 
[autoMALA: Locally adaptive Metropolis-adjusted Langevin algorithm](https://proceedings.mlr.press/v238/biron-lattes24a.html). 
*Proceedings of The 27th International Conference on Artificial Intelligence and Statistics*, 
in *Proceedings of Machine Learning Research* 238:4600-4608.

Liu, T., Surjanovic, N., Biron-Lattes, M., Bouchard-Côté, A., & Campbell, T. (2024). 
[AutoStep: Locally adaptive involutive MCMC](https://arxiv.org/abs/2410.18929). arXiv preprint arXiv:2410.18929.
