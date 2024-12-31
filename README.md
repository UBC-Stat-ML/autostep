# `autostep`

***A numpyro implementation of autoStep methods***

## Installation

```bash
pip install "autostep @ git+ssh://git@github.com/miguelbiron/autostep.git"
```

## Eight-schools example

```python
from jax import random
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC
from autostep.autorwmh import AutoRWMH

J = 8
y = jnp.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
sigma = jnp.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])

def eight_schools(J, sigma, y=None):
    mu = numpyro.sample('mu', dist.Normal(0, 5))
    tau = numpyro.sample('tau', dist.HalfCauchy(5))
    with numpyro.plate('J', J):
        theta = numpyro.sample('theta', dist.Normal(mu, tau))
        numpyro.sample('obs', dist.Normal(theta, sigma), obs=y)
kernel = AutoRWMH(eight_schools, base_step_size=0.4)
mcmc = MCMC(kernel, num_warmup=0, num_samples=2**18)
mcmc.run(random.key(9), J, sigma, y=y)
mcmc.print_summary()
```
```
                mean       std    median      5.0%     95.0%     n_eff     r_hat
        mu      4.50      3.56      4.62     -0.77     11.03     70.86      1.02
       tau      4.08      3.50      3.15      0.08      8.61     78.49      1.01
  theta[0]      6.54      6.36      6.03     -3.03     16.59    145.01      1.00
  theta[1]      5.12      4.97      5.20     -3.05     13.32    119.42      1.00
  theta[2]      3.81      5.86      4.25     -6.10     12.66    107.98      1.00
  theta[3]      5.08      5.39      5.11     -4.17     13.88    195.02      1.01
  theta[4]      3.53      5.11      4.11     -4.79     11.64     87.79      1.00
  theta[5]      4.11      5.31      4.53     -4.89     12.29    149.40      1.02
  theta[6]      6.66      5.14      6.24     -1.34     14.74    145.33      1.01
  theta[7]      5.36      6.23      5.11     -4.64     14.43    115.63      1.01
```

### autoStep diagnostics

```python
>>> print(f"Mean step-size: {mcmc.last_state.stats.mean_step_size}")
Mean step-size: 0.3895134925842285
>>> print(f"Mean acceptance probability: {mcmc.last_state.stats.mean_acc_prob}")
Mean acceptance probability: 0.4472614824771881
```

## TODO

- Preconditioning
- Jittered step sizes
- autoMALA

## References

Biron-Lattes, M., Surjanovic, N., Syed, S., Campbell, T., & Bouchard-Côté, A.. (2024). 
[autoMALA: Locally adaptive Metropolis-adjusted Langevin algorithm](https://proceedings.mlr.press/v238/biron-lattes24a.html). 
*Proceedings of The 27th International Conference on Artificial Intelligence and Statistics*, 
in *Proceedings of Machine Learning Research* 238:4600-4608.

Liu, T., Surjanovic, N., Biron-Lattes, M., Bouchard-Côté, A., & Campbell, T. (2024). 
[AutoStep: Locally adaptive involutive MCMC](https://arxiv.org/abs/2410.18929). arXiv preprint arXiv:2410.18929.
