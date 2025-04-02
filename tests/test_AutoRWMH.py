from tests import utils as testutils

import unittest
import jax
from jax import random
from jax import numpy as jnp

from numpyro.infer import MCMC

from autostep import autostep
from autostep import autorwmh
from autostep import selectors
from autostep import statistics
from autostep import utils


class TestAutoRWMH(unittest.TestCase):

    # test reducibility of autoRWMH with asymmetric selector when starting at the mode
    def test_reducibility(self):
        rng_key = random.key(1234)
        true_mean = 2.
        init_val = jnp.array([true_mean, true_mean])
        rng_key, run_key = random.split(rng_key)
        kernel = autorwmh.AutoRWMH(
            potential_fn=testutils.gaussian_potential, selector=selectors.AsymmetricSelector())
        mcmc = MCMC(kernel, num_warmup=0, num_samples=2**10, progress_bar=False)
        mcmc.run(run_key, init_params=init_val)
        self.assertTrue(jnp.all(mcmc.last_state.x == init_val))

if __name__ == '__main__':
    unittest.main()