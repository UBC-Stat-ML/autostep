from functools import partial
from tests import utils as testutils

import unittest

from jax import random
from jax import numpy as jnp

from numpyro.infer import MCMC

from autostep import autohmc
from autostep import autorwmh
from autostep import utils


class TestTempering(unittest.TestCase):

    TESTED_KERNELS = (
        autorwmh.AutoRWMH,
        autohmc.AutoMALA,
        partial(autohmc.AutoHMC, n_leapfrog_steps=32)
    )
    
    def test_tempered_moments(self):
        n_rounds = 14
        n_warmup, n_keep = utils.split_n_rounds(n_rounds) # translate rounds to warmup/keep
        model, model_args, model_kwargs = testutils.toy_conjugate_normal()
        rng_key = random.key(321453)
        for inv_temp in jnp.array([0., 0.25, 0.75, 1.0]):
            true_var = jnp.reciprocal(inv_temp + model_args[0] ** (-2))
            true_mean = inv_temp * model_args[1][0] * true_var
            for kernel_type in self.TESTED_KERNELS:
                with self.subTest(inv_temp=inv_temp, kernel_type=kernel_type):
                    rng_key, mcmc_key = random.split(rng_key) 
                    kernel = kernel_type(model, init_inv_temp=inv_temp)
                    mcmc = MCMC(
                        kernel, 
                        num_warmup=n_warmup, 
                        num_samples=n_keep, 
                        progress_bar=False
                    )
                    mcmc.run(mcmc_key, *model_args, **model_kwargs)
                    adapt_stats=mcmc.last_state.stats.adapt_stats
                    self.assertTrue(
                        jnp.allclose(adapt_stats.sample_mean, true_mean, atol=0.3, rtol=0.1) # need atol to handle mean=0 for inv_temp=0
                    )
                    self.assertTrue(
                        jnp.allclose(adapt_stats.sample_var, true_var, rtol=0.15)
                    )

if __name__ == '__main__':
    unittest.main()
