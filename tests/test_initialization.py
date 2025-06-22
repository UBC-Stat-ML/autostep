from tests import utils as testutils

from functools import partial
import unittest

from scipy import stats

import jax
from jax import lax
from jax import random
from jax import numpy as jnp

from numpyro.infer import MCMC

from autostep import autohmc
from autostep import autorwmh
from autostep import preconditioning
from autostep import selectors
from autostep import statistics
from autostep import utils


class TestInitialization(unittest.TestCase):

    TESTED_KERNELS = (
        autorwmh.AutoRWMH,
        autohmc.AutoMALA,
        partial(autohmc.AutoHMC, n_leapfrog_steps=32)
    )

    # check optimizing initial params improves rhat for unid target with many tosses
    def test_optimization(self):
        rng_key = random.key(9999)
        n_flips, n_heads = 100000, 50000
        n_optim = 2048
        prec = preconditioning.FixedDensePreconditioner()
        n_rounds = 12
        n_warmup, n_keep = utils.split_n_rounds(n_rounds)
        for kernel_class in self.TESTED_KERNELS:
            with self.subTest(kernel_class=kernel_class):
                max_rhats = {}
                for N in (0, n_optim):
                    print(f"kernel_class={kernel_class}, n_optim={N}")
                    rng_key, run_key = random.split(rng_key)
                    kernel = kernel_class(
                        testutils.toy_unid, 
                        preconditioner = prec, 
                        n_iter_opt_init_params = N
                    )
                    mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_keep, progress_bar=False)
                    mcmc.run(run_key, n_flips, n_heads=n_heads)
                    max_rhats[N] = testutils.extremal_diagnostics(mcmc)[0]
                self.assertTrue((max_rhats[0] >= max_rhats[n_optim]) or max_rhats[0] < 1.01)
                

if __name__ == '__main__':
    unittest.main()
