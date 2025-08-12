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
from autostep import initialization
from autostep import tempering
from autostep import utils


class TestInitialization(unittest.TestCase):

    # check optimizing initial params improves rhat for autoRWMH on the unid
    # target with many tosses
    def test_optimization(self):
        n_flips, n_heads = 100000, 50000
        init_params = {'p1': jnp.float32(-3), 'p2': jnp.float32(3)}
        all_initialization_settings = (
            None,
            {'strategy': "L-BFGS", 'params': {'n_iter': 64}},
            {'strategy': "ADAM", 'params': {'n_iter': 2048}}
        )
        prec = preconditioning.FixedDensePreconditioner()
        n_rounds = 12
        n_warmup, n_keep = utils.split_n_rounds(n_rounds)
        max_rhats = {}
        for seed in (9,99,999,9999):
            rng_key = random.key(seed)
            max_rhats = list()
            for initialization_settings in all_initialization_settings:
                str = None if initialization_settings is None else initialization_settings['strategy']
                with self.subTest(seed=seed, strategy=str):
                    rng_key, run_key = random.split(rng_key)
                    kernel = autorwmh.AutoRWMH(
                        testutils.toy_unid, 
                        preconditioner = prec, 
                        initialization_settings = initialization_settings
                    )
                    mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_keep, progress_bar=False)
                    mcmc.run(run_key, n_flips, n_heads=n_heads, init_params=init_params)
                    max_rhats.append(testutils.extremal_diagnostics(mcmc)[0])
            print(max_rhats)
            self.assertTrue(max_rhats[0] >= max_rhats[1])
            self.assertTrue(max_rhats[0] >= max_rhats[2])

    def test_MAP(self):
        rng_key = random.key(90)
        model, model_args, model_kwargs = testutils.make_eight_schools()
        MAP_estimate_unconstrained = initialization.MAP(
            model,
            rng_key,
            model_args,
            model_kwargs,
            options = {'strategy': "L-BFGS", 'params': {'n_iter': 64}}
        )[0]

        # transform to constrained space
        exec_trace = tempering.trace_from_unconst_samples(
            model, 
            model_args, 
            model_kwargs,
            MAP_estimate_unconstrained, 
        )
        MAP_estimate = {
            name: site["value"] for name, site in exec_trace.items() 
            if site["type"] == "sample" and not site["is_observed"]
        }
        self.assertTrue(jnp.allclose(MAP_estimate['mu'], MAP_estimate['theta']))
        self.assertLess(MAP_estimate['tau'], 1e-6)

if __name__ == '__main__':
    unittest.main()
