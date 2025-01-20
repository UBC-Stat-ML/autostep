from tests import utils as testutils

from functools import partial
import unittest

import jax
from jax import random
from jax import numpy as jnp

from numpyro.infer import MCMC

from autostep import autostep
from autostep import autohmc
from autostep import autorwmh
from autostep import selectors
from autostep import statistics
from autostep import utils


class TestKernels(unittest.TestCase):

    TESTED_KERNELS = (
        autorwmh.AutoRWMH,
        autohmc.AutoMALA,
        partial(autohmc.AutoHMC, n_leapgrog_steps=10)
    )
    
    def test_involution(self):
        d = 4
        rng_key = random.key(321)
        for i in range(10):
            for kernel_class in self.TESTED_KERNELS:
                with self.subTest(kernel_class=kernel_class, i=i):
                    rng_key, x_key, v_key, prec_key, step_key = random.split(rng_key, 5) 
                    r = kernel_class(potential_fn=testutils.gaussian_potential)
                    s = r.init(rng_key, 0, random.normal(x_key, d), (), ())
                    s = s._replace(
                        v_flat = random.normal(v_key, d),
                        base_step_size = random.exponential(step_key))
                    diag_precond = 0.1 * random.exponential(prec_key, d)
                    step_size = utils.step_size(s.base_step_size, -1)
                    s_half = r.involution_main(step_size, s, diag_precond)
                    s_one = r.involution_aux(step_size, s_half, diag_precond)
                    s_onehalf = r.involution_main(step_size, s_one, diag_precond)
                    s_two = r.involution_aux(step_size, s_onehalf, diag_precond)
                    self.assertTrue(jax.tree.all(jax.tree.map(partial(jnp.allclose, atol=1e-4), s_two, s)))

    def test_moments(self):

        init_val = jnp.array([1., 2.])
        rng_key = random.key(1234)
        true_mean = 2.
        true_var = 0.5
        n_rounds = 14
        n_warmup, n_keep = utils.split_n_rounds(n_rounds)
        tol = 0.05
        for kernel_class in self.TESTED_KERNELS:
            for sel in (
                    selectors.FixedStepSizeSelector(),
                    selectors.AsymmetricSelector(),
                    selectors.SymmetricSelector()
                ):
                with self.subTest(kernel_class=kernel_class, sel=sel):
                    rng_key, run_key = random.split(rng_key)
                    kernel = kernel_class(potential_fn=testutils.gaussian_potential, selector=sel)
                    mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_keep, progress_bar=False)
                    mcmc.run(run_key, init_params=init_val)
                    state = mcmc.last_state
                    stats = state.stats
                    adapt_stats = stats.adapt_stats
                    self.assertEqual(stats.n_samples, n_warmup+n_keep)
                    self.assertEqual(adapt_stats.sample_idx, n_keep)
                    self.assertEqual(n_keep, jnp.shape(mcmc.get_samples())[0])
                    self.assertTrue(jnp.allclose(state.estimated_std_devs, jnp.sqrt(true_var), atol=tol, rtol=tol))
                    self.assertTrue(jnp.allclose(adapt_stats.means_flat, true_mean, atol=tol, rtol=tol))
                    self.assertTrue(jnp.allclose(adapt_stats.vars_flat, true_var, atol=tol, rtol=tol))

    def test_numpyro_model(self):
        n_rounds = 10
        n_warmup, n_keep = utils.split_n_rounds(n_rounds)
        for kernel_class in self.TESTED_KERNELS:
                with self.subTest(kernel_class=kernel_class):
                    kernel = kernel_class(testutils.toy_unid)
                    mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_keep, progress_bar=False)
                    mcmc.run(random.key(9), 100, n_heads=50)
                    samples = mcmc.get_samples()
                    mean_p_prod = (samples["p1"] * samples["p2"]).mean()
                    self.assertTrue(abs(mean_p_prod - 0.5) < 0.05)

if __name__ == '__main__':
    unittest.main()
