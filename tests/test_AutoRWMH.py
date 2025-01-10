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

    def test_involution(self):
        d = 4
        rng_key = random.key(321)
        rng_key, x_key, v_key, prec_key, step_key = random.split(rng_key, 5) 
        x = random.normal(x_key, d)
        v_flat = random.normal(v_key, d)
        diag_precond = random.exponential(prec_key, d)
        r = autorwmh.AutoRWMH()
        s = autostep.AutoStepState(x, v_flat, 0., rng_key, statistics.AutoStepStats(), 1.0, None)
        step_size = utils.step_size(s.base_step_size, -1)
        s_half = r.involution_main(step_size, s, diag_precond)
        s_one = r.involution_aux(s_half)
        s_onehalf = r.involution_main(step_size, s_one, diag_precond)
        s_two = r.involution_aux(s_onehalf)
        self.assertTrue(jax.tree.all(jax.tree.map(jnp.allclose, s_two, s)))

    def test_moments(self):
        def f(x):
            return ((x - 2) ** 2).sum()
        init_val = jnp.array([1., 2.])
        rng_key = random.key(1234)
        true_mean = 2.
        true_var = 0.5
        n_rounds = 13
        n_warmup, n_keep = utils.split_n_rounds(n_rounds)
        tol = 0.05
        for sel in (
                selectors.FixedStepSizeSelector(),
                selectors.AsymmetricSelector(),
                selectors.SymmetricSelector()
            ):
            with self.subTest(sel=sel):
                rng_key, run_key = random.split(rng_key)
                kernel = autorwmh.AutoRWMH(potential_fn=f, selector=sel)
                mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_keep, progress_bar=False)
                mcmc.run(run_key, init_params=init_val)
                stats = mcmc.last_state.stats
                adapt_stats = stats.adapt_stats
                self.assertEqual(stats.n_samples, n_warmup+n_keep)
                self.assertEqual(adapt_stats.sample_idx, n_keep)
                self.assertEqual(n_keep, jnp.shape(mcmc.get_samples())[0])
                self.assertTrue(jnp.allclose(adapt_stats.means_flat, true_mean, atol=tol, rtol=tol))
                self.assertTrue(jnp.allclose(adapt_stats.vars_flat, true_var, atol=tol, rtol=tol))
        
        # test reducibility of autoRWMH with asymmetric selector when starting at the mode
        init_val = jnp.array([true_mean, true_mean])
        rng_key, run_key = random.split(rng_key)
        kernel = autorwmh.AutoRWMH(potential_fn=f, selector=selectors.AsymmetricSelector())
        mcmc = MCMC(kernel, num_warmup=0, num_samples=2**10, progress_bar=False)
        mcmc.run(run_key, init_params=init_val)
        self.assertTrue(jnp.all(mcmc.last_state.x == true_mean))

    def test_numpyro_model(self):
        n_rounds = 10
        n_warmup, n_keep = utils.split_n_rounds(n_rounds)
        kernel = autorwmh.AutoRWMH(testutils.toy_unid)
        mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_keep, progress_bar=False)
        mcmc.run(random.key(9), 100, n_heads=50)
        samples = mcmc.get_samples()
        mean_p_prod = (samples["p1"] * samples["p2"]).mean()
        self.assertTrue(abs(mean_p_prod - 0.5) < 0.05)

if __name__ == '__main__':
    unittest.main()