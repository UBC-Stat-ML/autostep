from tests import utils as testutils

import unittest
import jax
from jax import random
from jax import numpy as jnp

from numpyro.infer import MCMC

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
        s = autorwmh.AutoRWMHState(x, v_flat, 0., rng_key, statistics.AutoStepStats())
        step_size = utils.step_size(r.base_step_size, -1)
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
        tol = 0.05
        for sel in (
                selectors.FixedStepSizeSelector(),
                selectors.AsymmetricSelector(),
                selectors.SymmetricSelector()
            ):
            with self.subTest(sel=sel):
                rng_key, run_key = random.split(rng_key)
                kernel = autorwmh.AutoRWMH(potential_fn=f, selector=sel)
                mcmc = MCMC(kernel, num_warmup=0, num_samples=2**12, progress_bar=False)
                mcmc.run(run_key, init_params=init_val)
                stats = mcmc.last_state.stats
                self.assertTrue(jnp.allclose(stats.means_flat, true_mean, atol=tol, rtol=tol))
                self.assertTrue(jnp.allclose(stats.vars_flat, true_var, atol=tol, rtol=tol))
        
        # test reducibility of autoRWMH with asymmetric selector when starting at the mode
        init_val = jnp.array([true_mean, true_mean])
        rng_key, run_key = random.split(rng_key)
        kernel = autorwmh.AutoRWMH(potential_fn=f, selector=selectors.AsymmetricSelector())
        mcmc = MCMC(kernel, num_warmup=0, num_samples=2**10, progress_bar=False)
        mcmc.run(run_key, init_params=init_val)
        self.assertTrue(jnp.all(mcmc.last_state.x == true_mean))

    def test_numpyro_model(self):
        kernel = autorwmh.AutoRWMH(testutils.toy_unid, base_step_size=0.55)
        mcmc = MCMC(kernel, num_warmup=0, num_samples=2**13, progress_bar=False)
        mcmc.run(random.key(9), 100, n_heads=50)
        samples = mcmc.get_samples()
        self.assertTrue(abs((samples["p1"] * samples["p2"]).mean() - 0.5) < 0.02)

if __name__ == '__main__':
    unittest.main()