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
        rng_key, x_key, v_key, step_key = random.split(rng_key, 4) 
        x = random.normal(x_key, d)
        v_flat = random.normal(v_key, d)
        r = autorwmh.AutoRWMH()
        s = autorwmh.AutoRWMHState(x, v_flat, 0., rng_key, statistics.AutoStepStats())
        step_size = utils.step_size(r._base_step_size, -1)
        s_half = r.involution_main(step_size, s)
        s_one = r.involution_aux(s_half)
        s_onehalf = r.involution_main(step_size, s_one)
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
                xs = mcmc.get_samples()
                self.assertTrue(jnp.allclose(xs.mean(axis=0), true_mean, atol=tol, rtol=tol))
                self.assertTrue(jnp.allclose(xs.var(axis=0), true_var, atol=tol, rtol=tol))
        
        # test reducibility of autoRWMH with asymmetric selector when starting at the mode
        init_val = jnp.array([true_mean, true_mean])
        rng_key, run_key = random.split(rng_key)
        kernel = autorwmh.AutoRWMH(potential_fn=f, selector=selectors.AsymmetricSelector())
        mcmc = MCMC(kernel, num_warmup=0, num_samples=2**10, progress_bar=False)
        mcmc.run(run_key, init_params=init_val)
        self.assertTrue(jnp.all(mcmc.last_state.x == true_mean))

if __name__ == '__main__':
    unittest.main()