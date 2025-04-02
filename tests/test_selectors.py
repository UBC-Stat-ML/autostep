from tests import utils as testutils

import unittest
from jax import random
from jax import numpy as jnp

from numpyro.infer import MCMC

from autostep import autohmc
from autostep import selectors
from autostep import utils

class TestSelectors(unittest.TestCase):

    def test_selectors(self):
        asym_sel = selectors.AsymmetricSelector()
        sym_sel = selectors.SymmetricSelector()
        fix_sel = selectors.FixedStepSizeSelector()
        params = jnp.array([-1.5, -0.5])

        # sym and asym should give different decisions here
        log_diff = jnp.float32(0.8)
        self.assertTrue(asym_sel.should_grow(params, log_diff))
        self.assertFalse(sym_sel.should_grow(params, log_diff))
        self.assertFalse(fix_sel.should_grow(params, log_diff))
        self.assertFalse(asym_sel.should_shrink(params, log_diff))
        self.assertFalse(sym_sel.should_shrink(params, log_diff))
        self.assertFalse(fix_sel.should_shrink(params, log_diff))

        # sym and asym should agree in the following 2 cases
        log_diff = jnp.float32(-2)
        self.assertFalse(asym_sel.should_grow(params, log_diff))
        self.assertFalse(sym_sel.should_grow(params, log_diff))
        self.assertTrue(asym_sel.should_shrink(params, log_diff))
        self.assertTrue(sym_sel.should_shrink(params, log_diff))

        log_diff = jnp.float32(0)
        self.assertTrue(asym_sel.should_grow(params, log_diff))
        self.assertTrue(sym_sel.should_grow(params, log_diff))
        self.assertFalse(asym_sel.should_shrink(params, log_diff))
        self.assertFalse(sym_sel.should_shrink(params, log_diff))

    # fixed selector preserves the initial base step size
    def test_fixed_step_size_preserved(self):
        init_val = jnp.array([1., 2.])
        step_key, run_key = random.split(random.key(54))
        init_base_step_size = random.exponential(step_key)
        kernel = autohmc.AutoMALA(
            potential_fn=testutils.gaussian_potential,
            init_base_step_size = init_base_step_size,
            selector=selectors.FixedStepSizeSelector()
        )
        n_warmup, n_keep = utils.split_n_rounds(10)
        mcmc = MCMC(
            kernel, num_warmup=n_warmup, num_samples=n_keep, progress_bar=False
        )
        mcmc.run(run_key, init_params=init_val)
        self.assertTrue(mcmc.last_state.base_step_size == init_base_step_size)

if __name__ == '__main__':
    unittest.main()