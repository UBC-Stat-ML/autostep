import unittest
import jax
from jax import numpy as jnp
from autostep import selectors

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

if __name__ == '__main__':
    unittest.main()