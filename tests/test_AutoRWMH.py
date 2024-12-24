import unittest
import jax
from jax import random
from jax import numpy as jnp
from autosteppy import autorwmh

class TestAutoRWMH(unittest.TestCase):

    def test_involution(self):
        d = 4
        rng_key = random.key(321)
        rng_key, x_key, p_key, step_key = random.split(rng_key, 4) 
        x = random.normal(x_key, d)
        p = random.normal(p_key, d)
        step_size = random.exponential(step_key)

        r = autorwmh.AutoRWMH()
        s = autorwmh.AutoRWMHState(x, p, step_size, rng_key)
        s_new = r.involution(s)
        s_prev = r.involution(s)
        self.assertTrue(jax.tree.all(jax.tree.map(jnp.allclose, s_new, s_prev)))

if __name__ == '__main__':
    unittest.main()