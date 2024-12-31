import unittest
import jax
from jax import random
from jax import numpy as jnp

from autostep import autorwmh
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


if __name__ == '__main__':
    unittest.main()