from tests import utils as testutils

import unittest
from jax import random
from jax import numpy as jnp

from autostep import autohmc

class TestAutoHMC(unittest.TestCase):

    # test gradient function
    def test_gradient(self):
        def true_grad(x):
            return 2*(x-2)
        rng_key = random.key(1234)
        init_val = jnp.array([1., 2.])
        rng_key, run_key = random.split(rng_key)
        kernel = autohmc.AutoMALA(potential_fn=testutils.gaussian_potential)
        _ = kernel.init(run_key, 0, init_val, None, None)
        kernel_grad = kernel.integrator.__closure__[0].cell_contents
        for _ in range(20):
            rng_key, x_key = random.split(rng_key)
            x = random.normal(x_key, 2)
            self.assertTrue(jnp.all(kernel_grad(x) == true_grad(x)))

if __name__ == '__main__':
    unittest.main()