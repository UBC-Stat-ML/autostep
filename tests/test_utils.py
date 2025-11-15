import unittest

from jax import numpy as jnp
from automcmc import utils

class TestUtils(unittest.TestCase):

    def test_round_arithmetic(self):
        for n_rounds in jnp.arange(1,30):
            # print(f"n_rounds={n_rounds}")
            n_warmup, n_keep = utils.split_n_rounds(n_rounds)
            n_samples = n_warmup + n_keep
            self.assertEqual(n_rounds, utils.n_warmup_to_adapt_rounds(n_warmup) + 1)
            self.assertEqual(n_samples, 2**(n_rounds+1)-2)
            self.assertEqual(n_rounds, utils.current_round(n_samples))


if __name__ == '__main__':
    unittest.main()