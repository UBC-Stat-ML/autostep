from tests import utils as testutils

import unittest

from jax import random
import jax.numpy as jnp

from numpyro.infer import MCMC

from autostep.autohmc import AutoMALA
from autostep import utils
from autostep import preconditioning

class TestPreconditioning(unittest.TestCase):

    def test_preconditioning(self):
        dim = 3
        rho = 0.9
        tol = 0.15
        n_rounds = 14

        init_vals = jnp.ones(dim)
        S = testutils.make_const_off_diag_corr_mat(dim, rho)
        pot_fn = testutils.make_correlated_Gaussian_potential(S)
        n_warmup, n_keep = utils.split_n_rounds(n_rounds) # translate rounds to warmup/keep
        precs = (
            preconditioning.IdentityDiagonalPreconditioner(),
            preconditioning.FixedDiagonalPreconditioner(),
            preconditioning.FixedDensePreconditioner(),
        )
        all_min_ess = []
        for p in precs:
            with self.subTest(prec_class=type(p)):
                kernel = AutoMALA(potential_fn = pot_fn, preconditioner = p)
                mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_keep, progress_bar=False)
                mcmc.run(random.key(2349895454), init_params=init_vals)
                min_ess = testutils.extremal_diagnostics(mcmc)[1]
                all_min_ess.append(min_ess)
                print(f"{type(p)}: min_ess={min_ess}")
                self.assertGreater(min_ess, 100)
                last_round_used_var = mcmc.last_state.base_precond_state.var
                if preconditioning.is_dense(p):
                    last_round_estimate_var = mcmc.last_state.stats.adapt_stats.sample_var
                    self.assertTrue(jnp.allclose(S, last_round_used_var, rtol=tol))
                    self.assertTrue(jnp.allclose(S, last_round_estimate_var, rtol=tol))
                else:
                    diag_S = jnp.diag(S)
                    last_round_estimate_var = mcmc.last_state.stats.adapt_stats.sample_var
                    self.assertTrue(jnp.allclose(diag_S, last_round_estimate_var, rtol=tol))
                    if not isinstance(p, preconditioning.IdentityDiagonalPreconditioner):
                        self.assertTrue(jnp.allclose(diag_S, last_round_used_var, rtol=tol))

        # check that Dense gives best performance
        self.assertEqual(all_min_ess[-1], max(all_min_ess))

if __name__ == '__main__':
    unittest.main()
