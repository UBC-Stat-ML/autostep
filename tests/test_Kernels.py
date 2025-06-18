from tests import utils as testutils

from functools import partial
import unittest

from scipy import stats

import jax
from jax import lax
from jax import random
from jax import numpy as jnp

from numpyro.infer import MCMC

from autostep import autohmc
from autostep import autorwmh
from autostep import preconditioning
from autostep import selectors
from autostep import statistics
from autostep import utils

def rand_sample_var(rng_key, var_shape):
    N = random.normal(rng_key, var_shape)
    if len(var_shape) == 2:
        return N.T @ N
    else:
        return N * N

# sequentially sample multiple times, discard intermediate states
def loop_sample(kernel, n_refresh, kernel_state):
    return lax.scan(
        lambda state, _: (kernel.sample(state,(),{}), None),
        kernel_state,
        length=n_refresh
    )[0]

# run MCMC on an isotropic multivariate normal starting from the stationary dist
# collect only the last state, and only its first coordinate
def run_and_collect_last_sample(
        kernel, 
        dim, 
        true_mean, 
        true_sd, 
        n_warmup,
        n_refresh, 
        rng_key
    ):
    run_key, init_key = random.split(rng_key)

    # draw an iid sample from the target
    init_val = true_mean + true_sd*random.normal(init_key, dim)

    # initialize a kernel state with warmup to tune parameters
    # note: num_samples = 0 fails.
    mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=1, progress_bar=False)
    mcmc.run(run_key, init_params=init_val)
    kernel_state = mcmc.last_state

    # reset the initial state to the sample from the target 
    kernel_state = kernel_state._replace(x=init_val)

    # run
    kernel_state = loop_sample(kernel, n_refresh, kernel_state)
    return kernel_state.x[0]

class TestKernels(unittest.TestCase):

    TESTED_KERNELS = (
        autorwmh.AutoRWMH,
        autohmc.AutoMALA,
        partial(autohmc.AutoHMC, n_leapfrog_steps=32)
    )

    TESTED_PRECONDITIONERS = (
        preconditioning.IdentityDiagonalPreconditioner(),
        preconditioning.FixedDiagonalPreconditioner(),
        preconditioning.FixedDensePreconditioner()
    )

    TESTED_SELECTORS = (
        # no guarantee that other selectors produce good samples
        selectors.AsymmetricSelector(), selectors.SymmetricSelector()
    )
    
    def test_involution(self):
        d = 4
        rng_key = random.key(321)
        for prec in self.TESTED_PRECONDITIONERS:
            for i in range(10):
                rng_key, x_key, v_key, var_key, prec_key, step_key = random.split(rng_key, 6)
                init_x = random.normal(x_key, d)
                init_p_flat = random.normal(v_key, d)
                var_shape = (d,d) if preconditioning.is_dense(prec) else (d,)
                adapt_stats = statistics.AutoStepAdaptStats(
                    100000,
                    0.1 * random.exponential(step_key), 
                    0., 
                    jnp.zeros_like(init_x),
                    rand_sample_var(var_key, var_shape)
                )
                for kernel_class in self.TESTED_KERNELS:
                    with self.subTest(prec=prec, i=i, kernel_class=kernel_class):
                        r = kernel_class(
                            potential_fn=testutils.gaussian_potential,
                            preconditioner = prec
                        )
                        # rng_key is not used because we are passing initial parameters
                        s = r.init(rng_key, 0, init_x, (), ()) 
                        s = s._replace(
                            p_flat = init_p_flat, 
                            stats=s.stats._replace(adapt_stats=adapt_stats)
                        )
                        s = r.adapt(s, True) # populates base_precond_state and base_step_size
                        precond_state = prec.maybe_alter_precond_state(
                            s.base_precond_state, prec_key
                        )
                        step_size = utils.step_size(s.base_step_size, 0)
                        s_half = r.involution_main(step_size, s, precond_state)
                        s_one = r.involution_aux(step_size, s_half, precond_state)
                        s_onehalf = r.involution_main(step_size, s_one, precond_state)
                        s_two = r.involution_aux(step_size, s_onehalf, precond_state)
                        self.assertTrue(
                            jnp.allclose(s_two.x, s.x, rtol=1e-3),
                            msg=f"s.x={s.x} but s_two.x={s_two.x}"
                        )
                        self.assertTrue(
                            jnp.allclose(s_two.p_flat, s.p_flat, rtol=1e-3),
                            msg=f"s.p_flat={s.p_flat} but s_two.p_flat={s_two.p_flat}"
                        )

    def test_moments(self):

        init_val = jnp.array([1., 2.])
        rng_key = random.key(23)
        true_mean = 2.
        true_var = 0.5
        true_sd = jnp.sqrt(true_var)
        n_rounds = 14
        n_warmup, n_keep = utils.split_n_rounds(n_rounds)
        tol = 0.15
        for kernel_class in self.TESTED_KERNELS:
            for sel in self.TESTED_SELECTORS:
                with self.subTest(kernel_class=kernel_class, sel_type=type(sel)):
                    print(f"kernel_class={kernel_class}, sel_type={type(sel)}")
                    rng_key, run_key = random.split(rng_key)
                    kernel = kernel_class(potential_fn=testutils.gaussian_potential, selector=sel)
                    mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_keep, progress_bar=False)
                    mcmc.run(run_key, init_params=init_val)
                    state = mcmc.last_state
                    stats = state.stats
                    adapt_stats = stats.adapt_stats
                    self.assertEqual(stats.n_samples, n_warmup+n_keep)
                    self.assertEqual(adapt_stats.sample_idx, n_keep)
                    self.assertEqual(n_keep, jnp.shape(mcmc.get_samples())[0])
                    var_chol_tril = state.base_precond_state.var_chol_tril
                    self.assertTrue(
                        jnp.allclose(var_chol_tril, true_sd, rtol=tol),
                        msg=f"var_chol_tril={var_chol_tril} but true_sd={true_sd}"
                    )
                    self.assertTrue(
                        jnp.allclose(adapt_stats.sample_mean, true_mean, rtol=tol),
                        msg=f"sample_mean={adapt_stats.sample_mean} but true_mean={true_mean}"
                    )
                    self.assertTrue(
                        jnp.allclose(adapt_stats.sample_var, true_var, rtol=tol),
                        msg=f"sample_var={adapt_stats.sample_var} but true_var={true_var}"
                    )
    
    # invariance test on an isotropic multivariate normal
    def test_invariance(self):
        dim = 4
        true_mean = 2.
        true_var = 0.5
        true_sd = jnp.sqrt(true_var)
        true_cdf = partial(stats.norm.cdf, loc=true_mean, scale=true_sd)
        n_warmup = utils.split_n_rounds(10)[0]
        n_refresh = 1024
        n_samples = 1024
        pval_threshold = 0.01
        rng_key = random.key(2)
        
        for kernel_class in self.TESTED_KERNELS:
            for prec in self.TESTED_PRECONDITIONERS:
                for sel in self.TESTED_SELECTORS:
                    with self.subTest(kernel_class=kernel_class, prec_type=type(prec), sel_type=type(sel)):
                        print(f"kernel_class={kernel_class}, prec_type={type(prec)}, sel_type={type(sel)}")
                        rng_key, exp_key = random.split(rng_key)
                        kernel = kernel_class(
                            potential_fn=testutils.gaussian_potential, 
                            selector=sel,
                            preconditioner = prec
                        )
                        vmap_fn = jax.vmap(
                            partial(
                                run_and_collect_last_sample, 
                                kernel, 
                                dim, 
                                true_mean, 
                                true_sd, 
                                n_warmup, 
                                n_refresh
                            )
                        )
                        run_keys = random.split(exp_key, n_samples)
                        mcmc_samples = vmap_fn(run_keys)
                        ks_res = stats.ks_1samp(mcmc_samples, true_cdf)
                        self.assertGreater(ks_res.pvalue, pval_threshold)

    def test_numpyro_model(self):
        n_rounds = 10
        n_warmup, n_keep = utils.split_n_rounds(n_rounds)
        for kernel_class in self.TESTED_KERNELS:
                with self.subTest(kernel_class=kernel_class):
                    kernel = kernel_class(testutils.toy_unid)
                    mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_keep, progress_bar=False)
                    mcmc.run(random.key(9), 100, n_heads=50)
                    samples = mcmc.get_samples()
                    mean_p_prod = (samples["p1"] * samples["p2"]).mean()
                    self.assertTrue(abs(mean_p_prod - 0.5) < 0.05)

if __name__ == '__main__':
    unittest.main()
