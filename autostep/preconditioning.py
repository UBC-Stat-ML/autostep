from abc import ABC, abstractmethod

from collections import namedtuple

import jax
from jax import lax
from jax import numpy as jnp

###############################################################################
# Preconditioner state
# 
# The most demanding application is for dense preconditioners. To handle both
# AutoRWMH and AutoHMC. Let S = Cov(x), M = S^{-1}, an y~N(0,I). We need,
#   - S, for
#       - AutoHMC: kinetic energy 0.5p^TM^{-1}p = 0.5*jnp.dot(S@p,p)
#       - AutoHMC: LF postition update x' = x + eps M^{-1}p = x + eps Sp
#   - L = chol(S), for 
#       - AutoRWMH: proposal x' = x + epsLy
#   - A factorization of the form M = UU^T for
#       - AutoHMC: sample momentum p = Uy ~ N(0,M)
# 
# The last decomposition can be gotten by manipulating a Cholesky factorization
# of S, since
#     S = LL^T <=> M = S^{-1} = (L^T)^{-1}L^{-1} = UU^T  
# This approach---an O(d^3) chol followed by an O(d^2) triangular solve---is 
# much more efficient than doing 2 O(d^3) ops in chol(inv(S)). We don't need
# U to be lower triangular or even triangular. Any decomposition M=UU^T works. 
#
# An example to check that this works:
# 
# # sample a symmetric mat (a.s. posdef) to use as variance
# var_key, sim_key = random.split(random.key(2176543))
# N = random.normal(var_key, (2,2))
# S = N.T @ N

# # chol of variance mat
# L = jax.lax.linalg.cholesky(S)
# jnp.allclose(S, L @ L.T)

# # decompose M = S^{-1} = UU^T by doing a triangular solve of L^{T}
# U = jax.lax.linalg.triangular_solve(L.T, jnp.identity(2))
# M = jnp.linalg.inv(S)
# jnp.allclose(M, U @ U.T)

# # check that p=Uy ~ N(0,M) when y ~ N(0,I)
# ps = jax.vmap(
#     lambda rng_key: U @ random.normal(rng_key, (2,)),
#     out_axes=1
# )(random.split(sim_key, 10000))
# jnp.allclose(M, jnp.cov(ps), rtol=0.05)
###############################################################################

PreconditionerState = namedtuple(
    "PreconditionerState",
    [
        "var",
        "var_tril_factor",
        "inv_var_triu_factor"
    ]
)
"""
A :func:`~collections.namedtuple` defining the values associated with a 
preconditioner. It consists of the fields:

 - **var** - Estimated and regularized target variance.
 - **var_tril_factor** - A lower-triangular (tril) matrix `L` such that 
     `var == L @ L.T`.
 - **inv_var_triu_factor** - An upper-triangular (triu) matrix `U` such that 
     `inv(var) == U @ U.T`.
"""

###############################################################################
# type definitions
###############################################################################

class Preconditioner(ABC):

    @staticmethod
    @abstractmethod
    def maybe_alter_precond_state(precond_state, rng_key):
        """
        Build a (possible randomized) preconditioner.

        :param precond_state: An array representing the square-root of a covariance matrix.
            It can be a matrix---for dense preconditioners---or a vector---for
            the diagonal case.
        :param rng_key: A PRNG key that could be used for random preconditioning.
        :return: A vector representing a diagonal preconditioner.
        """
        raise NotImplementedError

#######################################
# Diagonal
#######################################

class IdentityDiagonalPreconditioner(Preconditioner):

    @staticmethod
    def maybe_alter_precond_state(precond_state, rng_key):
        I = jnp.ones_like(precond_state.var)
        return PreconditionerState(I, I, I)

class FixedDiagonalPreconditioner(Preconditioner):

    @staticmethod
    def maybe_alter_precond_state(precond_state, rng_key):
        return precond_state

#######################################
# Dense
#######################################

class FixedDensePreconditioner(Preconditioner):

    @staticmethod
    def maybe_alter_precond_state(precond_state, rng_key):
        return precond_state

def is_dense(preconditioner):
    return (
        isinstance(preconditioner, FixedDensePreconditioner)
    )

###############################################################################
# methods
###############################################################################

# initialization
def init_base_precond_state(sample_field_flat_shape, preconditioner):
    if is_dense(preconditioner):
        d = sample_field_flat_shape[0]
        return PreconditionerState(
            jnp.identity(d), jnp.identity(d), jnp.identity(d)
        )
    else: 
        return PreconditionerState(
            jnp.ones(sample_field_flat_shape),
            jnp.ones(sample_field_flat_shape),
            jnp.ones(sample_field_flat_shape)
        )


# adapt the base preconditioner state, regularizing via the old variance to 
# avoid issues with ill-conditioned sample variances. note: the specific weights
# apparently are used in Stan, according to NumPyro
# https://github.com/pyro-ppl/numpyro/blob/ab1f0dc6e954ef7d54724386667e33010b2cfc8b/numpyro/infer/hmc_util.py#L219
def adapt_base_precond_state(base_precond_state, sample_var, n):
    # define a nugget
    eps = 10 * lax.sqrt(jnp.finfo(sample_var.dtype).eps) * (
        jnp.identity(sample_var.shape[-1]) if jnp.ndim(sample_var)==2 else 1
    )

    # new variance as weighted average of old+nugget, and the estimated one
    var = (5*(eps+base_precond_state.var) + n*sample_var) / (n + 5)

    # compute the remaining terms, depending on shape
    if jnp.ndim(sample_var) == 2:
        var_tril_factor = lax.linalg.cholesky(var)
        inv_var_triu_factor = lax.linalg.triangular_solve(
            var_tril_factor, 
            jnp.identity(var.shape[-1]), 
            transpose_a=True, 
            lower=True
        )
    else:
        var_tril_factor = lax.sqrt(var)
        inv_var_triu_factor = jnp.reciprocal(var_tril_factor)

    return PreconditionerState(var, var_tril_factor, inv_var_triu_factor)
