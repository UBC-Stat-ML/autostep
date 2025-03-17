from abc import ABC, abstractmethod

import jax
from jax import lax
from jax import numpy as jnp
from jax import random

from numpyro import util

class Preconditioner(ABC):

    @staticmethod
    @abstractmethod
    def build_precond(sqrt_var, rng_key):
        """
        Build a (possible randomized) preconditioner.

        :param sqrt_var: An array representing the square-root of a covariance matrix.
            It can be a matrix---for dense preconditioners---or a vector---for
            the diagonal case.
        :param rng_key: A PRNG key that could be used for random preconditioning.
        :return: A vector representing a diagonal preconditioner.
        """
        raise NotImplementedError
    
class IdentityDiagonalPreconditioner(Preconditioner):

    @staticmethod
    def build_precond(sqrt_var, rng_key):
        return jnp.ones_like(sqrt_var)

class FixedDiagonalPreconditioner(Preconditioner):

    @staticmethod
    def build_precond(sqrt_var, rng_key):
        return sqrt_var

class MixDiagonalPreconditioner(Preconditioner):

    @staticmethod
    def build_precond(sqrt_var, rng_key):
        assert len(jnp.shape(sqrt_var)) == 1

        # uniform mixture in log space
        # p = exp(U*log(hat_sd) + (1-U)log(1)) = exp(log(hat_sd^U))) = hat_sd^U
        return sqrt_var ** random.uniform(rng_key)

#######################################
# Dense
#######################################

class FixedDensePreconditioner(Preconditioner):

    @staticmethod
    def build_precond(sqrt_var, rng_key):
        return sqrt_var

class MixDensePreconditioner(Preconditioner):

    @staticmethod
    def build_precond(sqrt_var, rng_key):
        assert len(jnp.shape(sqrt_var)) == 2

        # with equal prob choose the estimate or the identity
        return lax.cond(
            random.bernoulli(rng_key),
            util.identity,
            lambda M: jnp.eye(*M.shape),
            sqrt_var
        )

def is_dense(preconditioner):
    return (
        isinstance(preconditioner, FixedDensePreconditioner) or
        isinstance(preconditioner, MixDensePreconditioner)
    )
