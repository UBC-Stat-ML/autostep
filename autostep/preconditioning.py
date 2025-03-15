from abc import ABC, abstractmethod

from jax import numpy as jnp
from jax import random

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
        return jnp.ones_like(sqrt_var, (sqrt_var.shape[0],))

class FixedPreconditioner(Preconditioner):

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

