from abc import ABC, abstractmethod

from jax import numpy as jnp
from jax import random

class DiagonalPreconditioner(ABC):

    @staticmethod
    @abstractmethod
    def build_diag_precond(estimated_std_devs, rng_key):
        """
        Build a (possible randomized) diagonal preconditioner.

        :param diag_mat: The diagonal of a (diagonal) preconditioning matrix.   
        :param rng_key: A PRNG key that could be used for random preconditioning.
        :return: A vector representing a diagonal preconditioner.
        """
        raise NotImplementedError
    
class IdentityPreconditioner(DiagonalPreconditioner):

    @staticmethod
    def build_diag_precond(estimated_std_devs, rng_key):
        return jnp.ones_like(estimated_std_devs)

class FixedDiagonalPreconditioner(DiagonalPreconditioner):

    @staticmethod
    def build_diag_precond(estimated_std_devs, rng_key):
        return estimated_std_devs

class MixDiagonalPreconditioner(DiagonalPreconditioner):

    @staticmethod
    def build_diag_precond(estimated_std_devs, rng_key):
        # uniform mixture in log space
        # p = exp(U*log(hat_sd) + (1-U)log(1)) = exp(log(hat_sd^U))) = hat_sd^U
        return estimated_std_devs ** random.uniform(rng_key)

