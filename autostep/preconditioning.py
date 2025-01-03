from abc import ABC, abstractmethod

from jax import lax
from jax import numpy as jnp
from jax import random

class DiagonalPreconditioner(ABC):

    def __init__(self, dim = None):
        if dim is not None:
            self.init(dim)
        else:
            self.estimated_std_devs = None

    def init(self, dim):
        self.estimated_std_devs = jnp.ones(dim)
        
    def adapt_preconditioner(self, estimated_std_devs):
        self.estimated_std_devs = estimated_std_devs

    def precondition(self, rng_key, x):
        return self.estimated_std_devs * x

class IdentityPreconditioner(DiagonalPreconditioner):

    def adapt_preconditioner(self, estimated_std_devs):
        return

class FixedDiagonalPreconditioner(DiagonalPreconditioner): pass    
    
class MixDiagonalPreconditioner(DiagonalPreconditioner):

    def __init__(self, dim = None):
        if dim is not None:
            self.init(dim)
        else:
            self.log_estimated_std_devs = None

    def init(self, dim):
        self.log_estimated_std_devs = jnp.zeros(dim)

    def adapt_preconditioner(self, estimated_std_devs):
        self.log_estimated_std_devs = lax.log(estimated_std_devs)

    def precondition(self, rng_key, x):
        # uniform mixture in log space
        # p = exp(U*log(hat_sd) + (1-U)log(1)) = exp(Ulog_hat_sd))
        return x * lax.exp(random.uniform(rng_key) * self.log_estimated_std_devs)



