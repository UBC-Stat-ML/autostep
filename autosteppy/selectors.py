from abc import ABC, abstractmethod
from jax import random
from jax import lax
import jax.numpy as jnp

class StepSizeSelector(ABC):

    @abstractmethod
    def draw_parameters(self, rng_key):
        """
        Draw the random parameters used (if any) by the selector.

        :param rng_key: Random number generator key.
        :return: Random instance of the parameters used by the selector.
        """
        raise NotImplementedError

    @abstractmethod
    def should_grow(self, parameters, log_diff):
        """
        Decide if step size should grow based on `parameters` and current log joint
        difference `log_diff`.

        :param parameters: selector parameters.
        :return: `True` if step size should grow; `False` otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def should_shrink(self, parameters, log_diff):
        """
        Decide if step size should shrink based on `parameters` and current log joint
        difference `log_diff`.

        :param parameters: selector parameters.
        :return: `True` if step size should shrink; `False` otherwise.
        """
        raise NotImplementedError


def _draw_log_unif_bounds(rng_key):
    return lax.sort(random.exponential(rng_key, (2,)) * (-1))

class AsymmetricSelector(StepSizeSelector):
    """
    Asymmetric selector. 
    """

    def draw_parameters(self, rng_key):
        return _draw_log_unif_bounds(rng_key)

    def should_grow(self, bounds, log_diff):
        return log_diff > bounds[1]

    def should_shrink(self, bounds, log_diff):
        return jnp.logical_or(~lax.is_finite(log_diff), log_diff < bounds[0])


class SymmetricSelector(StepSizeSelector):
    """
    Symmetric selector. 
    """

    def draw_parameters(self, rng_key):
        return _draw_log_unif_bounds(rng_key)

    def should_grow(self, bounds, log_diff):
        return lax.abs(log_diff) + bounds[1] < 0

    def should_shrink(self, bounds, log_diff):
        invalid_log_diff = ~lax.is_finite(log_diff)
        return jnp.logical_or(invalid_log_diff, lax.abs(log_diff) + bounds[0] > 0)


class FixedStepSizeSelector(StepSizeSelector):
    """
    A dummy selector that never adjusts the step size. 
    """

    def draw_parameters(self, rng_key):
        return None

    def should_grow(self, bounds, log_diff):
        return False

    def should_shrink(self, bounds, log_diff):
        return False

