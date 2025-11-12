from abc import ABC, abstractmethod
from jax import random
from jax import lax
import jax.numpy as jnp

def _draw_log_unif_bounds(rng_key):
    return lax.sort(random.exponential(rng_key, (2,)) * (-1))

class StepSizeSelector(ABC):
    """
    Abstract class for defining step size selectors.

    :param max_n_iter: Maximum number of step size doubling/halvings.
    :param bounds_sampler: A function that takes a PRNG key and samples a pair
        of endpoints used in the step-size selection loop. Defaults to ordered
        log-uniform random variables.
    """

    def __init__(
            self, 
            max_n_iter=7, # 2**7 => step size changes +/- 2 orders of mag
            bounds_sampler=_draw_log_unif_bounds
        ):
        self.max_n_iter = max_n_iter
        self.bounds_sampler = bounds_sampler

    def draw_parameters(self, rng_key):
        """
        Draw the random parameters used (if any) by the selector.

        :param rng_key: Random number generator key.
        :return: Random instance of the parameters used by the selector.
        """
        return self.bounds_sampler(rng_key)

    @staticmethod
    @abstractmethod
    def should_grow(parameters, log_diff):
        """
        Decide if step size should grow based on `parameters` and current log joint
        difference `log_diff`.

        :param parameters: selector parameters.
        :return: `True` if step size should grow; `False` otherwise.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def should_shrink(parameters, log_diff):
        """
        Decide if step size should shrink based on `parameters` and current log joint
        difference `log_diff`.

        :param parameters: selector parameters.
        :return: `True` if step size should shrink; `False` otherwise.
        """
        raise NotImplementedError
    
    @staticmethod
    def adapt_base_step_size(base_step_size, mean_step_size, n_samples_in_round):
        return mean_step_size
    

class AsymmetricSelector(StepSizeSelector):
    """
    Asymmetric selector.
    """

    @staticmethod
    def should_grow(bounds, log_diff):
        return log_diff > bounds[1]

    @staticmethod
    def should_shrink(bounds, log_diff):
        return jnp.logical_or(
            jnp.logical_not(lax.is_finite(log_diff)), 
            log_diff < bounds[0]
        )

def make_deterministic_bounds_sampler(p_lo, p_hi):
    assert p_lo < p_hi and 0 < p_lo and p_hi <= 1
    fixed_bounds = jnp.log(jnp.array([p_lo, p_hi]))
    return (lambda _: fixed_bounds)

def DeterministicAsymmetricSelector(p_lo=0.01, p_hi=0.99, *args, **kwargs):
    """
    Asymmetric selector with fixed deterministic endpoints.

    :param p_lo: Left endpoint in [0,1].
    :param p_hi: Right endpoint in [0,1].
    :param *args: Additional arguments for `AsymmetricSelector`.
    :param **kwargs: Additional keyword arguments for `AsymmetricSelector`.
    """
    return AsymmetricSelector(
        *args,
        bounds_sampler = make_deterministic_bounds_sampler(p_lo, p_hi),
        **kwargs
    )

class SymmetricSelector(StepSizeSelector):
    """
    Symmetric selector.
    """

    @staticmethod
    def should_grow(bounds, log_diff):
        return lax.abs(log_diff) + bounds[1] < 0

    @staticmethod
    def should_shrink(bounds, log_diff):
        return jnp.logical_or(
            jnp.logical_not(lax.is_finite(log_diff)),
            lax.abs(log_diff) + bounds[0] > 0
        )

def DeterministicSymmetricSelector(p_lo=0.01, p_hi=0.99, *args, **kwargs):
    """
    Symmetric selector with fixed deterministic endpoints.

    :param p_lo: Left endpoint in [0,1].
    :param p_hi: Right endpoint in [0,1].
    :param *args: Additional arguments for `SymmetricSelector`.
    :param **kwargs: Additional keyword arguments for `SymmetricSelector`.
    """
    return SymmetricSelector(
        *args,
        bounds_sampler = make_deterministic_bounds_sampler(p_lo, p_hi),
        **kwargs
    )

class FixedStepSizeSelector(StepSizeSelector):
    """
    A dummy selector that never adjusts the step size. 
    """
    def __init__(self):
        super().__init__(
            max_n_iter = 0, 
            bounds_sampler = make_deterministic_bounds_sampler(0.4, 0.6) # bounds are irrelevant
        )

    @staticmethod
    def should_grow(bounds, log_diff):
        return False

    @staticmethod
    def should_shrink(bounds, log_diff):
        return False
    
    @staticmethod
    def adapt_base_step_size(base_step_size, mean_step_size, n_samples_in_round):
        return base_step_size

