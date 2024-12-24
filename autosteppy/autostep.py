from abc import ABCMeta, abstractmethod

from jax import random
from numpyro import infer

class AutoStep(infer.mcmc.MCMCKernel, metaclass=ABCMeta):

    @property
    def model(self):
        return self._model

    @abstractmethod
    def draw_aux_vars(self, rng_key, state):
        """
        Draw auxiliary variables required by the underlying involutive MCMC method.

        :param rng_key: Random number generator key.
        :param state: Current augmented (i.e., including previous aux vars) state.
        :return: New auxiliary values.
        """
        raise NotImplementedError
    
    @abstractmethod
    def involution(self, state):
        """
        Apply the deterministic involutive transformation of the underlying kernel.

        :param state: Current augmented (i.e., including previous aux vars) state.
        :return: New state.
        """
        raise NotImplementedError

    def sample(self, state, model_args, model_kwargs):
        """
        Perform one iteration of autoStep.

        :param rng_key: Random number generator key.
        :param state: Current augmented (i.e., including aux vars) state.
        :return: Updated state.
        """
        # draw auxiliary variables (e.g., momentum)
        new_aux_vars = self.draw_aux_vars(state)

        # draw selector parameters
        rng_key, selector_rng_key = random.split(rng_key)
        selector_params = self.selector.draw_parameters(selector_rng_key)

        # search for step size
        proposed_exponent = self.auto_step_size()
        proposed_step_size = self.step_size * (2 ** proposed_exponent)



