from abc import ABCMeta, abstractmethod

from jax import lax
from numpyro import infer
from numpyro import util

class AutoStep(infer.mcmc.MCMCKernel, metaclass=ABCMeta):

    @property
    def model(self):
        return self._model
    
    @abstractmethod
    def joint_potential(self, state):
        """
        Compute the total potential for all variables, i.e., including auxiliary.

        :param state: Current state.
        :return: Joint potential.
        """
        raise NotImplementedError

    @abstractmethod
    def refresh_aux_vars(self, rng_key, state):
        """
        Refresh auxiliary variables required by the underlying involutive MCMC method.

        :param rng_key: Random number generator key.
        :param state: Current augmented (i.e., including previous aux vars) state.
        :return: State with updated auxiliary variables.
        """
        raise NotImplementedError
    
    def involution(self, state):
        """
        Apply the (full) involution, which must satisfy for all `state`,
        `state == involution(involution(state))`.

        :param state: Current state.
        :return: Updated state.
        """
        return self.involution_aux(self.involution_main(state))

    @abstractmethod
    def involution_main(self, state):
        """
        Apply the main part of the involution. This is usually the part that 
        modifies the variables of interests.

        :param state: Current state.
        :return: Updated state.
        """
        raise NotImplementedError
    
    @abstractmethod
    def involution_aux(self, state):
        """
        Apply the auxiliary part of the involution. This is usually the part that
        is not necessary to implement for the respective involutive MCMC algorithm
        to work correctly (e.g., momentum flip in HMC).

        :param state: Current state.
        :return: Updated state.
        """
        raise NotImplementedError

    def sample(self, state, model_args, model_kwargs):
        # refresh auxiliary variables (e.g., momentum)
        state = self.refresh_aux_vars(state)

        # draw selector parameters
        selector_params = self.selector.draw_parameters(state)

        # search for step size
        proposed_exponent = self.auto_step_size(selector_params, state)
        proposed_step_size = self.step_size * (2 ** proposed_exponent)

    # def auto_step_size(self, selector_params, state):
    #     init_joint_potential = self.joint_potential(state)
    #     self.selector.should_grow(selector_params, log_diff)
    #     lax.cond()
    #     pass

    def postprocess_fn(self, model_args, model_kwargs):
        if self._postprocess_fn is None:
            return util.identity
        return self._postprocess_fn(*model_args, **model_kwargs)

