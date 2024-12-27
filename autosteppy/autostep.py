from abc import ABCMeta, abstractmethod

from jax import lax
from numpyro import infer
from numpyro import util

from autosteppy import utils
from autosteppy import stats

class AutoStep(infer.mcmc.MCMCKernel, metaclass=ABCMeta):

    @property
    def model(self):
        return self._model
    
    def postprocess_fn(self, model_args, model_kwargs):
        if self._postprocess_fn is None:
            return util.identity
        return self._postprocess_fn(*model_args, **model_kwargs)
    
    @abstractmethod
    def update_joint_potential(self, state):
        """
        Compute the total potential for all variables, i.e., including auxiliary.
        It should also update the gradient of this joint potential if it is part of
        the state.

        :param state: Current state.
        :return: Updated state.
        """
        raise NotImplementedError

    @abstractmethod
    def refresh_aux_vars(self, rng_key, state):
        """
        Refresh auxiliary variables required by the underlying involutive MCMC method.

        :param rng_key: Random number generator key.
        :param state: Current state.
        :return: State with updated auxiliary variables.
        """
        raise NotImplementedError
    
    def reset_exponent(self, state):
        return state._replace(exponent = 0)
    
    def step_size(self, state):
        """
        Compute the step size corresponding to the current exponent.

        :param state: Current state.
        :return: Step size.
        """
        return self._base_step_size * (2. ** state.exponent)

    @abstractmethod
    def reset_step_size(self, state):
        """
        Return state.step_size to self.base_step_size.

        :param state: Current state.
        :return: State with step_size == self.base_step_size.
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
        # reset exponent, refresh auxiliary variables (e.g., momentum), update
        # the joint potential, and finally check if the latter is finite
        state = self.update_joint_potential(self.refresh_aux_vars(self.reset_exponent(state)))
        utils.checkified_is_finite(state.joint_pot)

        # draw selector parameters
        selector_params = self.selector.draw_parameters(state)

        # forward step size search
        fwd_state = self.auto_step_size(selector_params, state)
        
        # collect statistics
        state._replace(stats = stats.record_post_sample_stats(
            state.stats, avg_fwd_bwd_step_size)
        )

    def auto_step_size(self, selector_params, state):
        fwd_state = self.update_joint_potential(self.involution_main(state))
        init_jp_diff = fwd_state.joint_pot - state.joint_pot
        must_shrink = self.selector.should_shrink(selector_params, init_jp_diff)
        lax.cond(must_shrink, true_fun, util.identity, fwd_state)
        must_grow = self.selector.should_grow(selector_params, init_jp_diff)

        pass


