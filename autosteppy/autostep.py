from abc import ABCMeta, abstractmethod

from jax import lax
from numpyro import infer
from numpyro import util

from autosteppy import utils
from autosteppy import statistics

class AutoStep(infer.mcmc.MCMCKernel, metaclass=ABCMeta):

    @property
    def model(self):
        return self._model
    
    def postprocess_fn(self, model_args, model_kwargs):
        if self._postprocess_fn is None:
            return util.identity
        return self._postprocess_fn(*model_args, **model_kwargs)
    
    @abstractmethod
    def update_log_joint(self, state):
        """
        Compute the log joint density for all variables, i.e., including auxiliary.
        This should also update the gradient of the log joint density whenever the
        these are part of the state of the underlying involutive sampler.

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
        # the log joint density, and finally check if the latter is finite
        state = self.update_log_joint(self.refresh_aux_vars(self.reset_exponent(state)))
        utils.checkified_is_finite(state.log_joint)

        # draw selector parameters
        selector_params = self.selector.draw_parameters(state)

        # forward step size search
        fwd_state = self.auto_step_size(state, selector_params)

        # backward step size search
        bwd_state = self.reset_exponent(self.involution_aux(fwd_state)) # init bwd state from fwd but reset exponent
        bwd_state = self.auto_step_size(bwd_state, selector_params)
        reversibility_passed = fwd_state.exponent == bwd_state.exponent
        
        # TODO: Metropolis accept-reject

        # collect statistics
        new_stats = statistics.record_post_sample_stats(
            state.stats, avg_fwd_bwd_step_size
        )
        return state._replace(stats = new_stats)

    def auto_step_size(self, state, selector_params):
        init_log_joint = state.log_joint
        next_state = self.update_log_joint(self.involution_main(state))

        # try shrinking (no-op if selector decides not to shrink)
        next_state = self.shrink_step_size(next_state, selector_params, init_log_joint)

        # try growing (no-op if selector decides not to grow)
        next_state = self.grow_step_size(next_state, selector_params, init_log_joint)

        return next_state


    def shrink_step_size_cond_fun(self, state, selector_params, init_log_joint):
        log_diff = state.log_joint - init_log_joint
        return self.selector.should_shrink(selector_params, log_diff)

    def shrink_step_size_body_fun(self, state, selector_params, init_log_joint):
        state = state._replace(exponent = state.exponent - 1)
        return self.update_log_joint(self.involution_main(state))

    def shrink_step_size(self, state, selector_params, init_log_joint):
        state, _ = lax.while_loop(
            self.shrink_step_size_cond_fun,
            self.shrink_step_size_body_fun,
            (state, selector_params, init_log_joint)
        )

        return state
    
    def grow_step_size_cond_fun(self, state, selector_params, init_log_joint):
        log_diff = state.log_joint - init_log_joint
        return self.selector.should_grow(selector_params, log_diff)

    def grow_step_size_body_fun(self, state, selector_params, init_log_joint):
        state = state._replace(exponent = state.exponent + 1)
        return self.update_log_joint(self.involution_main(state))

    def grow_step_size(self, state, selector_params, init_log_joint):
        state, _ = lax.while_loop(
            self.grow_step_size_cond_fun,
            self.grow_step_size_body_fun,
            (state, selector_params, init_log_joint)
        )

        # deduct 1 step to avoid cliffs, but only do this if we actually entered the loop
        state = lax.cond(
            state.exponent > 0,
            lambda s: s._replace(exponent = s.exponent - 1),
            lambda s: s,
            state
        )

        return state

