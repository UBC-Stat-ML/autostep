from abc import ABCMeta, abstractmethod

from jax import lax
from jax import random
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
        rng_key, selector_key = random.split(state.rng_key)
        state = state._replace(rng_key = rng_key) # always update the state with the modified key
        selector_params = self.selector.draw_parameters(selector_key)

        # forward step size search
        proposed_state = self.auto_step_size(state, selector_params)

        # backward step size search
        bwd_state = self.reset_exponent(self.involution_aux(proposed_state)) # bwd state: apply 2nd half of involution, then reset exponent
        bwd_state = self.auto_step_size(bwd_state, selector_params)
        reversibility_passed = proposed_state.exponent == bwd_state.exponent
        avg_fwd_bwd_step_size = (self.step_size(proposed_state) + self.step_size(bwd_state)) / 2

        # Metropolis-Hastings step
        log_joint_diff = proposed_state.log_joint - state.log_joint
        acc_prob = reversibility_passed * lax.clamp(0., lax.exp(log_joint_diff), 1.)
        rng_key, accept_key = random.split(bwd_state.rng_key) # note: bwd_state is the one with the "freshest" rng_key

        # build the next state depending on the MH outcome
        next_state = lax.cond(
            random.bernoulli(accept_key, acc_prob),
            utils.next_state_accepted,
            utils.next_state_rejected,
            (state, proposed_state, bwd_state, rng_key)
        )

        # collect statistics and return
        new_stats = statistics.record_post_sample_stats(
            next_state.stats, avg_fwd_bwd_step_size, acc_prob
        )
        return next_state._replace(stats = new_stats)

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
        state, *extra = lax.while_loop(
            self.shrink_step_size_cond_fun,
            self.shrink_step_size_body_fun,
            (self, state, selector_params, init_log_joint)
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
