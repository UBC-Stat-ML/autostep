from abc import ABCMeta
from functools import partial

from jax import flatten_util
from jax import numpy as jnp
from jax import lax
from jax import random

from autostep import automatic_mcmc

class SliceSampler(automatic_mcmc.AutomaticMCMC, metaclass=ABCMeta):
    """
    Interface for defining slice sampling algorithms that leverage the 
    univariate algorithm of Neal (2003). The `p_flat` component of the
    state dictates the direction along which the univariate sampler is
    applied. For example, by using one-hot vectors, we can implement
    either the deterministic or random scan versions. Similarly, by
    drawing arbitrary directions at random, we can implement a 
    Hit-and-Run slicer.
    """
    def __init__(
        self,
        *args,
        max_grow_steps = 20,
        max_shrink_steps = 40,
        init_window_size = 4.0, # min-cost-optimal for N(0,1)
        **kwargs
    ):
        """
        Initialize the slice sampler.

        :param args: Passed to the :class:`AutomaticMCMC` constructor.
        :param max_grow_steps: Maximum number of iterations of the stepping-out phase.
        :param max_shrink_steps: Maximum number of iterations of the shrink phase.
        :param init_window_size: Size of initial window. Internally we identify
            this parameter with the step size in the :class:`AutomaticMCMC` 
            interface.
        :param kwargs: Passed to the :class:`AutomaticMCMC` constructor.
        """
        super().__init__(*args, init_base_step_size=init_window_size, **kwargs)
        self.max_grow_steps = max_grow_steps
    
    @staticmethod
    def in_slice(state, target_log_joint):
        return state.log_joint > target_log_joint

    def init_endpoints(self, rng_key, state, precond_state):
        # randomly place a window of fixed width around the initial `x` in the
        # direction of `p_flat`
        width = state.base_step_size
        p_flat = state.p_flat
        x_flat, unravel_fn = flatten_util.ravel_pytree(state.x)
        u = random.uniform(rng_key)
        x_flat_bwd = x_flat - p_flat*width*u
        x_bwd = unravel_fn(x_flat_bwd)
        x_fwd = unravel_fn(x_flat_bwd + p_flat*width)
        
        # make new states, update their log joints, and return
        bwd_state = self.update_log_joint(
            state._replace(x=x_bwd), precond_state
        )
        fwd_state = self.update_log_joint(
            state._replace(x=x_fwd), precond_state
        )
        return bwd_state, fwd_state

    def grow_slice(self, rng_key, bwd_state, fwd_state, target_log_joint, precond_state):
        width = bwd_state.base_step_size
        max_grow_steps = self.max_grow_steps

        # randomly allocate the budget
        max_grow_steps_bwd = random.randint(rng_key, (), 0, max_grow_steps)
        max_grow_steps_fwd = max_grow_steps - max_grow_steps_bwd
        
        # loop condition
        def cond_fn(state, budget):
            return jnp.logical_and(
                budget>0, self.in_slice(state, target_log_joint)
            )
        
        # grow function
        def body_fn(direction, carry):
            state, budget = carry
            x_flat, unravel_fn = flatten_util.ravel_pytree(state.x)
            x_new = unravel_fn(x_flat + state.p_flat*width*direction)
            state_new = self.update_log_joint(
                state._replace(x=x_new), precond_state
            )
            return (state_new, budget-1)
        
        # grow bwd
        bwd_state, _ = lax.while_loop(
            cond_fn, 
            partial(body_fn, -1),
            (bwd_state, max_grow_steps_bwd)
        )

        # grow fwd
        fwd_state, _ = lax.while_loop(
            cond_fn, 
            partial(body_fn, 1),
            (fwd_state, max_grow_steps_fwd)
        )

        return bwd_state, fwd_state

    def shrink_slice(self, rng_key, bwd_state, fwd_state, target_log_joint, precond_state):
        def draw_new_position(draw_key, xf_bwd, xf_fwd, new_state):
            u = random.uniform(draw_key)
            x_new = unravel_fn(xf_bwd + u*(xf_fwd-xf_bwd))
            return self.update_log_joint(
                new_state._replace(x = x_new), precond_state
            )
        
        # initial position
        rng_key, init_key = random.split(rng_key)
        x_flat_bwd, unravel_fn = flatten_util.ravel_pytree(bwd_state.x)
        x_flat_fwd = flatten_util.ravel_pytree(fwd_state.x)[0]
        new_state = draw_new_position(
            init_key, x_flat_bwd, x_flat_fwd, bwd_state # use bwd_state as template
        )

        # shrinking loop
        def cond_fn(carry):
            *_, new_state, n_iter = carry
            return jnp.logical_and(
                n_iter<self.max_shrink_steps,
                jnp.logical_not(self.in_slice(new_state, target_log_joint))
            )
        
        def body_fn(carry):
            rng_key, x_flat_bwd, x_flat_fwd, new_state, n_iter = carry
            rng_key, draw_key = random.split(rng_key)
            new_state = draw_new_position(
                draw_key, x_flat_bwd, x_flat_fwd, new_state
            )
            


    def sample(self, state, model_args, model_kwargs):
        # generate rng keys and store the updated master key in the state
        (
            rng_key, 
            precond_key, 
            init_key, 
            height_key,
            grow_key
        ) = random.split(state.rng_key, 5)
        state = state._replace(rng_key = rng_key)

        # build a (possibly randomized) preconditioner
        precond_state = self.preconditioner.maybe_alter_precond_state(
            state.base_precond_state, precond_key
        )

        # refresh the direction
        state = self.update_log_joint(
            self.refresh_aux_vars(state, precond_state), precond_state
        )

        # draw a target log height
        # note: y = pi(z)U(0,1) <=> log(y) = logpi(z) - Exp(1)
        target_log_joint = state.log_joint - random.exponential(height_key)

        # init slice endpoints
        bwd_state, fwd_state = self.init_endpoints(init_key, state)

        # grow slice
        


        return

    
