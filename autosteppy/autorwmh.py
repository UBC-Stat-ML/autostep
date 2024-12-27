from collections import namedtuple

import jax

from jax import flatten_util
from jax import numpy as jnp
from jax import random
from numpyro import infer
from numpyro import util

from autosteppy import autostep
from autosteppy import statistics
from autosteppy import utils


AutoRWMHState = namedtuple(
    "AutoRWMHState",
    [
        "x",
        "v_flat",
        "log_joint",
        "exponent",
        "rng_key",
        "stats"
    ],
)

class AutoRWMH(autostep.AutoStep):

    def __init__(
        self,
        model=None,
        potential_fn=None,
        base_step_size=1.0,
    ):
        self._model = model
        self._potential_fn = potential_fn
        self._base_step_size = base_step_size
        self._postprocess_fn = None
        
    @property
    def sample_field(self):
        return "x"

    def update_log_joint(self, state):
        x, v_flat, _ = state
        new_log_joint = -self._potential_fn(x) - utils.std_normal_potential(v_flat)
        new_stats = statistics.increase_n_pot_evals_by_one(state.stats)
        return state._replace(log_joint = new_log_joint, stats = new_stats)

    def init(self, rng_key, num_warmup, init_params, model_args, model_kwargs):
        rng_key, rng_key_init = random.split(rng_key)
        
        # initialize the state and the model
        # store potential fn and postprocess fn
        init_params, self._potential_fn, self._postprocess_fn = utils.init_state_and_model(
            self._model, rng_key_init, model_args, model_kwargs, init_params
        )
        x_flat_shape = jnp.shape(flatten_util.ravel_pytree(init_params)[0])
        init_state = AutoRWMHState(
            init_params,
            jnp.zeros(x_flat_shape),
            0., # Note: not the actual log joint value; needs to be updated 
            self.base_step_size,
            rng_key,
            statistics.AutoStepStats()
        )
        return jax.device_put(init_state)
    
    def refresh_aux_vars(self, state):
        rng_key, v_key = random.split(state.rng_key)
        v_flat = random.normal(v_key, jnp.shape(state.v_flat))
        return state._replace(v_flat = v_flat, rng_key = rng_key)
    
    def involution_main(self, state):
        step_size = self.step_size(state)
        x_flat, unravel_fn = flatten_util.ravel_pytree(state.x)
        x_new = unravel_fn(x_flat + step_size * state.v_flat)
        return state._replace(x = x_new)
    
    def involution_aux(self, state):
        return state._replace(v_flat = -state.v_flat)
    
