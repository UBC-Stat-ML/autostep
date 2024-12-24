from collections import namedtuple

import autosteppy
import jax

from autosteppy import utils
from jax import flatten_util
from jax import numpy as jnp
from jax import random
from numpyro import infer
from numpyro import util

import autosteppy.autostep

AutoRWMHState = namedtuple(
    "AutoRWMHState",
    [
        "x",
        "p_flat",
        "step_size",
        "rng_key",
    ],
)

class AutoRWMH(autosteppy.autostep.AutoStep):

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
        
    def init(self, rng_key, num_warmup, init_params, model_args, model_kwargs):
        rng_key, rng_key_init = random.split(rng_key)
        
        # initialize the state and the model
        init_params, self._potential_fn, self._postprocess_fn = utils.init_state_and_model(
            self._model, rng_key_init, model_args, model_kwargs, init_params
        )
        potential_val = self._potential_fn(init_params)
        size = len(flatten_util.ravel_pytree(init_params)[0])
        init_state = AutoRWMHState(
            init_params,
            jnp.zeros((size,)),
            rng_key,
        )
        return jax.device_put(init_state)
    
    def postprocess_fn(self, model_args, model_kwargs):
        if self._postprocess_fn is None:
            return util.identity
        return self._postprocess_fn(*model_args, **model_kwargs)
    
    def draw_aux_vars(self, state):
        rng_key, momentum_key = random.split(state.rng_key)
        p_flat = random.normal(momentum_key, jnp.shape(state.p_flat))
        return AutoRWMHState(state.x, p_flat, state.step_size, rng_key)

    def involution(self, state):
        x, p_flat, step_size, rng_key = state
        x_flat, unravel_fn = flatten_util.ravel_pytree(x)
        x_new = unravel_fn(x_flat + step_size * p_flat)
        p_new_flat = -p_flat 
        return AutoRWMHState(x_new, p_new_flat, step_size, rng_key)
    
