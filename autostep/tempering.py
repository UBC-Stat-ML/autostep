from functools import partial

import jax
from jax import lax
from jax import numpy as jnp

from numpyro.handlers import substitute, trace
from numpyro.infer import util
from numpyro.distributions.util import is_identically_one, is_identically_zero


##############################################################################
# log density evaluation utilities
##############################################################################

# slight modification of numpyro.infer.util.compute_log_probs
def log_prob_site(site):
    value = site["value"]
    intermediates = site["intermediates"]
    scale = site["scale"]
    if intermediates:
        log_prob = site["fn"].log_prob(value, intermediates)
    else:
        guide_shape = jnp.shape(value)
        model_shape = tuple(site["fn"].shape()) # TensorShape from tfp needs casting to tuple
        try:
            lax.broadcast_shapes(guide_shape, model_shape)
        except ValueError:
            raise ValueError(
                "Model and guide shapes disagree at site: '{}': {} vs {}".format(
                    site["name"], model_shape, guide_shape
                )
            )
        log_prob = site["fn"].log_prob(value)

    log_prob_sum = jnp.sum(log_prob)

    if (scale is not None) and (not is_identically_one(scale)):
        log_prob_sum = scale * log_prob_sum
    
    return log_prob_sum

def is_observation(site):
    return site["is_observed"] and (not site["name"].endswith("_log_det"))

def log_prior(model_trace):
    return sum(
        log_prob_site(site) 
        for site in model_trace.values()
        if site["type"] == "sample" and (not is_observation(site))
    )

def log_lik(model_trace):
    return sum(
        log_prob_site(site) 
        for site in model_trace.values()
        if site["type"] == "sample" and is_observation(site)
    )

# full trace from unconstrained sample
def trace_from_unconst_samples(
        model, 
        model_args, 
        model_kwargs,
        unconstrained_sample, 
    ):
    """
    Generate a full model trace from a sample in unconstrained space.

    :param model: A numpyro model.
    :param model_args: Model arguments.
    :param model_kwargs: Model keyword arguments.
    :param unconstrained_sample: A sample from the model in unconstrained space.
    :return: A trace.
    """
    substituted_model = substitute(
        model, substitute_fn=partial(
            util._unconstrain_reparam, 
            unconstrained_sample
        )
    )
    return trace(substituted_model).get_trace(*model_args, **model_kwargs)

# tempered potential of a model
@partial(jax.jit, static_argnums=(0,))
def tempered_potential(model, model_args, model_kwargs, unconstrained_sample, inv_temp = None):
    """
    Build a tempered version of the potential function associated with the
    posterior distribution of a numpyro model. Specifically,

    .. code-block:: python
        tempered_potential(x) = -log(pi_beta(x))

    where `beta` is the inverse temperature, and

    .. code-block:: python
        pi_beta(x) = prior(x) * likelihood(x) ** beta

    Hence, when `inv_temp=0`, the tempered model reduces to the prior, whereas
    for `inv_temp=1`, the original posterior distribution is recovered.

    To achieve this, we first constrain the sample and then use it to update the 
    model trace to match the sampled values. Then, we iterate the trace and
    compute the logprior -- including any logabsdetjac terms due to change of
    variables -- and loglikelihood. Finally, we return their tempered sum.

    :param model: A numpyro model.
    :param model_args: Model arguments.
    :param model_kwargs: Model keyword arguments.
    :param unconstrained_sample: A sample from the model in unconstrained space.
    :param inv_temp: An inverse temperature (non-negative number).
    :return: The potential evaluated at the given sample.
    """
    model_trace = trace_from_unconst_samples(
        model, 
        model_args, 
        model_kwargs,
        unconstrained_sample
    )
    if inv_temp is None:
        # default to full log joint
        return -(log_prior(model_trace) + log_lik(model_trace))
    else:
        return -(log_prior(model_trace) + inv_temp*log_lik(model_trace))
   