import equinox as eqx
import jax.numpy as jnp

from .deepcomponent_eqx import DeepComponent
from .dmm_autoencoder_eqx import DeepMechanisticModel
from jax import tree_map
from typing import Union


def get_finite_grads(grads):
    return tree_map(
            lambda x: jnp.where(jnp.isfinite(x), x, jnp.zeros_like(x)),
            grads,
        )


def get_parameters(
        module: Union[DeepComponent, eqx.Module]
) -> jnp.ndarray:
    params = []
    for layer in module.layers:
        params.append(
            layer.weight.flatten()
        )

        # Check if the layer has a 'bias' attribute and append if it does
        if hasattr(layer, 'bias') and layer.bias is not None:
            params.append(
                layer.bias.flatten()
            )

    # Concatenate into single output array
    module_params = jnp.concatenate(params)
    return module_params


def map_params_to_array(
        model: DeepMechanisticModel
) -> jnp.ndarray:
    encoder_params = get_parameters(model.deep_encoder)
    inflater_params = get_parameters(model.deep_inflater)
    # TODO @GiacomoFabrini reinstate if reinstating .learned_median_params
    # kincombiner_params = model.kin_params_combiner.learned_median_params
    param_array = jnp.concatenate([
        module_params.flatten()
        for module_params in [
            encoder_params,
            inflater_params,
            # kincombiner_params
        ]
    ])
    if model.reconstruct:
        decoder_params = get_parameters(model.deep_decoder)
        param_array = jnp.concatenate([param_array.flatten(), decoder_params.flatten()])
    param_array = jnp.concatenate([param_array, model.kin_params_combiner.learned_global_kin_params.flatten()])
    if len(param_array) != len(model.x_names):
        raise ValueError("Number of parameters does not match number of parameter names!")
    return param_array
