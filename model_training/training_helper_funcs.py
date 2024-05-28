import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pypesto

from dmm.deepcomponent_eqx import DeepComponent
from dmm.dmm_autoencoder_eqx import DeepMechanisticModel
from jax import vmap
from jax.tree_util import tree_map
from jaxtyping import Array, PyTree
from pypesto.objective.jax import JaxObjective
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
    param_array = jnp.concatenate([
        module_params.flatten()
        for module_params in [
            encoder_params,
            inflater_params,
        ]
    ])
    if model.reconstruct:
        decoder_params = get_parameters(model.deep_decoder)
        param_array = jnp.concatenate(
            [
                param_array.flatten(),
                decoder_params.flatten()
            ]
        )
    param_array = jnp.concatenate(
        [
            param_array,
            model.kin_params_combiner.learned_global_kin_params.flatten()
        ]
    )
    return param_array


def zero_out_layer_params(
        param: Array,
        thresh: float,
):
    """
    Takes in input the parameters (weights/biases) of a layer and a threshold.
    Returns in output the same parameters with zeroed-out values below the threshold * max absolute value,
    as well as the reverse of the mask (True if parameter has not been zero-ed out and should be kept,
    False if parameter has been zero-ed out and should be frozen in further training).
    Currently, it does not affect biases (single bias value per layer, i.e. never masked out).
    """
    # Compute min accepted value as threshold `thresh` * max absolute value in a given layer
    min_accepted_value = thresh * jnp.max(jnp.abs(param))
    # return layer parameters with zeroed-out values below `min_accepted_value`
    mask = jnp.abs(param) < min_accepted_value
    new_param = jnp.where(mask, 0.0, param)
    return new_param, ~mask


def zero_out_and_freeze(
        model: DeepMechanisticModel,
        filter_spec: PyTree,
        threshold: float
):
    """
    Takes in input a DeepMechanisticModel, the corresponding filter_spec_per_param (all True) and a threshold.
    Returns in output the same model with zeroed out parameters below
    the threshold * max absolute value in the corresponding layer and category (weights - does not work on biases)
    of a given module, as well as a modified filter_spec_per_param, with zeroed-out values frozen (set to False).
    """

    # Define modules in the model and filter_spec_per_param
    modules = [model.deep_encoder, model.deep_inflater]
    filter_specs = [filter_spec.deep_encoder, filter_spec.deep_inflater]
    if model.reconstruct:
        modules.append(model.deep_decoder)
        filter_specs.append(filter_spec.deep_decoder)

    for module, fs_module in zip(
            modules,
            filter_specs
    ):
        # Iterate through layers, zeroing out weights/biases below the max absolute per-layer value * threshold
        for i, layer in enumerate(module.layers):
            if hasattr(layer, 'weight'):
                module.layers[i] = eqx.tree_at(
                    lambda lyr: lyr.weight, layer, zero_out_layer_params(layer.weight, threshold)[0]
                )
                fs_module.layers[i] = eqx.tree_at(
                    lambda lyr: lyr.weight, fs_module.layers[i], zero_out_layer_params(layer.weight, threshold)[1]
                )
            # Currently not filtering nor affecting biases, as there is only a single bias value per layer,
            # so it would never be masked out with the current strategy + there are only few bias terms.
            # if hasattr(layer, 'bias') and (layer.bias is not None):
            #     module.layers[i] = eqx.tree_at(
            #         lambda lyr: lyr.bias, layer, zero_out_layer_params(layer.bias, threshold)[0]
            #     )
            #     fs_module.layers[i] = eqx.tree_at(
            #         lambda lyr: lyr.bias, fs_module.layers[i], zero_out_layer_params(layer.bias, threshold)[1]
            #     )
    return model, filter_spec


def sparsify_model(
        model: DeepMechanisticModel,
        drop_regularisation_post_pretraining: bool,
        threshold: float,
):
    # Default to training all parameters
    filter_spec = tree_map(lambda _: True, model)

    if drop_regularisation_post_pretraining:
        # Zero out parameters that are below threshold * max in the corresponding layer and category (weight/bias
        # if any) and freeze corresponding zero-ed out parameters
        model, filter_spec = zero_out_and_freeze(model, filter_spec, threshold)
    return model, filter_spec


def apply_filter_to_updates(updates, filter_spec):
    """
    Zeroes out the updates corresponding to False values in the filter_spec_per_param.
    """

    def mask_update(update, mask):
        return jnp.where(mask, update, 0.0)

    # Apply the mask to zero out updates where filter_spec_per_param is False
    masked_updates = tree_map(mask_update, updates, filter_spec)
    return masked_updates


def generate_pypesto_objective(ae: DeepMechanisticModel) -> JaxObjective:
    """
    Creates a pypesto objective function (this is the loss function) that
    needs to be minimized to train the respective autoencoder

    :returns:
        Objective function that needs to be minimized for training.
    """
    # return JaxObjective(objective=ae.pypesto_subproblem.objective)
    return JaxObjective(
        objective=ae.pypesto_subproblem.objective,  # same base objective previously passed to JaxObjective
    )


def create_pypesto_problem(
        ae: DeepMechanisticModel,
) -> pypesto.Problem:
    """
    Creates a pypesto.Problem that defines the optimization problem to solve
    for the training of the provided DeepMechanisticModel/Autoencoder (ae).

    :param ae:
        Autoencoder that will be trained

    :returns:
        Optimization pypesto_problem that needs to be solved for training.
    """

    objective = generate_pypesto_objective(ae)
    return pypesto.Problem(
        objective=objective,
        lb=[-np.inf for _ in objective.x_names],  # extract names from objective
        ub=[np.inf for _ in objective.x_names],
    )


@eqx.filter_jit
def model_output_to_petab_input(
        model: DeepMechanisticModel,
        input_data,
):
    # Get model output (inflated cell-line-specific parameter deviations)
    pred = vmap(model)(input_data)[0]
    # Concatenate learnable global kinetic parameters with pred
    augmented_pred = jnp.concatenate(
        [
            model.kin_params_combiner.learned_global_kin_params,
            pred.flatten()
        ]
    )
    return augmented_pred
