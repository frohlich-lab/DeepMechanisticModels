import equinox as eqx
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pypesto
import seaborn as sns

from common import Conf
from cytof.problem import CytofProblem
from dmm.deepcomponent_eqx import DeepComponent
from dmm.dmm_autoencoder_eqx import DeepMechanisticModel
from jax import vmap
from jax.tree_util import tree_map
from jaxtyping import Array, PyTree
from pathlib import Path
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


def remove_close_elements(arr: np.ndarray, min_dist: int) -> np.ndarray:
    # Initialize with the first element
    filtered_arr = [arr[0]]

    # Iterate through the array starting from the second element
    for elem in arr[1:]:
        if elem - filtered_arr[-1] >= min_dist:
            filtered_arr.append(elem)
    return np.array(filtered_arr)


def generate_log_epochs(n_epoch: int, num_samples: int, min_dist: int) -> np.ndarray:
    """
    Returns epochs regularly spaced in log10 space but no closer than min_dist
    to prevent running into filestream rate limit in W&B.
    """
    log_epochs = np.unique(
        np.logspace(0, np.log10(n_epoch), num=num_samples).astype(int)
    )
    return remove_close_elements(log_epochs, min_dist)


def plot_model_weights(model: DeepMechanisticModel, filename: str = None):
    num_columns = len([*model.deep_encoder.layers, *model.deep_inflater.layers]) + 2
    single_module_height = (model.module_depth) * 2 + 1
    num_rows = 2 * single_module_height + 3

    fig = plt.figure()
    fig.set_figheight(num_rows)
    fig.set_figwidth(num_columns)

    # Plot input - features
    ax = plt.subplot2grid(
        shape=(num_rows, num_columns),
        loc=(int(num_rows / 2) - len(model.deep_encoder.layers) + 1, 0),
        colspan=1,
        rowspan=single_module_height
    )
    sns.heatmap(
        np.zeros(shape=(model.deep_encoder.layers[0].weight.T.shape[0], 1)),
        cbar=False, xticklabels=False, yticklabels=False, ax=ax
    )
    ax.set_title(f"in: {model.deep_encoder.layers[0].weight.T.shape[0]}")
    for i, layer in enumerate(model.deep_encoder.layers[::-1]):
        ax = plt.subplot2grid(
            shape=(num_rows, num_columns),
            loc=(int(num_rows / 2) - i, len(model.deep_encoder.layers) - i),
            colspan=1,
            rowspan=1 + 2 * i
        )
        sns.heatmap(layer.weight.T, cbar=False, xticklabels=False, yticklabels=False, ax=ax)
        ax.set_title(f"e.{i}: {layer.weight.T.shape}")

    # Plot output - kinetic parameters
    ax = plt.subplot2grid(
        shape=(num_rows, num_columns),
        loc=(0, num_columns - 1),
        colspan=1,
        rowspan=single_module_height
    )
    sns.heatmap(
        np.zeros(shape=(model.deep_inflater.layers[-1].weight.T.shape[0], 1)),
        cbar=False, xticklabels=False, yticklabels=False, ax=ax
    )
    ax.set_title(f"out: {model.deep_inflater.layers[-1].weight.shape[0]}")

    for i, layer in enumerate(model.deep_inflater.layers[::-1]):
        ax = plt.subplot2grid(
            shape=(num_rows, num_columns),
            loc=(i, num_columns - 2 - i),
            colspan=1,
            rowspan=single_module_height - 2 * i
        )
        sns.heatmap(layer.weight.T, cbar=False, xticklabels=False, yticklabels=False, ax=ax)
        ax.set_title(f"i.{len(model.deep_inflater.layers) - i}: {layer.weight.T.shape}")

    if model.reconstruct:
        # Plot input - features
        ax = plt.subplot2grid(
            shape=(num_rows, num_columns),
            loc=(2 + int(num_rows / 2), num_columns - 1),
            colspan=1,
            rowspan=single_module_height
        )
        sns.heatmap(
            np.zeros(shape=(model.deep_encoder.layers[0].weight.T.shape[0], 1)),
            cbar=False, xticklabels=False, yticklabels=False, ax=ax
        )
        ax.set_title(r"$\rm \widehat{in}$:" + str(model.deep_encoder.layers[0].weight.T.shape[0]))

        for i, layer in enumerate(model.deep_decoder.layers[::-1]):
            ax = plt.subplot2grid(
                shape=(num_rows, num_columns),
                loc=(num_rows - single_module_height + i, num_columns - 2 - i),
                colspan=1,
                rowspan=single_module_height - 2 * i
            )
            sns.heatmap(layer.weight, cbar=False, xticklabels=False, yticklabels=False, ax=ax)
            ax.set_title(f"d.{len(model.deep_decoder.layers) - i}: {layer.weight.T.shape}")

    if filename is not None:
        plt.savefig(filename, facecolor='w')
    plt.show()


def test_save_reload_model(
        model: DeepMechanisticModel,
        filename: Path,
        samples_name_list_dict: dict,
        conf: Conf,
        dataset: str,
        input_data: np.ndarray
):
    filename.parent.mkdir(exist_ok=True, parents=True)
    # Save
    model.save(filename, samples_name_list_dict)

    # Get cytof problem
    cytof_problem = CytofProblem(conf.model)
    # Get petab_base_files
    petab_base_files = load_petab_base_files(conf, reweight=True)
    # Use class method to load an instance from file
    re_model = DeepMechanisticModel.load(
        filename=filename,
        problem=cytof_problem,
        dataset=dataset,
        petab_base_files=petab_base_files,
    )

    assert (vmap(model)(input_data)[0] == vmap(re_model)(input_data)[0]).all()
    assert (vmap(model)(input_data)[1] == vmap(re_model)(input_data)[1]).all()
