import equinox as eqx
import jax.numpy as jnp
import jax.flatten_util as jfu
import matplotlib.pyplot as plt
import numpy as np
import petab
import pypesto
import seaborn as sns

from amici.petab.simulations import rdatas_to_simulation_df
from .deepcomponent_eqx import DeepComponent
from .dmm_autoencoder_eqx import DeepMechanisticModel
from jax import vmap
from jax.tree_util import tree_map
from jaxtyping import Array, PyTree
from optax import adam, adamw, GradientTransformationExtraArgs, Schedule, sgdr_schedule
from optax.contrib import schedule_free_adamw, schedule_free_eval_params
from pathlib import Path
from pypesto.C import MODE_RES, RDATAS
from pypesto.objective.jax import JaxObjective
from typing import Dict, Optional, Tuple, Union


def get_scheduler(
        conf: Dict,
        n_epoch: int,
        pretraining: bool = False,
) -> Schedule:
    """Get the learning rate scheduler.

    Parameters
    ----------
    conf : configuration object
    n_epoch : int - total number of training epochs
    pretraining : bool - discriminates between network pretraining and full DMM training stages

    Returns
    ----------
    optax.sgdr_schedule
        The learning rate scheduler.
    """
    if pretraining:
        max_lrate = conf["max_lrate"]/conf["lrate_pretraining_ratio"]
    else:
        max_lrate = conf["max_lrate"]

    if conf["use_simple_linear_schedule"]:
        # Define custom steps to use the same machinery as below - schedule config should
        # be entirely within conf object
        schedules = [
            {
                'init_value': max_lrate / conf["lrate_span"],  # before warm-up
                'peak_value': max_lrate,  # after warm-up
                'warmup_steps': int(n_epoch * conf["warmup_fct"]),
                'decay_steps': n_epoch - int(n_epoch * conf["warmup_fct"]),  # n_epoch - warmup steps
                'end_value': max_lrate * conf["lrate_decay"]**n_epoch,  # after decay
            }  # single linear schedule
        ]
    else:
        epochs_per_schedule = np.array([
            conf["opt_steps"] * (conf["opt_mult"] ** i)
            for i in range(int(n_epoch // conf["opt_steps"]))
            if conf["opt_steps"] * (conf["opt_mult"] ** i) <= n_epoch
        ])
        schedules = [
            {
                'init_value': conf["max_lrate"] / conf["lrate_span"] * conf["lrate_decay"] ** i_schedule,
                'warmup_steps': int(
                    (conf["opt_steps"] * (conf["opt_mult"] ** i_schedule))
                    * conf["warmup_fct"]
                ),
                'decay_steps': int(conf["opt_steps"] * (conf["opt_mult"] ** i_schedule)),
                'end_value': max_lrate/ conf["lrate_span"] * conf["lrate_decay"] ** (i_schedule + 1),
            }
            for i_schedule in range(len(epochs_per_schedule))
        ]
    return sgdr_schedule(schedules)


def get_optimiser_and_opt_state(
        conf: Dict,
        n_epoch: int,
        model: DeepMechanisticModel,
        filter_spec: Optional[PyTree] = None
) -> Tuple[GradientTransformationExtraArgs, PyTree]:
    """
    Returns the optimiser and optimiser state for training the model.
    :param conf:
        configuration object (dmm.config_options -> Conf) converted to dictionary.
    :param n_epoch:
        number of training epochs.
    :param model:
        DeepMechanisticModel instance.
    :param filter_spec:
        Optional filter specification for the model parameters.

    :return:
        Tuple containing the optimiser and optimiser state.
    """
    # Get dynamic model parameters
    if filter_spec is not None:
        diff_model, _ = eqx.partition(model, filter_spec)
    else:
        diff_model, _ = eqx.partition(model, eqx.is_array)
    # Initialise optimiser and optimiser state
    if conf["optimiser"] == 'adamw_sf':  # sf = schedule-free
        opt = schedule_free_adamw(
            learning_rate=conf["max_lrate"],
            warmup_steps=int(n_epoch * conf["warmup_fct"]),
            b1=conf["momentum"],
            weight_decay=conf["weight_decay"],
        )
        flat_params, _ = jfu.ravel_pytree(diff_model)
        opt_state = opt.init(flat_params)
        return opt, opt_state
    elif conf["optimiser"] == 'adam':
        optimiser = adam
        extra_args = None
    elif conf["optimiser"] == 'adamw':
        optimiser = adamw
        extra_args = {"weight_decay": conf["weight_decay"]}
    else:
        raise ValueError(f"Unknown optimiser: {conf['optimiser']}")
    # If not schedule-free, get schedule and initialise optimiser and optimiser state accordingly
    schedule = get_scheduler(conf, n_epoch)
    if extra_args is not None:
        opt = optimiser(schedule, **extra_args)
    else:
        opt = optimiser(schedule)
    opt_state = opt.init(diff_model)
    return opt, opt_state


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
        input_data: np.ndarray,
):
    # Get model output (inflated cell-line-specific parameter deviations)
    pred = vmap(model)(input_data)["inflated"]
    # Concatenate learnable global kinetic parameters with pred
    augmented_pred = jnp.concatenate(
        [
            model.kin_params_combiner.learned_global_kin_params,
            pred.flatten()
        ]
    )
    return augmented_pred


def model_output_to_petab_input_nojit(
        model: DeepMechanisticModel,
        input_data: np.ndarray,
):
    # Get model output (inflated cell-line-specific parameter deviations)
    pred = vmap(model)(input_data)["inflated"]
    # Concatenate learnable global kinetic parameters with pred
    augmented_pred = jnp.concatenate(
        [
            model.kin_params_combiner.learned_global_kin_params,
            pred.flatten()
        ]
    )
    return augmented_pred


def enforce_minimum_spacing(arr: np.ndarray, min_dist: int) -> np.ndarray:
    """
    Filters an input array, arr, and keeps the original items that are at least min_dist apart.
    """
    prev = - np.inf  # ensure first element is always kept
    return np.array([prev := x for x in arr if x - prev >= min_dist])


def get_eval_model(
        conf: Dict,
        model: DeepMechanisticModel,
        opt_state: PyTree,
        filter_spec: Optional[PyTree]
) -> DeepMechanisticModel:
    """
    Returns the evaluation model for schedule-free learning, the model itself otherwise.
    For schedule-free learning, optimiser tracks sequence of iterates `y`, on which gradients are evaluated.
    Optimiser state keeps track of sequence of iterates `z`. Weights needed to evaluate the model, `x`, need to
    be computed on the fly and stored in the model for accurate evaluation.

    :param conf:
        configuration object (dmm.config_options -> Conf) converted to dictionary.
    :param model:
        DeepMechanisticModel instance.
    :param opt_state:
        optimiser state.

    :return:
        Evaluation model for schedule-free learning, the model itself otherwise.
    """
    # For schedule-free learning, we need to get the evaluation parameters
    if conf["optimiser"] == "adamw_sf":
        if filter_spec is not None:
            diff_model, static_model = eqx.partition(model, filter_spec)
        else:
            diff_model, static_model = eqx.partition(model, eqx.is_array)
        flat_params, unflatten_params = jfu.ravel_pytree(diff_model)
        eval_params = schedule_free_eval_params(opt_state, flat_params)
        eval_diff_model = unflatten_params(eval_params)
        eval_model = eqx.combine(eval_diff_model, static_model)
    else:
        # if not using schedule-free, we can just use the previous step model (not next_model)
        eval_model = model
    return eval_model


def generate_log_epochs(n_epoch: int, num_samples: int, min_dist: int) -> np.ndarray:
    """
    Returns epochs regularly spaced in log10 space but no closer than min_dist
    to prevent running into filestream rate limit in W&B.
    """
    log_epochs = np.unique(
        np.logspace(0, np.log10(n_epoch), num=num_samples).astype(int)
    )
    return enforce_minimum_spacing(log_epochs, min_dist)


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


def rmse(
        pp,
        model: DeepMechanisticModel,
        input_data
):
    try:
        x = model_output_to_petab_input(model, input_data)
        obj = pp.objective.base_objective
        amici_model = obj.amici_model
        petab_problem = obj.amici_object_builder.petab_problem
        res = obj(x, mode=MODE_RES, return_dict=True)
        simulation_df = rdatas_to_simulation_df(
            res[RDATAS],
            model=amici_model,
            measurement_df=petab_problem.measurement_df,
        )
        return np.sqrt(
            np.mean(
                np.square(
                    simulation_df[petab.SIMULATION]
                    - petab_problem.measurement_df[petab.MEASUREMENT]
                )
            )
        )
    except Exception as e:
        print(e)
        return np.NaN


def test_save_reload_model(
        model: DeepMechanisticModel,
        model_filename: Path,
        samples_name_list_dict: dict,
        cytof_problem,  # avoids importing from CytofProblem
        petab_base_files,  # avoids importing from util
        dataset: str,
        input_data: np.ndarray
):
    model_filename.parent.mkdir(exist_ok=True, parents=True)
    # Save
    model.save(model_filename, samples_name_list_dict)

    # Use class method to load an instance from file
    re_model = DeepMechanisticModel.load(
        filename=model_filename,
        problem=cytof_problem,
        dataset=dataset,
        petab_base_files=petab_base_files,
    )

    # TODO add checks on RMSE -- need to import problem_train, problem_test and compute RMSE on both
    # return RMSE on validation and assert it's the same as the best one // could write another function for this
    assert (vmap(model)(input_data)["inflated"] == vmap(re_model)(input_data)["inflated"]).all()
    assert (vmap(model)(input_data)["decoded"] == vmap(re_model)(input_data)["decoded"]).all()


def check_best_model(
        best_model_filename: Path,
        cytof_problem,  # avoids importing from CytofProblem
        petab_base_files,  # avoids importing from util
        input_data: np.ndarray,
        pp: pypesto.Problem,
        best_rmse_val: float,
):
    # Use class method to load an instance from file
    re_model = DeepMechanisticModel.load(
        filename=best_model_filename,
        problem=cytof_problem,
        dataset="test",
        petab_base_files=petab_base_files,
    )
    # Compute RMSE with reloaded model
    re_model_rmse_val = rmse(pp, re_model, input_data)

    print(f"Reloaded model RMSE val: {re_model_rmse_val}, original RMSE val: {best_rmse_val}")

    # Cannot assert equality on NaN or inf values
    if (not np.isfinite(best_rmse_val)) and (not np.isfinite(re_model_rmse_val)):
        pass
    else:
        assert re_model_rmse_val == best_rmse_val
