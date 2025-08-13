from pathlib import Path
from typing import Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import petab.v1 as petab
import pypesto
import seaborn as sns
from amici import AMICI_SUCCESS
from amici.petab.simulations import rdatas_to_simulation_df
from jax import vmap
from jax.tree_util import tree_map
from jaxtyping import Array
from optax import (
    GradientTransformationExtraArgs,
    OptState,
    Schedule,
    adam,
    adamw,
    constant_schedule,
    inject_hyperparams,
    piecewise_interpolate_schedule,
    sgdr_schedule,
)
from pypesto.C import MODE_RES, RDATAS, ModeType
from pypesto.objective.base import ResultDict
from pypesto.objective.jax import JaxObjective

import wandb

from .config_options import Conf
from .deepcomponent_eqx import DeepComponent
from .dmm_autoencoder_eqx import DeepMechanisticModel


def get_scheduler(
    conf: Conf,
    n_epoch: int,
) -> Schedule:
    """Get the learning rate scheduler.

    Parameters
    ----------
    conf : configuration object
    n_epoch : int - total number of training epochs

    Returns
    -------
    optax.sgdr_schedule
        The learning rate scheduler.

    """

    if conf.use_simple_linear_schedule:
        # Bypass for constant schedule if needed
        if (
            conf.lrate_decay == 1
            and conf.lrate_span == 1
            and conf.warmup_fct == 0
        ):
            return constant_schedule(conf.max_lrate)

        assert (
            conf.lrate_span >= 1
        ), "lrate_span must be greater than or equal to 1!"
        assert (
            conf.lrate_decay >= 0
        ), "lrate_decay must be greater than or equal to 0!"
        assert (
            conf.warmup_fct >= 0
        ), "warmup_fct must be greater than or equal to 0!"
        assert conf.warmup_fct < 1, "warmup_fct must be less than 1!"

        # Handle warmup/no warmup
        if conf.warmup_fct > 0:
            boundaries_and_scales = {
                int(conf.warmup_fct * n_epoch): conf.lrate_span,
                n_epoch - 1: conf.lrate_decay,
            }
        else:
            boundaries_and_scales = {
                n_epoch - 1: conf.lrate_span,
            }
        return piecewise_interpolate_schedule(
            interpolate_type="linear",
            init_value=conf.max_lrate / conf.lrate_span,
            boundaries_and_scales=boundaries_and_scales,
        )
    else:
        # Cosine annealing
        epochs_per_schedule = np.array(
            [
                conf.opt_steps * (conf.opt_mult**i)
                for i in range(int(n_epoch // conf.opt_steps))
                if conf.opt_steps * (conf.opt_mult**i) <= n_epoch
            ]
        )
        schedules = [
            {
                "init_value": conf.max_lrate
                / conf.lrate_span
                * conf.lrate_decay**i_schedule,
                "peak_value": conf.max_lrate * conf.lrate_decay**i_schedule,
                "warmup_steps": int(
                    (conf.opt_steps * (conf.opt_mult**i_schedule))
                    * conf.warmup_fct
                ),
                "decay_steps": int(
                    conf.opt_steps * (conf.opt_mult**i_schedule)
                ),
                "end_value": conf.max_lrate
                / conf.lrate_span
                * conf.lrate_decay ** (i_schedule + 1),
            }
            for i_schedule in range(len(epochs_per_schedule))
        ]
        return sgdr_schedule(schedules)


def get_optimiser_and_opt_state(
    conf: Conf,
    n_epoch: int,
    model: DeepMechanisticModel,
    log_wandb: bool = False,
) -> Tuple[GradientTransformationExtraArgs, OptState]:
    """Returns the optimiser and optimiser state for training the model.
    :param conf:
        configuration object (dmm.config_options -> Conf) converted to dictionary.
    :param n_epoch:
        number of training epochs.
    :param model:
        DeepMechanisticModel instance.
    :param log_wandb:
        boolean flag to log learning rate schedule chart to wandb.

    :return:
        Tuple containing the optimiser and optimiser state.
    """
    # Get dynamic model parameters
    diff_model, _ = eqx.partition(model, eqx.is_array)

    # Initialise optimiser and optimiser state
    if conf.optimiser == "adam":
        optimiser = adam
        extra_args = {}
    elif conf.optimiser == "adamw":
        optimiser = adamw
        extra_args = {"weight_decay": conf.weight_decay}
    else:
        raise ValueError(f"Unknown optimiser: {conf.optimiser}")

    # Get schedule and initialise optimiser and optimiser state accordingly
    schedule = get_scheduler(conf, n_epoch)

    if log_wandb:  # do not log by default
        # Log learning rate schedule chart to wandb
        plt.plot(jnp.arange(n_epoch), schedule(jnp.arange(n_epoch)))
        plt.ylabel("Learning Rate")
        plt.xlabel("Epoch")
        wandb.log({"Learning Rate Schedule": plt}, step=0)

    opt = inject_hyperparams(optimiser)(learning_rate=schedule, **extra_args)
    opt_state = opt.init(diff_model)
    return opt, opt_state


def get_finite_grads(grads):
    return tree_map(
        lambda x: jnp.where(jnp.isfinite(x), x, jnp.zeros_like(x)),
        grads,
    )


def get_parameters(module: Union[DeepComponent, eqx.Module]) -> jnp.ndarray:
    params = []
    for layer in module.layers:
        params.append(layer.weight.flatten())

        # Check if the layer has a 'bias' attribute and append if it does
        if hasattr(layer, "bias") and layer.bias is not None:
            params.append(layer.bias.flatten())

    # Concatenate into single output array
    module_params = jnp.concatenate(params)
    return module_params


def map_params_to_array(model: DeepMechanisticModel) -> jnp.ndarray:
    encoder_params = get_parameters(model.deep_encoder)
    inflater_params = get_parameters(model.deep_inflater)
    param_array = jnp.concatenate(
        [
            module_params.flatten()
            for module_params in [
                encoder_params,
                inflater_params,
            ]
        ]
    )
    if isinstance(model.deep_decoder, DeepComponent):
        decoder_params = get_parameters(model.deep_decoder)
        param_array = jnp.concatenate(
            [param_array.flatten(), decoder_params.flatten()]
        )
    param_array = jnp.concatenate(
        [
            param_array,
            model.kin_params_combiner.learned_global_kin_params.flatten(),
        ]
    )
    return param_array


def zero_out_layer_params(
    param: Array,
    thresh: float,
):
    """Takes in input the parameters (weights/biases) of a layer and a threshold.
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


class Chi2Objective(pypesto.objective.Objective):
    base_objective: pypesto.objective.AmiciObjective

    def __init__(self, base_objective):
        self.base_objective = base_objective

    def fun(self, x: np.ndarray, **kwargs) -> np.ndarray:
        return self.call_unprocessed(x, (0,), pypesto.C.MODE_FUN, **kwargs)[
            pypesto.C.FVAL
        ]

    def grad(self, x: np.ndarray, **kwargs) -> np.ndarray:
        return self.call_unprocessed(x, (1,), pypesto.C.MODE_FUN, **kwargs)[
            pypesto.C.GRAD
        ]

    @property
    def x_names(self) -> list[str]:
        return self.base_objective.x_names

    @property
    def pre_post_processor(self):
        return self.base_objective.pre_post_processor

    @pre_post_processor.setter
    def pre_post_processor(self, pre_post_processor):
        self.base_objective.pre_post_processor = pre_post_processor

    @property
    def amici_model(self):
        return self.base_objective.amici_model

    @property
    def history(self):
        return self.base_objective.history

    @history.setter
    def history(self, history):
        self.base_objective.history = history

    @property
    def amici_object_builder(self):
        return self.base_objective.amici_object_builder

    @property
    def res(self):
        return self.base_objective.res

    @property
    def sres(self):
        return self.base_objective.sres

    @property
    def has_hess(self):
        return False

    def call_unprocessed(
        self,
        x: np.ndarray,
        sensi_orders: tuple[int, ...],
        mode: ModeType,
        **kwargs,
    ) -> ResultDict:
        assert mode in [pypesto.C.MODE_FUN], "Only residual mode is supported"
        # TODO @FabianFrohlich: add some additional safeguards
        res = self.base_objective.call_unprocessed(
            x, sensi_orders, mode, return_dict=True, **kwargs
        )
        ndata = sum(
            sum(np.not_equal(r[pypesto.C.RES], 0.0))
            for r in res[RDATAS]
            if r.status == AMICI_SUCCESS
        )

        ret = {}
        if 0 in sensi_orders:
            mse = sum(
                r["chi2"] for r in res[RDATAS] if r.status == AMICI_SUCCESS
            ) / max(ndata, 1)
            if not all(
                r.status == AMICI_SUCCESS for r in res[RDATAS]
            ):  # catch failure and set MSE to inf -> loss will be inf -> will be caught by patience counter
                mse = np.inf
            ret[pypesto.C.FVAL] = mse
        if 1 in sensi_orders:
            smse = res[pypesto.C.GRAD] / max(ndata, 1)
            ret[pypesto.C.GRAD] = smse
        return ret


def generate_pypesto_objective(pypesto_subproblem) -> JaxObjective:
    """Creates a pypesto objective function (this is the loss function) that
    needs to be minimized to train the respective dmm

    :returns:
        Objective function that needs to be minimized for training.
    """
    return JaxObjective(
        objective=Chi2Objective(pypesto_subproblem.objective),
    )


def create_pypesto_problem(
    subproblem: pypesto.Problem,
) -> pypesto.Problem:
    """Creates a pypesto.Problem that defines the optimization problem to solve
    for the training of the provided DeepMechanisticModel/Autoencoder (ae).

    :param ae:
        Autoencoder that will be trained

    :returns:
        Optimization pypesto_problem that needs to be solved for training.
    """
    objective = generate_pypesto_objective(subproblem)
    return pypesto.Problem(
        objective=objective,
        lb=[
            -np.inf for _ in objective.x_names
        ],  # extract names from objective
        ub=[np.inf for _ in objective.x_names],
    )


@eqx.filter_jit
def model_output_to_petab_input(
    model: DeepMechanisticModel,
    input_data: np.ndarray,
    key,
):
    # Get model output (inflated cell-line-specific parameter deviations)
    pred = model.inflate_params(input_data, key)
    # Concatenate learnable global kinetic parameters (medians) with predicted deviations
    augmented_pred = jnp.concatenate(
        [model.kin_params_combiner.learned_global_kin_params, pred.flatten()]
    )
    return augmented_pred


# Only used in training
@eqx.filter_jit
def model_output_to_petab_input_frozen_medians(
    model: DeepMechanisticModel,
    input_data: np.ndarray,
    key: jr.PRNGKey,
):
    # Get model output (inflated cell-line-specific parameter deviations)
    pred = model.inflate_params(input_data, key)
    # Concatenate FROZEN global kinetic parameters with predicted deviations
    augmented_pred = jnp.concatenate(
        [
            jax.lax.stop_gradient(
                model.kin_params_combiner.learned_global_kin_params
            ),
            pred.flatten(),
        ]
    )
    return augmented_pred


def model_output_to_petab_input_nojit(
    model: DeepMechanisticModel,
    input_data: np.ndarray,
    key: jr.PRNGKey,
):
    # Get model output (inflated cell-line-specific parameter deviations)
    pred = model.inflate_params(input_data, key)
    # Concatenate learnable global kinetic parameters with pred
    augmented_pred = jnp.concatenate(
        [model.kin_params_combiner.learned_global_kin_params, pred.flatten()]
    )
    return augmented_pred


def enforce_minimum_spacing(arr: np.ndarray, min_dist: int) -> np.ndarray:
    """Filters an input array, arr, and keeps the original items that are at least min_dist apart."""
    prev = -np.inf  # ensure first element is always kept
    return np.array([prev := x for x in arr if x - prev >= min_dist])


def generate_log_epochs(
    n_epoch: int, num_samples: int, min_dist: int
) -> np.ndarray:
    """Returns epochs regularly spaced in log10 space but no closer than
    min_dist to prevent running into filestream rate limit in W&B.
    """
    log_epochs = np.unique(
        np.logspace(0, np.log10(n_epoch), num=num_samples).astype(int)
    )
    return enforce_minimum_spacing(log_epochs, min_dist)


def plot_model_weights(model: DeepMechanisticModel, filename: str = None):
    num_columns = (
        len([*model.deep_encoder.layers, *model.deep_inflater.layers]) + 2
    )
    single_module_height = model.module_depth * 2 + 1
    num_rows = 2 * single_module_height + 3

    fig = plt.figure()
    fig.set_figheight(num_rows)
    fig.set_figwidth(num_columns)

    # Plot input - features
    ax = plt.subplot2grid(
        shape=(num_rows, num_columns),
        loc=(int(num_rows / 2) - len(model.deep_encoder.layers) + 1, 0),
        colspan=1,
        rowspan=single_module_height,
    )
    sns.heatmap(
        np.zeros(shape=(model.deep_encoder.layers[0].weight.T.shape[0], 1)),
        cbar=False,
        xticklabels=False,
        yticklabels=False,
        ax=ax,
    )
    ax.set_title(f"in: {model.deep_encoder.layers[0].weight.T.shape[0]}")
    for i, layer in enumerate(model.deep_encoder.layers[::-1]):
        ax = plt.subplot2grid(
            shape=(num_rows, num_columns),
            loc=(int(num_rows / 2) - i, len(model.deep_encoder.layers) - i),
            colspan=1,
            rowspan=1 + 2 * i,
        )
        sns.heatmap(
            layer.weight.T,
            cbar=False,
            xticklabels=False,
            yticklabels=False,
            ax=ax,
        )
        ax.set_title(f"e.{i}: {layer.weight.T.shape}")

    # Plot output - kinetic parameters
    ax = plt.subplot2grid(
        shape=(num_rows, num_columns),
        loc=(0, num_columns - 1),
        colspan=1,
        rowspan=single_module_height,
    )
    sns.heatmap(
        np.zeros(shape=(model.deep_inflater.layers[-1].weight.T.shape[0], 1)),
        cbar=False,
        xticklabels=False,
        yticklabels=False,
        ax=ax,
    )
    ax.set_title(f"out: {model.deep_inflater.layers[-1].weight.shape[0]}")

    for i, layer in enumerate(model.deep_inflater.layers[::-1]):
        ax = plt.subplot2grid(
            shape=(num_rows, num_columns),
            loc=(i, num_columns - 2 - i),
            colspan=1,
            rowspan=single_module_height - 2 * i,
        )
        sns.heatmap(
            layer.weight.T,
            cbar=False,
            xticklabels=False,
            yticklabels=False,
            ax=ax,
        )
        ax.set_title(
            f"i.{len(model.deep_inflater.layers) - i}: {layer.weight.T.shape}"
        )

    if model.reconstruct:
        # Plot input - features
        ax = plt.subplot2grid(
            shape=(num_rows, num_columns),
            loc=(2 + int(num_rows / 2), num_columns - 1),
            colspan=1,
            rowspan=single_module_height,
        )
        sns.heatmap(
            np.zeros(
                shape=(model.deep_encoder.layers[0].weight.T.shape[0], 1)
            ),
            cbar=False,
            xticklabels=False,
            yticklabels=False,
            ax=ax,
        )
        ax.set_title(
            r"$\rm \widehat{in}$:"
            + str(model.deep_encoder.layers[0].weight.T.shape[0])
        )

        for i, layer in enumerate(model.deep_decoder.layers[::-1]):
            ax = plt.subplot2grid(
                shape=(num_rows, num_columns),
                loc=(num_rows - single_module_height + i, num_columns - 2 - i),
                colspan=1,
                rowspan=single_module_height - 2 * i,
            )
            sns.heatmap(
                layer.weight,
                cbar=False,
                xticklabels=False,
                yticklabels=False,
                ax=ax,
            )
            ax.set_title(
                f"d.{len(model.deep_decoder.layers) - i}: "
                f"{layer.weight.T.shape}"
            )

    if filename is not None:
        plt.savefig(filename, facecolor="w")
    plt.show()


def compute_simulation_from_model(
    pp,
    model: DeepMechanisticModel,
    input_data: jnp.ndarray,
    return_petab_problem: bool = False,
):
    # put model in inference mode and use dummy key
    x = model_output_to_petab_input(
        eqx.nn.inference_mode(model), input_data, jr.PRNGKey(0)
    )
    obj = pp.objective.base_objective.base_objective
    amici_model = obj.amici_model
    petab_problem = obj.amici_object_builder.petab_problem
    res = obj(x, mode=MODE_RES, return_dict=True)
    simulation_df = rdatas_to_simulation_df(
        res[RDATAS],
        model=amici_model,
        measurement_df=petab_problem.measurement_df,
    )
    return (
        (simulation_df, petab_problem)
        if return_petab_problem
        else simulation_df
    )


def rmse(pp, model: DeepMechanisticModel, input_data: np.ndarray):
    try:
        simulation_df, petab_problem = compute_simulation_from_model(
            pp=pp,
            model=model,
            input_data=input_data,
            return_petab_problem=True,
        )
        is_cytof = simulation_df[petab.OBSERVABLE_ID].str.startswith("p")
        residuals = (
            simulation_df[petab.SIMULATION]
            - petab_problem.measurement_df[petab.MEASUREMENT]
        ) / simulation_df[petab.NOISE_PARAMETERS]
        return np.sqrt(np.mean(np.square(residuals[is_cytof].values)))
    except Exception as e:
        print(e)
        return np.inf


def rmse_ensemble(
    pp, best_models: list[tuple[int, float, DeepMechanisticModel]], input_data
):
    try:
        residuals = []
        for _, _, model in best_models:
            simulation_df, petab_problem = compute_simulation_from_model(
                pp=pp,
                model=model,
                input_data=input_data,
                return_petab_problem=True,
            )
            # track residuals for each model
            residuals.append(
                (
                    simulation_df[petab.SIMULATION]
                    - petab_problem.measurement_df[petab.MEASUREMENT]
                ).values
            )
        return np.sqrt(
            np.mean(
                np.square(
                    np.array(residuals).mean(axis=0)
                    # equivalent to the residual of the mean simulation
                )
            )
        )
    except Exception as e:
        print(e)
        return np.inf


def test_save_reload_model(
    model: DeepMechanisticModel,
    model_filename: Path,
    samples_name_list_dict: dict,
    cytof_problem,  # avoids importing from CytofProblem
    petab_base_files,  # avoids importing from util
    dataset: str,
    input_data: np.ndarray,
):
    # Set model to inference mode
    model = eqx.nn.inference_mode(model)

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
    re_model = eqx.nn.inference_mode(re_model)

    # TODO(Giacomo): add checks on RMSE -- need to import problem_train,
    #  problem_test and compute RMSE on both
    # return RMSE on validation and assert it's the same as the best one
    # could write another function for this
    assert (
        model.inflate_params(input_data, key=jr.PRNGKey(0))
        == re_model.inflate_params(input_data, jr.PRNGKey(0))
    ).all()
    assert (
        vmap(model.decode)(input_data, jr.PRNGKey(0))
        == vmap(re_model.decode)(input_data, jr.PRNGKey(0))
    ).all()


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

    print(
        f"Reloaded model RMSE val: {re_model_rmse_val}, "
        f"original RMSE val: {best_rmse_val}"
    )

    # Cannot assert equality on NaN or inf values
    if (not np.isfinite(best_rmse_val)) and (
        not np.isfinite(re_model_rmse_val)
    ):
        pass
    else:
        assert re_model_rmse_val == best_rmse_val


class MetricHandler:
    def __init__(
        self, patience=2
    ):  # reduced default patience down to 2 (from 5)
        self.counter = 0
        self.patience = patience
        self.invalid_metric_detected = (
            False  # Flag to track invalid metrics in the current epoch
        )

    def handle_invalid_metrics(self, metrics, epoch):
        # Lower flags
        should_break = False
        self.invalid_metric_detected = False

        for metric in metrics:
            if not np.isfinite(metric):  # if any metric is invalid, raise flag
                self.invalid_metric_detected = (
                    True  # Set flag if any metric is invalid
                )

        if self.invalid_metric_detected:
            self.counter += 1
            if self.counter >= self.patience:  # fixed budget of patience
                print(f"Too many invalid values, breaking at epoch {epoch}")
                wandb.log(
                    {
                        "integration_error": epoch,
                    }
                )
                should_break = True

        return should_break


def plot_inflater_encoder_product(model: DeepMechanisticModel, filepath: Path):
    # Compute the encoder matrix product
    encoder_mat = model.deep_encoder.layers[0].weight
    for layer in model.deep_encoder.layers[1:]:
        encoder_mat = np.matmul(layer.weight, encoder_mat)

    # Compute the inflater matrix product
    inflater_mat = model.deep_inflater.layers[0].weight
    for layer in model.deep_inflater.layers[1:]:
        inflater_mat = np.matmul(layer.weight, inflater_mat)

    # Compute the product of inflater and encoder matrices
    inflater_encoder_product = np.matmul(inflater_mat, encoder_mat)

    # Plot
    plt.imshow(inflater_encoder_product)
    plt.colorbar()
    plt.savefig(filepath)
    plt.show()
