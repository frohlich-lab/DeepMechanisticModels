import equinox as eqx
import jax
import numpy as np
import optax
import pypesto
import subprocess
import wandb

# from common import TRAINED_MODEL_WEIGHT_PLOTS  # TODO - fix imports from common (not in use)
from .config_options import EarlyStoppingParams
from .dmm_autoencoder_eqx import DeepMechanisticModel
# from .training_helper_funcs import test_save_reload_model, plot_model_weights
from .wandb_init_log import log_extra_loss_terms, log_model_stats
from .training_helper_funcs import (apply_filter_to_updates, generate_log_epochs, get_eval_model, get_finite_grads,
                                    get_optimiser_and_opt_state, map_params_to_array, model_output_to_petab_input,
                                    rmse, rmse_ensemble)
from flax.training.early_stopping import EarlyStopping
# doc: flax.readthedocs.io/en/latest/_modules/flax/training/early_stopping.html
from jaxtyping import Array, Float, PyTree
from pathlib import Path
from typing import Dict

trace_path = Path(__file__).parents[1] / "traces"
TRACE_FILE_TEMPLATE = "{pathway}__{data}__{n_hidden}__{job}__{{id}}.csv"


def update_best_models(
        model: DeepMechanisticModel,
        rmse_val: float,
        best_models: list,
        max_models: int
) -> list[tuple[float, DeepMechanisticModel]]:
    """
    Update the list of best models with the current model and rmse_val
    """
    # Insert the new (rmse_val, model) in the appropriate position in the list
    best_models.append((rmse_val, model))
    best_models = sorted(best_models, key=lambda x: x[0])  # Sort in ascending order by rmse_val (first = lowest = best)
    # Keep only the top 'max_models' entries
    return best_models[:max_models]


@eqx.filter_value_and_grad(has_aux=True)
def loss_fn(
        model: DeepMechanisticModel,
        conf: Dict,
        input_data,
        problem_train: pypesto.Problem,
):
    # problem_train.objective() now needs to get in input what was previously the output of the jax_fun, i.e. the output
    # of ae.embedding(x). x contained the parameters (encoder, inflater, kinetic parameters) and ae.embedding(x)
    # transformed the parameters into kinetic parameters (global) + inflated parameters (i.e. input data passed
    # through encoder + inflater and flattened). This is now the first component of the output of the model.
    # call.
    fval = problem_train.objective(
        model_output_to_petab_input(model, input_data)
    )

    loss_value = (
            fval
            + model.orth_encode_reg(scale=conf["oreg_encode"])
            + model.orth_inflate_reg(scale=conf["oreg_inflate"])
    )
    if model.reconstruct:
        loss_value += (
                model.reconstruction_loss(x=input_data, scale=conf["recon_loss"])
                + model.symmetry_loss(scale=conf["symm_reg"])
                + model.orth_decode_reg(scale=conf["oreg_encode"])
        )

    # If keeping regularisation, add L1 regularisation terms (encoder, inflater, optionally decoder)
    if not conf["drop_reg_after_pretrain"]:
        loss_value += (
                model.l1_encode_reg(scale=conf["l1reg_encode"])
                + model.l1_inflate_reg(scale=conf["l1reg_inflate"])
        )
        if model.reconstruct:
            loss_value += model.l1_decode_reg(scale=conf["l1reg_encode"])

    return loss_value, fval


@eqx.filter_jit
def make_step(
        model: DeepMechanisticModel,
        filter_spec_per_param: PyTree,
        opt: optax.GradientTransformation,
        opt_state: PyTree,
        input_data: Float[Array, '...'],  # TODO @GiacomoFabrini fix input data shape?
        problem_train: pypesto.Problem,
        conf: Dict,
):
    (loss_value, fval), grads = loss_fn(
        model,
        conf,
        input_data,
        problem_train,
    )
    grads = get_finite_grads(grads)
    if conf["optimiser"] == "adamw_sf":
        # Need to flatten parameters and gradients for schedule-free optimisation
        diff_model, static_model = eqx.partition(model, eqx.is_array)
        flat_params, unflatten_params = jax.flatten_util.ravel_pytree(diff_model)
        flat_grads, _ = jax.flatten_util.ravel_pytree(grads)
        updates, opt_state = opt.update(flat_grads, opt_state, flat_params)
        # Unflatten the updates, filter them, flatten back to update flat_params
        filtered_updates, _ = jax.flatten_util.ravel_pytree(
            apply_filter_to_updates(unflatten_params(updates), filter_spec_per_param)
        )
        # Update flat model parameters and unflatten into next_diff_model
        flat_params = eqx.apply_updates(flat_params, filtered_updates)
        next_diff_model = unflatten_params(flat_params)
        next_model = eqx.combine(next_diff_model, static_model)
    else:
        updates, opt_state = opt.update(grads, opt_state, model)
        # Zero-out the updates based on filter_spec_per_param (keep where True).
        # This is equivalent to freezing on a per-parameter basis
        filtered_updates = apply_filter_to_updates(updates, filter_spec_per_param)
        # Update model in `next_model`, but keep current one in `model` for current epoch metric logging
        next_model = eqx.apply_updates(model, filtered_updates)
    return next_model, model, opt_state, loss_value, fval, grads


def train(
        model: DeepMechanisticModel,
        filter_spec_per_param: PyTree,
        problem_train: pypesto.Problem,
        problem_test: pypesto.Problem,
        input_features_train,
        input_features_test,
        # rfile: Path,
        model_file: str,
        samples_name_list_dict: dict,
        conf: Dict,
        n_epoch,
        early_stopping_params: EarlyStoppingParams,
        debug_mode: bool = False,
        ensemble_members: int = 5,
) -> list[tuple[float, DeepMechanisticModel]]:
    """
    Trains the provided autoencoder by solving the optimization problem
    generated by :py:func:`create_pypesto_problem`
    """

    # Initialise optimiser and its state
    opt, opt_state = get_optimiser_and_opt_state(
        conf=conf, n_epoch=n_epoch, model=model, filter_spec=None, pretraining=False,
    )

    # Initialise default values for early_stopper, epoch and tolerance for invalid RMSEs (integration errors)
    early_stopper = None
    epoch = 0
    patience_counter_invalid_loss = 0

    # Use pretrained/randomly initialised model (if not pretrained) to get initial rmse_test_min and
    # the collection of best_models for the ensemble
    rmse_test_min = rmse(problem_test, model, input_features_test)
    best_models = [
        (rmse_test_min, model)  # each item comprises the RMSE validation score and the model itself
        for i in range(ensemble_members)
    ]

    # Check Early-stopping parameters have been set correctly and instantiate early stopper
    if conf["use_early_stopping"]:
        if early_stopping_params.patience is None:
            raise ValueError("Patience value for early stopping is undefined.")
        elif early_stopping_params.min_improvement is None:
            raise ValueError("Minimum absolute improvement for early stopping is undefined.")
        else:
            early_stopper = EarlyStopping(
                min_delta=early_stopping_params.min_improvement,
                patience=early_stopping_params.patience
            )

    # Generate regularly log-spaced epochs for early-stopping evaluation + model stat logging (100 points overall)
    log_epochs = generate_log_epochs(n_epoch=n_epoch, num_samples=100, min_dist=5)  # same min_dist as before

    # Training loop
    for epoch in range(1, n_epoch + 1):  # natural counting
        next_model, model, opt_state, loss_train, fval, grads = make_step(
            model=model,
            filter_spec_per_param=filter_spec_per_param,
            opt=opt,
            opt_state=opt_state,
            input_data=input_features_train,
            problem_train=problem_train,
            conf=conf,
        )

        # Log loss_train
        wandb.log(
            {
                "loss": loss_train,
            },
            step=epoch
        )

        # Handle NaN or Inf loss_train arising from simulation errors
        if not (np.isfinite(loss_train)):
            patience_counter_invalid_loss += 1
            if patience_counter_invalid_loss >= 5:  # fixed budget of patience
                print(f"Too many invalid fval values, breaking at epoch {epoch}")
                wandb.log(
                    {
                        "integration_error": epoch,
                    }
                )
                break  # if this happens, we still serialise `best_models` and return it
        else:
            patience_counter_invalid_loss = 0  # reset counter to 0

        # Log extra terms (regularisation)
        log_extra_loss_terms(
            model=model,
            conf=conf,
            input_data=input_features_train,  # use training features for RECON_LOSS
            epoch=epoch,
            nn_pretrain=False,  # full DMM training stage
        )

        # Overwrite model with updated next_model
        model = next_model

        # Get evaluation model (move from y to x params in Schedule-free) // eval_model = model if not Schedule-free
        eval_model = get_eval_model(conf=conf, model=model, opt_state=opt_state, filter_spec=None)

        # Update x - same param array that we had before
        x = map_params_to_array(model)

        # Log RMSE values + check early-stopping criteria at log-spaced epochs
        if epoch in log_epochs:
            rmse_dict = dict()
            for dataset, pp, input_data in zip(
                    ("train", "test"),
                    (problem_train, problem_test),
                    (input_features_train, input_features_test)
            ):
                rmse_dict[dataset] = rmse(pp, eval_model, input_data)

            # Update tally of best models on validation score -- ensemble members
            best_models = update_best_models(
                model=eval_model,
                rmse_val=rmse_dict["test"],
                best_models=best_models,
                max_models=ensemble_members,
            )

            # Compute fval on train/val datasets using eval_model
            fval_train, fval_val = (
                problem.objective(
                    model_output_to_petab_input(eval_model, input_data)
                )
                for problem, input_data in zip(
                    [problem_train, problem_test], [input_features_train, input_features_test]
                )
            )

            # Log RMSE, fval (both train/val) and model stats
            wandb.log(
                {
                    "rmse_train": rmse_dict["train"],
                    "rmse_val": rmse_dict["test"],
                    "fval_train": fval_train,
                    "fval_val": fval_val,
                    **log_model_stats(eval_model, grads, pretrain=False)
                },
                step=epoch
            )

            # Progress/debugging statements
            if debug_mode:
                print(
                    f" | epoch {epoch} "
                    f" | rmse_train {rmse_dict['train']} "
                    f" | rmse_val {rmse_dict['test']} "
                    f" | fval_train {fval_train} "
                    f" | fval_val {fval_val} | "
                )

            if conf["use_early_stopping"]:
                # Update early stopper
                early_stopper = early_stopper.update(rmse_dict["test"])
                # Debugging statements
                if debug_mode:
                    print(
                        f" | has improved? {early_stopper.has_improved} "
                        f" | patience count {early_stopper.patience_count} |"
                    )
                # Log current patience count
                wandb.log(
                    {
                        "patience_counter": early_stopper.patience_count,
                    },
                    step=epoch
                )
                # Stop training if we have run out of patience
                if early_stopper.should_stop:
                    print(f'Met early stopping criteria, breaking at epoch {epoch}')
                    break

        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            # keep track of integration errors
            wandb.log(
                {
                    "integration_error": epoch,
                }
            )
            break

    # Compute RMSE val of the ensemble of best_models
    ensemble_rmse_val = rmse_ensemble(
        pp=problem_test,
        best_models=best_models,
        input_data=input_features_test,
    )
    # Performance printouts
    print(f"Best single model rmse_val: {best_models[0][0]}")  # rmse_val of first model = best performing one
    print(f"Best model ensemble rmse_val: {ensemble_rmse_val}")

    # W&B logs
    wandb.log({"final_epoch": epoch})
    wandb_stripped_dir = wandb.run.dir.rsplit('/files', 1)[0]
    command = f"wandb sync {wandb_stripped_dir}"
    # Plot model weights - proxy for model architecture -- disabled for now
    # TODO @GiacomoFabrini - fix this if we want to use this!
    # plot_model_weights(model, filename=Path(TRAINED_MODEL_WEIGHT_PLOTS.format(**conf)))
    # wandb.log({"trained_model_weights": wandb.Image(Path(TRAINED_MODEL_WEIGHT_PLOTS.format(**conf)))})
    # Save best models
    for ensemble_id, (_, ensemble_model_member) in enumerate(best_models):
        # Format ensemble_model_file and check parent exists
        ensemble_model_file = Path(model_file.format(ensemble_id=ensemble_id))
        ensemble_model_file.parent.mkdir(exist_ok=True, parents=True)
        # Serialise ensemble model member
        ensemble_model_member.save(
            ensemble_model_file,
            samples_name_list_dict,
        )
        # Log serialised ensemble member model
        wandb.log_model(path=ensemble_model_file, name=f"trained_dmm_{ensemble_id}")
    # Close and sync W&B run
    wandb.finish()
    try:
        _ = subprocess.run(command, shell=True)
    except subprocess.CalledProcessError as e:
        raise ValueError(f"Error syncing wandb directory: {e}")
    return best_models
