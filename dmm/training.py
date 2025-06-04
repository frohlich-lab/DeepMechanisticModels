import equinox as eqx
import numpy as np
import optax
import pypesto
# import subprocess
import wandb

# from common import TRAINED_MODEL_WEIGHT_PLOTS  # TODO - fix imports from common (not in use)
from .config_options import EarlyStoppingParams
from .dmm_autoencoder_eqx import DeepMechanisticModel
# from .training_helper_funcs import test_save_reload_model, plot_model_weights
from .wandb_init_log import log_extra_loss_terms, log_model_stats, log_param_norms
from .training_helper_funcs import (generate_log_epochs, get_finite_grads,
                                    get_optimiser_and_opt_state, map_params_to_array,
                                    model_output_to_petab_input, model_output_to_petab_input_frozen_medians,
                                    rmse, rmse_ensemble, MetricHandler)
from flax.training.early_stopping import EarlyStopping
# doc: flax.readthedocs.io/en/latest/_modules/flax/training/early_stopping.html
from jaxtyping import Array, Float, PyTree
from pathlib import Path
from typing import Callable, Dict


def update_best_models(
        model: DeepMechanisticModel,
        rmse_val: float,
        epoch: int,
        best_models: list,
        max_models: int,
        l1reg_scheduling: bool
) -> list[tuple[float, float, DeepMechanisticModel]]:
    """
    Update the list of best models with the current model and rmse_val
    """
    # Insert the new (rmse_val, model) in the appropriate position in the list
    best_models.append((epoch, rmse_val, model))
    # Prune entries from pre-sparsity stage, i.e. with epoch = 0
    if l1reg_scheduling:
        best_models = [(epoch, rmse_val, model) for (epoch, rmse_val, model) in best_models if epoch != 0]
    # Sort in ascending order by rmse_val (first = lowest = best)
    best_models = sorted(best_models, key=lambda x: x[1])
    # Keep only the top 'max_models' entries
    return best_models[:max_models]


@eqx.filter_jit
def jitted_objective(problem: pypesto.Problem, model: DeepMechanisticModel, data, base_obj_fn: Callable):
    return problem.objective(
        base_obj_fn(model, data)
    )

@eqx.filter_value_and_grad(has_aux=True)
def loss_fn(
        model: DeepMechanisticModel,
        conf: Dict,
        input_data,
        problem_train: pypesto.Problem,
        base_obj_fn: Callable,
        regularise_inflater_output: bool,
        median_init_arr: Array,
):
    # problem_train.objective() now needs to get in input what was previously the output of the jax_fun, i.e. the output
    # of ae.embedding(x). x contained the parameters (encoder, inflater, kinetic parameters) and ae.embedding(x)
    # transformed the parameters into kinetic parameters (global) + inflated parameters (i.e. input data passed
    # through encoder + inflater and flattened). This is now the first component of the output of the model.
    # call.
    fval = problem_train.objective(
        base_obj_fn(model, input_data)
    )

    loss_value = (
            fval
            + model.orth_encode_reg(scale=conf["oreg_encode"])
            + model.orth_inflate_reg(scale=conf["oreg_inflate"])
            + model.constrain_median(x=median_init_arr, scale=conf["median_reg"])
    )

    # Enable inflater output regularisation based on flag (epoch)
    if regularise_inflater_output:
        loss_value += model.l1reg_inflater_output(x=input_data, scale=conf["l1reg_inflater_output"])

    if model.reconstruct:
        loss_value += (
                model.reconstruction_loss(x=input_data, scale=conf["recon_loss"])
                + model.symmetry_loss(scale=conf["symm_reg"])
                + model.orth_decode_reg(scale=conf["oreg_encode"])
        )

    # Add L1 regularisation terms (encoder, inflater, optionally decoder)
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
        opt: optax.GradientTransformation,
        opt_state: PyTree,
        input_data: Float[Array, '...'],  # TODO @GiacomoFabrini fix input data shape?
        problem_train: pypesto.Problem,
        base_obj_fn: Callable,
        conf: Dict,
        regularise_inflater_output: bool,
        median_init_arr: Array,
):
    (loss_value, fval), grads = loss_fn(
        model,
        conf,
        input_data,
        problem_train,
        base_obj_fn,
        regularise_inflater_output,
        median_init_arr,
    )
    grads = get_finite_grads(grads)
    updates, opt_state = opt.update(grads, opt_state, model)
    # Update model in `next_model`, but keep current one in `model` for current epoch metric logging
    next_model = eqx.apply_updates(model, updates)
    return next_model, model, opt_state, loss_value, fval, grads


def train(
        model: DeepMechanisticModel,
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
) -> list[tuple[float, float, DeepMechanisticModel]]:
    """
    Trains the provided autoencoder by solving the optimization problem
    generated by :py:func:`create_pypesto_problem`
    """

    # Initialise optimiser and its state
    opt, opt_state = get_optimiser_and_opt_state(
        conf=conf, n_epoch=n_epoch, model=model
    )

    # Initialise default values for early_stopper, epoch and metric handler (invalid fval/RMSE metrics)
    early_stopper = None
    epoch = 0
    metric_handler = MetricHandler()

    # Setup base obj_fn
    base_obj_fn = model_output_to_petab_input_frozen_medians if conf["freeze_medians"] else model_output_to_petab_input

    # Use randomly initialised model to get initial rmse_test_min and
    # the collection of best_models for the ensemble. Returns np.inf is something fails.
    # rmse_train_start = rmse(problem_train, model, input_features_train)
    rmse_test_min = rmse(problem_test, model, input_features_test)
    wandb.log(
        {
            "start_rmse_val": rmse_test_min,
        },
        step=0
    )

    best_models = [
        (epoch, rmse_test_min, model)  # each item comprises the epoch, the RMSE validation score and the model itself
        for _ in range(ensemble_members)
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

    # Get median_init_arr for median regularisation
    median_init_arr = model.kin_params_combiner.learned_global_kin_params  # prior to any training (epoch 0)

    # Training loop
    for epoch in range(1, n_epoch + 1):  # natural counting
        # At inflater_output_reg_epoch, update sparsity mask in the model inflated output and lift inflater output
        # regularisation to fine-tune only above threshold param dev
        if epoch == conf["inflater_output_reg_epoch"]:
            model = model.update_output_sparsity_binary_mask(
                x=input_features_train,
                threshold_perc=conf["sparse_threshold_perc"]
            )
            # TODO - fix this update draft, depends on inputs needed to update input binary mask
            model = model.update_input_sparsity_binary_mask(
                x=input_features_train,
            )
        next_model, model, opt_state, loss_train, fval, grads = make_step(
            model=model,
            opt=opt,
            opt_state=opt_state,
            input_data=input_features_train,
            problem_train=problem_train,
            base_obj_fn=base_obj_fn,
            conf=conf,
            regularise_inflater_output=epoch < conf["inflater_output_reg_epoch"],  # changed behaviour -- after epoch is reached, regularisation behaviour changes
            median_init_arr=median_init_arr,
        )

        # Log loss_train
        wandb.log(
            {
                "loss": loss_train,
            },
            step=epoch
        )

        # Log extra terms (regularisation)
        log_extra_loss_terms(
            model=model,
            conf=conf,
            input_data=input_features_train,  # use training features for RECON_LOSS
            epoch=epoch,
            median_init_arr=median_init_arr,
        )

        # Log norms (max absolute value + 2-norm) of parameter deviations and medians
        log_param_norms(
            model=model,
            input_data=input_features_train,
            epoch=epoch,
        )

        # Overwrite model with updated next_model
        model = next_model

        # Get evaluation model (simply model without schedule-free)
        eval_model = model

        # Update x - same param array that we had before
        x = map_params_to_array(model)

        # Log RMSE values + check early-stopping criteria + check for invalid metrics
        if epoch in log_epochs:

            # rmse_dict = dict()
            # for dataset, pp, input_data in zip(
            #         ("train", "test"),
            #         (problem_train, problem_test),
            #         (input_features_train, input_features_test)
            # ):
            #     rmse_dict[dataset] = rmse(pp, eval_model, input_data)

            # Compute fval on train/val datasets using eval_model
            fval_train, fval_val = (
                jitted_objective(problem, eval_model, input_data, base_obj_fn)
                for problem, input_data in zip(
                [problem_train, problem_test], [input_features_train, input_features_test]
            )
            )

            # Handle invalid loss_train (fval_train) and RMSE
            should_break = metric_handler.handle_invalid_metrics(
                metrics=[loss_train, fval_train, fval_val],
                epoch=epoch,
            )
            if should_break:
                break

            # Update tally of best models on validation score (ensemble members) if either no l1 regularisation is
            # being applied (hence no scheduling) or after lifting l1 regularisation and imposing sparsity

            if (
                    (conf["l1reg_inflater_output"] == 0) or (int(conf["sparse_threshold_perc"]) == 100) or
                    (conf["l1reg_inflater_output"] > 0 and epoch >= conf["inflater_output_reg_epoch"])
            ):
                best_models = update_best_models(
                    model=eval_model,
                    rmse_val=np.sqrt(fval_val),
                    epoch=epoch,
                    best_models=best_models,
                    max_models=ensemble_members,
                    l1reg_scheduling=bool(
                        conf["l1reg_inflater_output"] and (not int(conf["sparse_threshold_perc"]) == 100)
                    ),
                )

            # Log RMSE, fval (both train/val) and model stats
            wandb.log(
                {
                    "rmse_train": np.sqrt(fval_train),
                    "rmse_val": np.sqrt(fval_val),
                    "fval_train": fval_train,
                    "fval_val": fval_val,
                    **log_model_stats(eval_model, grads)
                },
                step=epoch
            )

            # Progress/debugging statements
            if debug_mode:
                print(
                    f" | epoch {epoch} "
                    f" | rmse_train {np.sqrt(fval_train)}"
                    f" | rmse_val {np.sqrt(fval_val)} "
                    f" | fval_train {fval_train} "
                    f" | fval_val {fval_val} | "
                )

            if conf["use_early_stopping"]:
                # Update early stopper
                early_stopper = early_stopper.update(np.sqrt(fval_val))
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

    # Compute RMSE val of the ensemble of best_models -- returns NaN if something fails
    ensemble_rmse_val = rmse_ensemble(
        pp=problem_test,
        best_models=best_models,
        input_data=input_features_test,
    )
    # Performance printouts
    print(f"Best single model rmse_val: {best_models[0][1]} at epoch {best_models[0][0]}")
    print(f"Best model ensemble rmse_val: {ensemble_rmse_val}")

    # W&B logs
    rmse_val_final = rmse(problem_test, model, input_features_test)
    wandb.log(
        {
            "final_rmse_val": rmse_val_final,
            "final_epoch": epoch,
        },
        step=epoch
    )
    # wandb_stripped_dir = wandb.run.dir.rsplit('/files', 1)[0]
    # command = f"wandb sync {wandb_stripped_dir}"

    # TODO @GiacomoFabrini - fix this if we want to use this!
    # Plot model weights - proxy for model architecture -- disabled for now
    # plot_model_weights(model, filename=Path(TRAINED_MODEL_WEIGHT_PLOTS.format(**conf)))
    # wandb.log({"trained_model_weights": wandb.Image(Path(TRAINED_MODEL_WEIGHT_PLOTS.format(**conf)))})

    # Save best models
    for ensemble_id, (_, _, ensemble_model_member) in enumerate(best_models):
        # Format ensemble_model_file and check parent exists
        ensemble_model_file = Path(model_file.format(ensemble_id=ensemble_id))
        ensemble_model_file.parent.mkdir(exist_ok=True, parents=True)
        # Serialise ensemble model member
        ensemble_model_member.save(
            ensemble_model_file,
            samples_name_list_dict,
        )
        # Log serialised ensemble member model -- temporarily disabled
        # wandb.log_model(path=ensemble_model_file, name=f"trained_dmm_{ensemble_id}")

    # Close and sync W&B run (online)
    wandb.finish()
    # try:
    #     _ = subprocess.run(command, shell=True)
    # except subprocess.CalledProcessError as e:
    #     raise ValueError(f"Error syncing wandb directory: {e}")
    return best_models
