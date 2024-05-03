import equinox as eqx
# import itertools as itt
import jax
import jax.numpy as jnp
import numpy as np
import petab
import pypesto
import wandb

from .dmm_autoencoder_eqx import DeepMechanisticModel
from .wandb_init import log_model_stats
from .deepcomponent_eqx import DeepComponent
# CHECK WHETHER WE NEED TO ROLL BACK
from amici.petab.simulations import rdatas_to_simulation_df
# from amici.petab_objective import rdatas_to_simulation_df
from common import EarlyStoppingParams, get_scheduler, optimisers, RECON_LOSS, SYMM_LOSS, L1EREG, OEREG, L1IREG, OIREG
from flax.training.early_stopping import EarlyStopping
# doc: flax.readthedocs.io/en/latest/_modules/flax/training/early_stopping.html
from jaxtyping import Float, PyTree
from jaxtyping import Array
from pathlib import Path
# from .problem import Problem
from pypesto import Result
from pypesto.C import MODE_RES, RDATAS
from pypesto.objective.jax import JaxObjective
from pypesto.result.optimize import OptimizeResult, OptimizerResult
from pypesto.store import OptimizationResultHDF5Writer
from typing import (Dict, Union)


trace_path = Path(__file__).parents[1] / "traces"
TRACE_FILE_TEMPLATE = "{pathway}__{data}__{n_hidden}__{job}__{{id}}.csv"


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
        # jax_fun=ae.embedding,
        # x_names=ae.x_names
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
    # return pypesto.Problem(
    #     objective=generate_pypesto_objective(ae),
    #     x_names=ae.x_names,
    #     lb=[-np.inf for _ in ae.x_names],
    #     ub=[np.inf for _ in ae.x_names],
    # )


# TODO @GiacomoFabrini expand to get biases as well (and rename)
def get_weights(
        module: Union[DeepComponent, eqx.Module]
) -> jnp.ndarray:
    weights = jnp.concatenate(
        [
            module.layers[i].weight.flatten()
            for i in range(len(module.layers))
        ]
    )
    return weights


def map_params_to_array(
        model: DeepMechanisticModel
) -> jnp.ndarray:
    encoder_params = get_weights(model.deep_encoder)
    inflater_params = get_weights(model.deep_inflater)
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
        decoder_params = get_weights(model.deep_decoder)
        param_array = jnp.concatenate([param_array.flatten(), decoder_params.flatten()])
    param_array = jnp.concatenate([param_array, model.kin_params_combiner.learned_global_kin_params.flatten()])
    if len(param_array) != len(model.x_names):
        raise ValueError("Number of parameters does not match number of parameter names!")
    return param_array


@eqx.filter_value_and_grad
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
    fval = problem_train.objective(model(input_data)[0])
    loss_value = (
            fval
            + model.l1_encode_reg(scale=conf["l1reg_encode"])
            + model.orth_encode_reg(scale=conf["oreg_encode"])
            + model.l1_inflate_reg(scale=conf["l1reg_inflate"])
            + model.orth_inflate_reg(scale=conf["oreg_inflate"])
    )

    if model.reconstruct:
        loss_value += (
                model.reconstruction_loss(x=input_data, scale=conf["recon_loss"])
                + model.symmetry_loss(scale=conf["symm_reg"])
        )

    return loss_value


def train(
        model: DeepMechanisticModel,
        problem_train: pypesto.Problem,
        problem_test: pypesto.Problem,
        input_features_train,
        input_features_test,
        rfile: Path,
        conf: Dict,
        n_epoch,
        x0,  # PEtab-compatible embedding of initial parameters
        early_stopping_params: EarlyStoppingParams,
) -> pypesto.Result:

    """
    Trains the provided autoencoder by solving the optimization problem
    generated by :py:func:`create_pypesto_problem`
    """

    # Get schedule and initialise optimiser
    # TODO @GiacomoFabrini: add more complex scheduler
    schedule = get_scheduler(conf, n_epoch)
    opt = optimisers[conf["optimiser"]](schedule)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    # TODO @GiacomoFabrini Do we still need these? Or can we change them in some way?
    x = x0.copy()
    opt_x = x.copy()
    opt_fval = np.inf
    opt_grads = np.NaN * np.ones_like(x)
    rmse_test_min = np.inf

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

    @eqx.filter_jit
    def make_step(
            model: DeepMechanisticModel,
            opt_state: PyTree,
            input_data: Float[Array, '...'],  # TODO @GiacomoFabrini fix input data shape
            problem_train: pypesto.Problem,
            conf: Dict,
    ):
        loss_value, grads = loss_fn(
            model,
            conf,
            input_data,
            problem_train,
        )
        # p, s = eqx.partition(model, eqx.is_array)
        # loss_w_grads = jax.value_and_grad(type(m).loss, argnums=0)
        # f, grads = loss_w_grads(p, static=s, conf=conf, **kwargs)
        grads = jax.tree_map(
            lambda x: jnp.where(jnp.isfinite(x), x, jnp.zeros_like(x)),
            grads,
        )
        updates, opt_state = opt.update(grads, opt_state, model)
        # Update model in `next_model`, but keep current one in `model` for current epoch metric logging
        next_model = eqx.apply_updates(model, updates)
        return next_model, model, opt_state, loss_value, grads

    # Training loop
    for epoch in range(n_epoch + 1):
        next_model, model, opt_state, loss_train, grads = make_step(
            model=model,
            opt_state=opt_state,
            input_data=input_features_train,
            problem_train=problem_train,
            conf=conf,
        )

        # Get current fval for logging purposes
        fval = problem_train.objective(model(input_features_train)[0])

        # Log fval and loss_train at this epoch
        wandb.log(
            {
                "fval": fval,
                "loss": loss_train,
            },
            step=epoch
        )

        # Log regularisation terms that are used both with one and two heads
        for reg_fun, label in zip(
            (
                model.l1_encode_reg,
                model.orth_encode_reg,
                model.l1_inflate_reg,
                model.orth_inflate_reg,
            ),
            (L1EREG, OEREG, L1IREG, OIREG),
        ):
            if conf[label] > 0:
                wandb.log(
                    {
                        label: reg_fun(scale=conf[label])
                    },
                    step=epoch
                )

        # Log decoder-dependent loss terms
        if model.reconstruct:
            wandb.log(
                {
                    RECON_LOSS: model.reconstruction_loss(
                        x=input_features_train,
                        scale=conf[RECON_LOSS]
                    ),
                    SYMM_LOSS: model.symmetry_loss(scale=conf[SYMM_LOSS])
                },
                step=epoch
            )

        # Overwrite model with updated next_model
        model = next_model

        # Update x - same param array that we had before
        x = map_params_to_array(model)
        # Map grads to same shape param array
        grads_array = map_params_to_array(grads)

        # Log rmse values every 5 epochs + check early-stopping criteria
        if epoch % 5 == 0:
            rmse_dict = dict()
            # evaluate rmse on train and test dataset only after a certain number (5) of epochs
            for dataset, pp, input_data in zip(
                    ("train", "test"),
                    (problem_train, problem_test),
                    (input_features_train, input_features_test)
            ):
                rmse_dict[dataset] = rmse(pp, model, input_data)

            if rmse_dict["test"] < rmse_test_min:
                rmse_test_min = rmse_dict["test"]
                opt_x = x.copy()
                opt_fval = fval
                opt_grads = grads_array.copy()

            wandb.log(
                {
                    "rmse_train": rmse_dict["train"],
                    "rmse_val": rmse_dict["test"],
                    **log_model_stats(model, grads, pretrain=False)
                },
                step=epoch
            )

            if conf["use_early_stopping"]:
                # Update early stopper
                early_stopper = early_stopper.update(rmse_dict["test"])
                # Debugging statements
                print(
                    f"epoch {epoch} | "
                    f"rmse_val {rmse_dict['test']} | "
                    f"has improved? {early_stopper.has_improved} | "
                    f"patience count {early_stopper.patience_count}"
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

        if np.any(np.isnan(x)):
            # keep track of integration errors
            wandb.log(
                {
                    "integration_error": epoch,
                }
            )
            break

    wandb.log({"final_epoch": epoch})
    wandb.finish()

    # Saving epoch number inside n_fval (number of function evaluations)
    optimization_result = OptimizeResult()
    optimization_result.append(
        OptimizerResult(
            fval=opt_fval,
            # n_fval=epoch,  # save epoch number to diagnose early stopping
            x=opt_x,
            grad=opt_grads,
            x0=x0,
            id=str(conf["job"]),
        )
    )
    result = Result(
        problem=problem_train,
        optimize_result=optimization_result,
    )

    rfile.parent.mkdir(exist_ok=True, parents=True)
    writer = OptimizationResultHDF5Writer(str(rfile))
    writer.write(result, overwrite=True)

    return result


def rmse(pp,
         model: DeepMechanisticModel,
         input_data):
    try:
        x = model(input_data)[0]
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



