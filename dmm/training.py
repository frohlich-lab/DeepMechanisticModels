import equinox as eqx
import git
import itertools as itt
import jax
import jax.numpy as jnp
import numpy as np
import petab
import pypesto
import wandb

from .dmm_autoencoder_eqx import DeepMechanisticModel
from .deepcomponent_eqx import DeepComponent
# CHECK WHETHER WE NEED TO ROLL BACK
from amici.petab.simulations import rdatas_to_simulation_df
# from amici.petab_objective import rdatas_to_simulation_df
from common import Conf, EarlyStoppingParams
from flax.training.early_stopping import EarlyStopping
# doc: flax.readthedocs.io/en/latest/_modules/flax/training/early_stopping.html
from jax import value_and_grad
from jaxtyping import Array, Float, Int, PyTree
from jaxtyping import Array
from optax import adam, apply_updates, linear_schedule
from pathlib import Path
from .problem import Problem
from pypesto import Result
from pypesto.C import MODE_RES, RDATAS
from pypesto.objective.jax import JaxObjective
from pypesto.result.optimize import OptimizeResult, OptimizerResult
from pypesto.store import OptimizationResultHDF5Writer
from typing import Dict


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


def get_weights(
        module: DeepComponent
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
    kincombiner_params = model.kin_params_combiner.learned_median_params
    param_array = jnp.concatenate([
        module_params.flatten()
        for module_params in [encoder_params, inflater_params, kincombiner_params]
    ])
    if model.reconstruct:
        decoder_params = get_weights(model.deep_decoder)
        param_array = jnp.concatenate([param_array.flatten(), decoder_params.flatten()])
    param_array = jnp.concatenate([param_array, model.kin_params_combiner.learned_global_kin_params.flatten()])
    if len(param_array) != len(model.x_names):
        raise ValueError("Number of parameters does not match number of parameter names!")
    return param_array


# TODO @GiacomoFabrini: fetch scale params from conf
@eqx.filter_value_and_grad
def loss_fn(
        model: DeepMechanisticModel,
        conf: Conf,
        input_data,
        problem_train: pypesto.Problem,
):
    # problem_train.objective() now needs to get in input what was previously the output of the jax_fun, i.e. the output
    # of ae.embedding(x). x contained the parameters (encoder, inflater, kinetic parameters) and ae.embedding(x)
    # transformed the parameters into kinetic parameters (global) + inflated parameters (i.e. input data passed
    # through the encoder, through the inflater and unrolled). This is now the first component of the output of the model
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


L1EREG = "l1reg_encode"
OEREG = "oreg_encode"
L1IREG = "l1reg_inflate"
OIREG = "oreg_inflate"
RECON_LOSS = "recon_loss"
SYMM_LOSS = "symm_reg"


def train(
        model: DeepMechanisticModel,
        problem_train: pypesto.Problem,
        problem_test: pypesto.Problem,
        input_features_train,
        input_features_test,
        rfile: Path,
        conf: Conf,
        schedule_config: Dict,
        n_epoch,
        x0,  # PEtab-compatible mapping from deep_inflater/kin_params_combiner output
        early_stopping_params: EarlyStoppingParams,
) -> pypesto.Result:

    """
    Trains the provided autoencoder by solving the optimization problem
    generated by :py:func:`create_pypesto_problem`
    """

    repo = git.Repo(search_parent_directories=True)

    if (len(conf["encoder_layer_sizes"]) == 0) and (len(conf["inflater_layer_sizes"]) == 0):
        # default is "relu" but it is not applied unless there is at least 1 hidden layer
        activation_fn_tag = "None"
        linear_benchmark_tag = conf["linear_benchmark"]
    else:
        activation_fn_tag = conf["activation_fn_name"]
        # in these circumstances, linear_benchmark gets ignored
        linear_benchmark_tag = "overridden"

    wandb.init(
        project=f"DeepMechanisticModels.{conf['data']}.{conf['model']}",
        group=f"{conf['context']}_{conf['features']}",
        config={
            **conf,
            "use_early_stopping": early_stopping_params.use_early_stopping,  # early-stopping enabled/disabled
            "patience": early_stopping_params.patience
                if early_stopping_params.use_early_stopping else None,
            "min_improvement": early_stopping_params.min_improvement
                if early_stopping_params.use_early_stopping else None,
            "schedule_config": schedule_config,
            "optimizer": "adam",
            "scheduler": "linear",
            "reconstruct": conf["reconstruct"],
        },
        name="__".join(
            str(hyperparam_label)
            for hyperparam_label in (
                conf["samples"],
                conf["n_hidden"],
                conf["orth_reg_strategy"],
                activation_fn_tag,
                conf["reconstruct"],
                linear_benchmark_tag,
                conf["encoder_layer_sizes"],
                conf["encoder_layer_biases"],
                conf["inflater_layer_sizes"],
                conf["inflater_layer_biases"],
                conf["decoder_layer_biases"] if conf["reconstruct"] else None,
                conf["l1reg_encode"],
                conf["oreg_encode"],
                conf["l1reg_inflate"],
                conf["oreg_inflate"],
                conf["recon_loss"] if conf["reconstruct"] else None,
                conf["symm_reg"] if conf["reconstruct"] else None,
                conf["job"],
            )
        ),
        settings=wandb.Settings(
            start_method="fork",
            git_commit=repo.head.object.hexsha,
            git_remote_url=repo.remotes.origin.url,
        ),
        tags=[
            "deep_model",
            "early_stop" if early_stopping_params.use_early_stopping else "no_early_stop",
        ]
    )

    # Define W&B metrics in modular fashion
    metrics = {
        "rmse_train": "min",
        "rmse_val": "min",
        "patience_counter": None,
        "integration_error": None,
        "fval": "min",
        "loss": "min",
        L1EREG: "min",
        OEREG: "min",
        L1IREG: "min",
        OIREG: "min",
        RECON_LOSS: "min",
        SYMM_LOSS: "min",
    }
    for metric in metrics.keys():
        # if metric summary not specified
        if metrics[metric] is None:
            wandb.define_metric(metric)
        else:
            wandb.define_metric(metric, summary=metrics[metric])

    # TODO @GiacomoFabrini - this needs to be changed!
    # par_labels = ("encode", "inflate", "kinetic")
    # par_dims = (
    #     model.n_encode_weights,
    #     model.n_encoder_pars,
    #     model.n_kin_params,
    # )
    # for val_type, xname in itt.product(("x", "g"), par_labels):
    #     wandb.define_metric(f"{val_type}_{xname}")

    # This will be a PEtab-compatible format
    # x = map_params_to_array(model)

    # Optimiser and related schedule defined here
    # TODO @GiacomoFabrini: change optimiser to optax.adamw (weight decay) and change schedule (later?!)
    schedule = linear_schedule(**schedule_config)
    opt = adam(schedule)
    # TODO @GiacomoFabrini: need to replace with equinox-jax optimisation!
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    # opt_x = x.copy()
    # opt_fval = np.inf
    # opt_grads = np.NaN * np.ones_like(x)
    # rmse_test_min = np.inf

    # Check Early-stopping parameters have been set correctly and instantiate early stopper
    if early_stopping_params.use_early_stopping:
        if early_stopping_params.patience is None:
            raise ValueError("Patience value for early stopping is undefined.")
        elif early_stopping_params.min_improvement is None:
            raise ValueError("Minimum absolute improvement for early stopping is undefined.")
        else:
            early_stopper = EarlyStopping(
                min_delta=early_stopping_params.min_improvement,
                patience=early_stopping_params.patience
            )

    # TODO @GiacomoFabrini restore jitting once fixed
    @eqx.filter_jit
    def make_step(
            model: DeepMechanisticModel,
            opt_state: PyTree,
            input_data: Float[Array, '...'],  # TODO @GiacomoFabrini fix input data shape
            problem_train: pypesto.Problem,
            conf: Conf,
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
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value, grads

    # Training loop
    for epoch in range(n_epoch + 1):
        model, opt_state, loss_value, grads = make_step(
            model=model,
            opt_state=opt_state,
            input_data=input_features_train,
            problem_train=problem_train,
            conf=conf,
        )

        # TODO @GiacomoFabrini: replace once I find a fix
        fval = problem_train.objective(model(input_features_train)[0])

        # Update x
        x = model(input_features_train)[0]

        # TODO @GiacomoFabrini only compute function values and log (restore) -- OK
        # Log regularisation terms - not input dependent
        for reg_fun, label in zip(
            (
                model.l1_encode_reg,
                model.orth_encode_reg,
                model.l1_inflate_reg,
                model.orth_inflate_reg,
                model.symmetry_loss,
            ),
            (L1EREG, OEREG, L1IREG, OIREG, SYMM_LOSS),
        ):
            # this is where the scale parameter of the various regularisation
            # methods get changed via the hyperparameters in training_configuration.py
            if conf[label] > 0:
                # Simply compute the value of the function
                value_reg = reg_fun(scale=conf[label])
                wandb.log({label: value_reg}, step=epoch)
        # Log Reconstruction Loss (which requires input data)
        wandb.log(
            {
                RECON_LOSS: model.reconstruction_loss(
                    x=input_features_train,
                    scale=conf[RECON_LOSS]
                )
            }
        )
        # Log fval and loss function value at this epoch
        wandb.log(
            {
                "fval": fval,
                "loss": loss_value
            },
            step=epoch
        )

        # Log rmse values every 5 epochs + check early-stopping criteria
        if epoch % 5 == 0:
            rmse_dict = dict()
            # evaluate rmse on train and test dataset only after a certain number (5) of epochs
            for dataset, pp in zip(
                ("train", "test"), (problem_train, problem_test)
            ):
                # TODO @GiacomoFabrini I expect this will need input_features_test?!
                rmse_dict[dataset] = rmse(pp, x)

            if rmse_dict["test"] < rmse_test_min:
                rmse_test_min = rmse_dict["test"]
                # TODO @GiacomoFabrini this needs to be changed as well
                opt_x = x.copy()
                opt_fval = fval
                opt_grads = grads.copy()

            wandb.log(
                {
                    "rmse_train": rmse_dict["train"],
                    "rmse_val": rmse_dict["test"],
                    **{
                        f"{val_type}_{xname}": None
                        if not np.all(np.isfinite(value))
                        else wandb.Histogram(value)
                        if val_type == "x"
                        else wandb.Histogram(
                            np.log10(np.abs(value[value != 0]))
                        )
                        # TODO @GiacomoFabrini: need to fix this with proper grads and x - what shall I do with xname?
                        for val_type, values in (
                            ("x", x),
                            ("g", grads),
                        )
                        for xname, value in zip(
                            par_labels,
                            np.split(values, par_dims[:-1]),
                        )
                    },
                },
                step=epoch,
            )

            if early_stopping_params.use_early_stopping:
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
                    step=epoch,
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
            n_fval=epoch,  # save epoch number to diagnose early stopping
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


def rmse(pp, xx):
    try:
        x = pp.objective.jax_fun(xx)
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
