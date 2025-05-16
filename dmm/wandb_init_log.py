import equinox as eqx
import git
import jax.numpy as jnp
import numpy as np
import wandb

from .config_options import (Conf, EarlyStoppingParams,
                             L1EREG, OEREG, L1DREG, ODREG, L1IREG, OIREG, L1REG_IO, RECON_LOSS, SYMM_LOSS, MEDIAN_REG)
from .custom_layers_eqx import CustomInitLinear
from .dmm_autoencoder_eqx import DeepMechanisticModel
from functools import partial
from jax import vmap
from typing import Optional


def init_wandb(
        model: DeepMechanisticModel,
        conf: Conf,
        early_stopping_params: EarlyStoppingParams,
):
    """
    Initialise W&B run. Run name = chosen hyperparameters in configuration string representation.
    """
    repo = git.Repo(search_parent_directories=True)

    # default is "relu" but it is not applied unless there is at least 1 hidden layer in a given model module
    activation_fn_tag = "None" if conf.depth == 0 else conf.activation_fn_name
    group = f"{conf.context}_{conf.features}"

    # Init wandb (default: new 'core' backend)
    wandb.init(
        # v2: Equinox
        # v3: Equinox, back to basics -- no decoder, simple decay learning rate schedule, first local attempts
        # v4: Equinox, basics - LinearScans
        # v5: Equinox, no network pretraining + leave-one-out cross-validation - Linear Scans
        # v6: same as v5, reduced, run locally due to wandb issues with cluster
        # v7: exploring various regularisation strategies, cluster, old wandb backend
        # v8: various tests on l1reg_inflater_output and epoch + median_reg
        # v11: l1reg scheduling
        # v12: MOSA 200 starts, no l1reg scheduling
        # v13: l1reg scheduling with fixed inflater_output_reg_epoch (500, half range), scanning sparsity percentage
        # v14: l1reg scheduling as above, but fixed best_models behaviour + updated sparse_threshold_perc behaviour
        # v15: l1reg scheduling, scanning optimal value for l1reg_inflater_output on CV 1of5
        # v16: updated feature selection (uniform across CV splits), regressors with feature selection, unregularised
        # v17: new/old mechanistic model, no reweighing, fixed Chi2 (MSE), no biases on last inflater layer (deviations)
        # v18: fixed mechanistic model, removed pretraining and relevant code, removed schedule-free optimisers
        # v19: same as v18 but with frozen kinetic param median
        # v20: updated metric to impose sparsity (median -> standard deviation); updated learning rate schedule
        # v21: force selection of pERBB2 features in cytof_init + n_hidde=3 for all contexts
        # v22: new CV split (cHCC2185 -> cUACC3199), new feature selection (per_cv vs across_cv)
        project=f"DeepMechanisticModels.v22.{conf.data}.{conf.model}",
        group=group,
        config={
            **conf.__dict__,
            "use_early_stopping": conf.use_early_stopping,  # early-stopping enabled/disabled
            "patience": early_stopping_params.patience if conf.use_early_stopping else None,
            "min_improvement": early_stopping_params.min_improvement if conf.use_early_stopping else None,
            "scheduler": "linear" if conf.use_simple_linear_schedule else "custom",
        },
        name=conf.__str__(replace={"activation_fn_name": activation_fn_tag}),
        settings=wandb.Settings(
            start_method="fork",
            git_commit=repo.head.object.hexsha,
            git_remote_url=repo.remotes.origin.url,
        ),
        tags=[
            "shallow_model" if conf.depth == 0 else "deep_model",
            "early_stop" if conf.use_early_stopping else "no_early_stop",
            "linear_benchmark" if (conf.linear_benchmark and conf.depth == 0) else "not_benchmark",
            conf.run_mode_tag,  # label run type (linear scans, grid search, refinement/tuning of best runs
            conf.date_tag  # label experiment with date of experiment start
        ],
        mode="online"  # to run more jobs simultaneously on the cluster
    )

    # Define W&B metrics
    metrics = {
        "loss": "min",
        "fval_train": "min",
        "fval_val": "min",
        "integration_error": None,
        "max_abs_par_dev": "min",  # max absolute value of parameter deviation
        "par_dev_frob_norm": "min",  # 2-norm of parameter deviation
        "max_abs_par_median": "min",  # max absolute value of parameter median
        "par_median_frob_norm": "min",  # 2-norm of parameter median
    }
    # common metrics - orthogonal regularisation + patience_counter
    for metric in ["rmse_train", "rmse_val", OEREG, OIREG]:
        metrics[metric] = "min"
    metrics["patience_counter"] = None
    metrics["start_rmse_val"] = None
    metrics["final_rmse_val"] = None
    # optional metrics depending on the presence of decoder head
    if model.reconstruct:
        metrics[RECON_LOSS] = "min"
        metrics[SYMM_LOSS] = "min"
        metrics[ODREG] = "min"

    reg_metrics = {
        L1EREG: "min",
        L1IREG: "min",
        L1REG_IO: "min",
        MEDIAN_REG: "min",
    }
    # Add decoder regularisation terms if the model has a decoder head
    if model.reconstruct:
        reg_metrics[L1DREG] = "min"

    # Get final metrics
    metrics = {**metrics, **reg_metrics}

    for metric in metrics.keys():
        # if metric summary not specified
        if metrics[metric] is None:
            wandb.define_metric(metric)
        else:
            wandb.define_metric(metric, summary=metrics[metric])

    model_modules = {
        "encoder": model.deep_encoder,
        "inflater": model.deep_inflater,
    }
    if model.reconstruct:
        model_modules["decoder"] = model.deep_decoder

    # Iterate over the modules to define metrics based on the presence of layers and biases
    for module, model_module in model_modules.items():
        num_layers = len(model_module.layers)  # Dynamically get the number of layers
        for layer_index in range(num_layers):
            # Check for weights; assume weights always exist
            for val_type in ['vals', 'grads']:
                metric_name = f"{module}.layer{layer_index}.w_{val_type}"
                wandb.define_metric(metric_name)

            # Check if bias exists and is not None before defining bias metrics
            if hasattr(model_module.layers[layer_index], 'bias') and model_module.layers[layer_index].bias is not None:
                for val_type in ['vals', 'grads']:
                    metric_name = f"{module}.layer{layer_index}.b_{val_type}"
                    wandb.define_metric(metric_name)


def log_model_stats(
        model: DeepMechanisticModel,
        grad: DeepMechanisticModel,
):
    """
    Log parameter values (vals) and grads (grads) to wandb.

    :param model: DeepMechanisticModel containing parameter values
    :param grad: DeepMechanisticModel containing parameter gradients
    """

    model_modules = {
        "encoder": model.deep_encoder,
        "inflater": model.deep_inflater,
    }
    grad_modules = {"encoder": grad.deep_encoder,
        "inflater": grad.deep_inflater,
    }
    if model.reconstruct:
        model_modules["decoder"] = model.deep_decoder
        grad_modules["decoder"] = grad.deep_decoder

    layers = {
        key: [
            (par_layer, grad_layer)
            for par_layer, grad_layer in zip(
                model_module.layers, grad_modules[key].layers,
            )
            if isinstance(par_layer, (eqx.nn.Linear, CustomInitLinear))  # Check for both Linear and CustomInitLinear
        ]
        for key, model_module in model_modules.items()
    }

    layer_stats = {}
    for module, layer_list in layers.items():
        for ilayer, (par_layer, grad_layer) in enumerate(layer_list):
            # Handle weights
            weight_vals = par_layer.weight.ravel()
            grad_weight_vals = grad_layer.weight.ravel()
            layer_stats[f'{module}.layer{ilayer}.w_vals'] = wandb.Histogram(list(weight_vals))
            layer_stats[f'{module}.layer{ilayer}.w_grads'] = wandb.Histogram(
                np.log10(np.abs(np.array(grad_weight_vals[grad_weight_vals != 0])))
            )

            # Handle biases, if they exist
            if hasattr(par_layer, 'bias') and par_layer.bias is not None:
                bias_vals = par_layer.bias.ravel()
                grad_bias_vals = grad_layer.bias.ravel()
                layer_stats[f'{module}.layer{ilayer}.b_vals'] = wandb.Histogram(list(bias_vals))
                layer_stats[f'{module}.layer{ilayer}.b_grads'] = wandb.Histogram(
                    np.log10(np.abs(np.array(grad_bias_vals[grad_bias_vals != 0])))
                )

    # First approach: two plots (value, grad) per parameter, but hists do not make much sense in that case
    # kin_params_stats = {
    #     f'global_kin_param.{par_label}_{value_label}': wandb.Histogram(
    #         np.log10(np.abs(np.array(par_val[par_val != 0])))
    #     )
    #     if value_label == 'grads'
    #     else wandb.Histogram(par_val)
    #     for value_label, par_vals in zip(
    #         ('vals', 'grads'),
    #         (
    #             model.kin_params_combiner.learned_global_kin_params,
    #             grad.kin_params_combiner.learned_global_kin_params
    #         ),
    #     )
    #     for par_label, par_val in zip(
    #         range(len(model.kin_params_combiner.learned_global_kin_params)),
    #         np.array(par_vals).ravel(),
    #     )
    # }

    # Second approach: log values and grads altogether (2 histograms overall)
    kin_params_stats = {
        f'global_kin_params.{value_label}': wandb.Histogram(
            np.log10(np.abs(np.array(par_vals[par_vals != 0])))
        )
        if value_label == 'grads'
        else wandb.Histogram(par_vals)
        for value_label, par_vals in zip(
            ('vals', 'grads'),
            (
                np.array(model.kin_params_combiner.learned_global_kin_params).ravel(),
                np.array(grad.kin_params_combiner.learned_global_kin_params).ravel()
            ),
        )
    }
    # Augment stats with global kinetic parameters
    stats = {**layer_stats, **kin_params_stats}

    return stats


def log_param_norms(
        model: DeepMechanisticModel,
        input_data: jnp.ndarray,
        epoch: int,
):
    par_dev = vmap(model)(input_data)["inflated"]
    par_medians = model.kin_params_combiner.learned_global_kin_params
    wandb.log(
        {
            "max_abs_par_dev": jnp.max(jnp.abs(par_dev)),
            "par_dev_frob_norm": jnp.linalg.norm(x=par_dev, ord=None),
            "max_abs_par_median": jnp.max(jnp.abs(par_medians)),
            "par_median_frob_norm": jnp.linalg.norm(x=par_medians, ord=None),
        },
        step=epoch,
    )


def log_extra_loss_terms(
        model: DeepMechanisticModel,
        conf: dict,
        input_data: jnp.ndarray,
        epoch: int,
        median_init_arr: Optional[jnp.ndarray] = None,
):
    """
    Function to log extra loss terms (not fval nor loss itself) to W&B: regularisation terms, reconstruction loss.

    :param model:
        DeepMechanisticModel instance.
    :param conf:
        configuration dictionary.
    :param input_data:
        data to compute reconstruction loss on.
    :param epoch:
        training iteration/epoch.
    :param median_init_arr:
        array of median initialisation values (Optional, only DMM training).

    :return:
        n/a (simply logs to W&B).
    """
    # Define regularisation functions and labels
    reg_funs = [
        model.orth_encode_reg,
        model.orth_inflate_reg,
    ]
    log_labels = [OEREG, OIREG]
    hp_names = [OEREG, OIREG]
    # Add extra regularisation terms
    reg_funs.extend(
        [
            model.l1_encode_reg,
            model.l1_inflate_reg,
        ]
    )
    log_labels.extend([L1EREG, L1IREG, MEDIAN_REG])
    hp_names.extend([L1EREG, L1IREG, MEDIAN_REG])
    # Add epoch-dependent inflater output regularisation
    if epoch < conf["inflater_output_reg_epoch"]:
        reg_funs.extend([partial(model.l1reg_inflater_output, x=input_data),])
        log_labels.extend([L1REG_IO,])
        hp_names.extend([L1REG_IO, ])

    if median_init_arr is not None:
        reg_funs.append(
            partial(model.constrain_median, x=median_init_arr)
        )
        log_labels.append(MEDIAN_REG)
        hp_names.append(MEDIAN_REG)

    # Add extra terms if the DMM has a decoder head
    if model.reconstruct:
        reg_funs.extend([model.orth_decode_reg, model.l1_decode_reg])
        log_labels.extend([ODREG, L1DREG])
        hp_names.extend([OEREG, L1EREG])  # scales of decoder reg match encoder!
        # and log additional decoder-related loss terms (reconstruction and symmetry loss)
        wandb.log(
            {
                # for reconstruction loss: log validation loss
                RECON_LOSS: model.reconstruction_loss(
                    x=input_data,
                    scale=conf[RECON_LOSS],
                ),
                SYMM_LOSS: model.symmetry_loss(scale=conf[SYMM_LOSS])
            },
            step=epoch,
        )

    # Log metrics defined above
    for (reg_fun, log_label, hp_name) in zip(reg_funs, log_labels, hp_names):
        if conf[hp_name] > 0:
            # Simply compute the value of the function
            value_reg = reg_fun(scale=conf[hp_name])
            wandb.log({log_label: value_reg}, step=epoch)
