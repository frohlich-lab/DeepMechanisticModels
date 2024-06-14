import equinox as eqx
import git
import numpy as np
import wandb

from common import Conf, EarlyStoppingParams, L1EREG, OEREG, L1IREG, OIREG, RECON_LOSS, SYMM_LOSS
from dmm.custom_layers_eqx import CustomInitLinear
from dmm.dmm_autoencoder_eqx import DeepMechanisticModel


def init_wandb(
        model: DeepMechanisticModel,
        conf: Conf,
        early_stopping_params: EarlyStoppingParams,
        pretrain: bool,
):
    """
    Initialise W&B run. Run name = chosen hyperparameters in configuration string representation.
    """
    repo = git.Repo(search_parent_directories=True)

    # TODO @GiacomoFabrini - check that this works! If not, reinstantiate the process_model_layers call
    if (len(model.deep_encoder.layers) == 1) and (len(model.deep_inflater.layers) == 1):
        no_hidden_layers = True
    else:
        no_hidden_layers = False

    # default is "relu" but it is not applied unless there is at least 1 hidden layer in a given model module
    activation_fn_tag = "None" if no_hidden_layers else conf.activation_fn_name

    if pretrain:
        group = f"{conf.context}_{conf.features}_network_pretrain"  # distinguish from whole DMM training
    else:
        group = f"{conf.context}_{conf.features}"

    # Instantiate a new Conf object with layer sizes processed as lists (easier to read in W&B)
    config_conf = conf
    config_conf.encoder_layer_sizes = model.deep_encoder.layer_sizes
    config_conf.inflater_layer_sizes = model.deep_inflater.layer_sizes

    wandb.init(
        project=f"DeepMechanisticModels.v2.{conf.data}.{conf.model}",  # v2 = Equinox port
        group=group,
        config={
            **config_conf.__dict__,
            "use_early_stopping": conf.use_early_stopping,  # early-stopping enabled/disabled
            "patience": early_stopping_params.patience if conf.use_early_stopping else None,
            "min_improvement": early_stopping_params.min_improvement if conf.use_early_stopping else None,
            "scheduler": "linear" if conf.use_simple_linear_schedule else "custom",
            # Add clearer info on depth of encoder and inflater modules (to use in parallel coordinates)
            "encoder_depth": len(model.deep_encoder.layers),
            "inflater_depth": len(model.deep_encoder.layers),
        },
        name=config_conf.__str__(replace={"activation_fn_name": activation_fn_tag}),
        settings=wandb.Settings(
            start_method="fork",
            git_commit=repo.head.object.hexsha,
            git_remote_url=repo.remotes.origin.url,
        ),
        tags=[
            "shallow_model" if no_hidden_layers else "deep_model",
            "early_stop" if conf.use_early_stopping else "no_early_stop",
            "linear_benchmark" if (conf.linear_benchmark and no_hidden_layers) else "not_benchmark",
            "network_pretraining" if pretrain else "DMM_training",
            "sparse_no_regularisation" if (~pretrain and conf.drop_reg_after_pretrain) else "full_regularisation",
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
    }
    # If in pretraining, keep regularisation terms
    # Same if in full model training with drop_reg_after_pretrain=False
    if pretrain or not conf.drop_reg_after_pretrain:
        reg_metrics = {
            L1EREG: "min",
            OEREG: "min",
            L1IREG: "min",
            OIREG: "min",
            RECON_LOSS: "min",
            SYMM_LOSS: "min",
        }
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
        pretrain: bool,  # needed?
):
    """
    Log parameter values (vals) and grads (grads) to wandb.

    :param model: DeepMechanisticModel containing parameter values
    :param grad: DeepMechanisticModel containing parameter gradients
    :param pretrain: boolean flag indicating whether this is a pretraining run (model.kin_params_combiner frozen)
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
    stats = {**layer_stats}

    # TODO @GiacomoFabrini: discuss with Fabian - does it make sense to log a histogram if
    #  these are individual values? Other option (currently selected): log them altogether as a single histogram
    if not pretrain:
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
