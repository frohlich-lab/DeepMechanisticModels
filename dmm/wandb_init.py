import git
import wandb

from common import Conf, EarlyStoppingParams, L1EREG, OEREG, L1IREG, OIREG, RECON_LOSS, SYMM_LOSS
from typing import Dict


def init_wandb(
        conf: Conf,
        early_stopping_params: EarlyStoppingParams,
        schedule_config: Dict,
        pretrain: bool,
):
    """
    Initialise W&B run. Run name = chosen hyperparameters in configuration string representation.
    """
    repo = git.Repo(search_parent_directories=True)

    if (len(conf.encoder_layer_sizes) == 0) and (len(conf.inflater_layer_sizes) == 0):
        # default is "relu" but it is not applied unless there is at least 1 hidden layer
        activation_fn_tag = "None"
        linear_benchmark_tag = conf.linear_benchmark
    else:
        activation_fn_tag = conf.activation_fn_name
        # in these circumstances, linear_benchmark gets ignored
        linear_benchmark_tag = "overridden"

    if pretrain:
        group = f"{conf.context}_{conf.features}_network_pretrain"  # distinguish from whole DMM training
    else:
        group = f"{conf.context}_{conf.features}"

    wandb.init(
        project=f"DeepMechanisticModels.{conf.data}.{conf.model}",
        group=group,
        config={
            **conf.__dict__,
            "use_early_stopping": early_stopping_params.use_early_stopping,  # early-stopping enabled/disabled
            "patience": early_stopping_params.patience
            if early_stopping_params.use_early_stopping else None,
            "min_improvement": early_stopping_params.min_improvement
            if early_stopping_params.use_early_stopping else None,
            "schedule_config": schedule_config,
            "scheduler": "linear",
        },
        name=conf.__str__(),
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

    # TODO @GiacomoFabrini - this needs to be changed! Implement pretrain checking (no values and grads for
    #  kinetic parameters)
    # par_labels = ("encode", "inflate", "kinetic")
    # par_dims = (
    #     model.n_encode_weights,
    #     model.n_encoder_pars,
    #     model.n_kin_params,
    # )
    # for val_type, xname in itt.product(("x", "g"), par_labels):
    #     wandb.define_metric(f"{val_type}_{xname}")
