import itertools as itt
from typing import Union

import numpy as np

from dmm.config_options import scan_attributes
from training_configuration import (
    ACTIVATION_FNS,
    ALPHAS,
    BETAS,
    DELTAS,
    DROPOUT_RATES,
    EPSILONS,
    ETAS,
    FREEZE_MEDIANS,
    GAMMAS,
    INFLATER_BOUND,
    # Regularisation-adjacent
    INFLATER_OUTPUT_REG_EPOCHS,
    LATENT_DIMS,
    LEARNING_RATE_DECAYS,
    LEARNING_RATE_SPANS,
    MAX_LEARNING_RATES,
    MOMENTUM,
    MULTIHEADED,
    NEPOCH,
    NETWORK_DEPTH,
    NN_INIT_FN,
    NN_INIT_SCALES,
    OMEGAS,
    OPT_MULT,
    OPT_STEPS,
    OPTIMISERS,
    # Regularisation
    ORTH_REG_STRATEGIES,
    SPARSE_THRESH_PERCS,
    SPLITS,
    STANDARDISE_FEATURES,
    SYNC_ENCODER_INFLATER_REG,
    THETAS,
    USE_BIAS,
    # OTHER OPTIONS
    USE_EARLY_STOP,
    WARMUP_FCTS,
    WEIGHT_DECAY,
    ZETAS,
)


def format_floats(
    run_configs: list[dict],
    cols_to_check: list,
    precision: Union[int, str] = "adaptive",
):
    def format_string(precision):
        return "{:." + str(precision) + "f}"

    for config in run_configs:
        for col in cols_to_check:
            if isinstance(precision, int):
                format_prefix = format_string(precision)
                config[col] = format_prefix.format(config[col])
            elif precision == "adaptive":
                if config[col] == 0:
                    config[col] = "0.0"
                else:
                    oom = int(np.log10(config[col]))
                    if oom < 0:
                        oom = abs(oom)
                        format_prefix = format_string(oom)
                        config[col] = format_prefix.format(config[col])
                    else:
                        config[col] = str(
                            config[col]
                        )  # regular values, e.g. 1, 10, 100
    return run_configs


def prune_config(run_config: dict):
    prune = False
    hps_to_prune = []

    # Learning-rate scheduling
    if (
        "use_simple_linear_schedule" in run_config
        and run_config["use_simple_linear_schedule"]
    ):  # only with adam or adamw
        hps_to_prune.extend(
            ["opt_steps", "opt_mult", "momentum"]
        )  # use default momentum value
        if run_config["optimiser"] == "adam":
            hps_to_prune.append(
                "weight_decay"
            )  # no weight decay for regular Adam, but keep it for AdamW
        prune = True

    # If warm-up is applied, override it to end at the epoch at which sparsity is imposed
    if "warmup_fct" in run_config and run_config["warmup_fct"] > 0:
        run_config["warmup_fct"] = (
            run_config["inflater_output_reg_epoch"] / run_config["n_epoch"]
        )

    if prune:
        for hp in hps_to_prune:
            run_config[hp] = 0

    return run_config


linear_hyperparameters = {
    "n_hidden": LATENT_DIMS,
    "depth": NETWORK_DEPTH,
    "dropout_rate": DROPOUT_RATES,
    "nn_init_scale": NN_INIT_SCALES,
    "l1reg_inflate": ALPHAS,
    "oreg_inflate": BETAS,
    "l1reg_encode": GAMMAS,
    "oreg_encode": DELTAS,
    "l1reg_inflater_output": OMEGAS,
    "l2reg_inflater_output": THETAS,  # same as l1reg_inflater_output
    "inflater_output_reg_epoch": INFLATER_OUTPUT_REG_EPOCHS,
    "sparse_threshold_perc": SPARSE_THRESH_PERCS,
    "recon_loss": EPSILONS,
    "symm_reg": ZETAS,
    "median_reg": ETAS,
    "max_lrate": MAX_LEARNING_RATES,
    "lrate_span": LEARNING_RATE_SPANS,
    "lrate_decay": LEARNING_RATE_DECAYS,
    "warmup_fct": WARMUP_FCTS,
    "opt_steps": OPT_STEPS,
    "opt_mult": OPT_MULT,
    "weight_decay": WEIGHT_DECAY,
    "momentum": MOMENTUM,
    "n_epoch": NEPOCH,
    "inflater_bound": INFLATER_BOUND,
}
product_hyperparameters = {
    "activation_fn_name": ACTIVATION_FNS,
    "optimiser": OPTIMISERS,
    "orth_reg_strategy": ORTH_REG_STRATEGIES,
    "use_layer_bias": USE_BIAS,
    "nn_init_fn": NN_INIT_FN,
    "multiheaded": MULTIHEADED,
    "standardise_features": STANDARDISE_FEATURES,
    "sync_encoder_inflater_reg": SYNC_ENCODER_INFLATER_REG,
    "freeze_medians": FREEZE_MEDIANS,
    "use_early_stopping": USE_EARLY_STOP,
    "samples": SPLITS,
}


def generate_linear_scan(
    contexts_features: list[tuple],
    starts: list[str],
    select_central_values: bool,
    params_to_scan: list = None,
):
    # Check that all hyperparameter options are dicts (central value, range)
    if not all(
        isinstance(hyperparam, dict)
        for hyperparam in linear_hyperparameters.values()
    ):
        raise TypeError("Inconsistent typing for linear scans!")

    # Get central values
    central_values = {
        hyperparam: linear_hyperparameters[hyperparam]["central_value"]
        for hyperparam in scan_attributes
        if hyperparam in linear_hyperparameters
    }

    if select_central_values:
        linear_scan_configs = [prune_config(central_values)]
    else:
        scan_params = (
            params_to_scan if params_to_scan is not None else scan_attributes
        )
        linear_scan_configs = [
            prune_config({**central_values, **{param: value}})
            for param in scan_params
            if param in linear_hyperparameters
            for value in linear_hyperparameters[param]["range"]
            if linear_hyperparameters[param]["central_value"] != value
        ] + [prune_config(central_values)]

    # product expand for starts, contexts, and features
    linear_scan_configs = [
        {**config, **{"context": context, "features": features, "job": start}}
        for config in linear_scan_configs
        for start, (context, features) in itt.product(
            starts, contexts_features
        )
    ]

    # TODO: fix this, this does not work! (linear scans ok, cartesian product broken)
    # product expand for product hyperparameters
    for param in scan_attributes:
        if param in product_hyperparameters:
            linear_scan_configs = [
                {**config, **{param: value}}
                for config in linear_scan_configs
                for value in product_hyperparameters[param]
            ]

    # Set multiheaded to False if context is not multimodal
    linear_scan_configs = [
        {**cfg, "multiheaded": False}
        if ((cfg["context"] != "multimodal") or ("best" in cfg["features"]))
        else cfg
        for cfg in linear_scan_configs
    ]

    return linear_scan_configs


def generate_run_configs(
    contexts_features: list[tuple],
    n_starts: int,
    select_central_values: bool = False,
    params_to_scan: list = None,
):
    STARTS = [str(i) for i in range(n_starts)]
    return generate_linear_scan(
        contexts_features=contexts_features,
        starts=STARTS,
        select_central_values=select_central_values,
        params_to_scan=params_to_scan,
    )


if __name__ == "__main__":
    from training_configuration import (
        CONTEXTS_FEATURES_BY_FIGURE,
        PARAMS_TO_SCAN,
        PATHWAYS_BY_FIGURE,
        SELECT_CENTRAL_VALUES_BY_FIGURE,
    )

    print("=" * 80)
    print("CONFIGURATION SUMMARIES FOR ALL FIGURES")
    print("=" * 80)

    # Default n_starts for summary
    N_STARTS = 5

    for figure_name in sorted(CONTEXTS_FEATURES_BY_FIGURE.keys()):
        print(f"\n{'=' * 80}")
        print(f"FIGURE: {figure_name.upper()}")
        print(f"{'=' * 80}")

        contexts_features = CONTEXTS_FEATURES_BY_FIGURE[figure_name]
        select_central = SELECT_CENTRAL_VALUES_BY_FIGURE[figure_name]
        params_to_scan = PARAMS_TO_SCAN[figure_name]

        print(f"  Contexts/Features combinations: {len(contexts_features)}")
        for i, (ctx, feat) in enumerate(contexts_features[:5], 1):
            print(f"    {i}. {ctx} + {feat}")
        if len(contexts_features) > 5:
            print(f"    ... and {len(contexts_features) - 5} more")

        print(f"\n  Pathways: {len(PATHWAYS_BY_FIGURE.get(figure_name, []))}")
        for i, pathway in enumerate(
            PATHWAYS_BY_FIGURE.get(figure_name, [])[:3], 1
        ):
            print(f"    {i}. {pathway}")
        if len(PATHWAYS_BY_FIGURE.get(figure_name, [])) > 3:
            print(
                f"    ... and {len(PATHWAYS_BY_FIGURE.get(figure_name, [])) - 3} more"
            )

        print(f"\n  Select central values only: {select_central}")
        if params_to_scan:
            print(f"  Params to scan: {params_to_scan}")
        else:
            print("  Params to scan: All scan_attributes")

        # Generate configs
        configs = generate_run_configs(
            contexts_features=contexts_features,
            n_starts=N_STARTS,
            select_central_values=select_central,
            params_to_scan=params_to_scan,
        )

        print(f"\n  Total configurations generated: {len(configs)}")
        print(f"  Configurations per start: {len(configs) // N_STARTS}")

        # Show unique values for key hyperparameters
        if configs:
            key_params = [
                "n_hidden",
                "depth",
                "dropout_rate",
                "nn_init_scale",
                "l1reg_inflater_output",
            ]
            print("\n  Unique values scanned:")
            for param in key_params:
                if param in configs[0]:
                    unique_vals = sorted({cfg[param] for cfg in configs})
                    if len(unique_vals) > 1:
                        print(f"    {param}: {unique_vals}")
                    else:
                        print(f"    {param}: {unique_vals[0]} (fixed)")

        # Show sample config
        if configs:
            print("\n  Sample configuration:")
            sample = configs[0]
            for key in [
                "context",
                "features",
                "samples",
                "job",
                "n_hidden",
                "depth",
                "dropout_rate",
                "nn_init_scale",
                "l1reg_inflater_output",
                "recon_loss",
            ]:
                if key in sample:
                    print(f"    {key}: {sample[key]}")

    print(f"\n{'=' * 80}")
    print("SUMMARY COMPLETE")
    print(f"{'=' * 80}\n")
