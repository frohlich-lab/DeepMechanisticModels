import itertools as itt
from typing import Union

import numpy as np

from dmm.config_options import scan_attributes
from training_configuration import (
    ACTIVATION_FNS,
    DROPOUT_RATES,
    FREEZE_MEDIANS,
    INFLATER_BOUND,
    # Regularisation-adjacent
    INFLATER_OUTPUT_REG_EPOCHS,
    L1_ENCODE_REGS,
    L1_INFLATE_OUTPUT_REGS,
    L1_INFLATE_REGS,
    L2_INFLATE_OUTPUT_REGS,
    LATENT_DIMS,
    LATENT_DIMS_BY_CONTEXT,
    NEPOCH,
    NETWORK_DEPTH,
    NETWORK_DEPTH_BY_CONTEXT,
    NN_INIT_FN,
    NN_INIT_SCALES,
    OPTIMISERS,
    OREG_ENCODE_REGS,
    OREG_INFLATE_REGS,
    # Regularisation
    ORTH_REG_STRATEGIES,
    RECON_REGS,
    SPARSE_THRESH_PERCS,
    SPLITS,
    STANDARDISE_FEATURES,
    SYMMETRY_REGS,
    USE_BIAS,
    # OTHER OPTIONS
    USE_EARLY_STOP,
)

# Context-specific central-value overrides: param -> {context -> central_value}
_CENTRAL_VALUE_OVERRIDES = {
    "n_hidden": LATENT_DIMS_BY_CONTEXT,
    "depth": NETWORK_DEPTH_BY_CONTEXT,
}

# Default product hyperparameters (uses global SPLITS)


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
    "l1reg_inflate": L1_INFLATE_REGS,
    "oreg_inflate": OREG_INFLATE_REGS,
    "l1reg_encode": L1_ENCODE_REGS,
    "oreg_encode": OREG_ENCODE_REGS,
    "l1reg_inflater_output": L1_INFLATE_OUTPUT_REGS,
    "l2reg_inflater_output": L2_INFLATE_OUTPUT_REGS,
    "inflater_output_reg_epoch": INFLATER_OUTPUT_REG_EPOCHS,
    "sparse_threshold_perc": SPARSE_THRESH_PERCS,
    "recon_loss": RECON_REGS,
    "symm_reg": SYMMETRY_REGS,
    "n_epoch": NEPOCH,
    "inflater_bound": INFLATER_BOUND,
}


# Default product hyperparameters (uses global SPLITS)
def get_product_hyperparameters(splits=None):
    if splits is None:
        splits = SPLITS
    return {
        "activation_fn_name": ACTIVATION_FNS,
        "optimiser": OPTIMISERS,
        "orth_reg_strategy": ORTH_REG_STRATEGIES,
        "use_layer_bias": USE_BIAS,
        "nn_init_fn": NN_INIT_FN,
        "standardise_features": STANDARDISE_FEATURES,
        "freeze_medians": FREEZE_MEDIANS,
        "use_early_stopping": USE_EARLY_STOP,
        "samples": splits,
    }


# Keep backward compatibility
product_hyperparameters = get_product_hyperparameters()


def generate_linear_scan(
    contexts_features: list[tuple],
    starts: list[str],
    select_central_values: bool,
    params_to_scan: list = None,
    splits: set = None,
):
    # Get product hyperparameters with the appropriate splits
    current_product_hyperparameters = get_product_hyperparameters(splits)

    # Check that all hyperparameter options are dicts (central value, range)
    if not all(
        isinstance(hyperparam, dict)
        for hyperparam in linear_hyperparameters.values()
    ):
        raise TypeError("Inconsistent typing for linear scans!")

    # Collect the set of contexts that have any central-value override
    _contexts_with_overrides = {
        ctx for ctx_map in _CENTRAL_VALUE_OVERRIDES.values() for ctx in ctx_map
    }

    # Get global central values
    central_values = {
        hyperparam: linear_hyperparameters[hyperparam]["central_value"]
        for hyperparam in scan_attributes
        if hyperparam in linear_hyperparameters
    }

    def _central_values_for_context(context: str) -> dict:
        """Return central values with context-specific overrides applied.
        Only the central value is overridden; the scan range stays global."""
        if context not in _contexts_with_overrides:
            return central_values
        cv = dict(central_values)
        for param, ctx_map in _CENTRAL_VALUE_OVERRIDES.items():
            if context in ctx_map:
                cv[param] = ctx_map[context]
        return cv

    def _central_for_context(context: str, param: str):
        """Return the central value for *param* in the given *context*."""
        ctx_map = _CENTRAL_VALUE_OVERRIDES.get(param, {})
        if context in ctx_map:
            return ctx_map[context]
        return linear_hyperparameters[param]["central_value"]

    if select_central_values:
        # One config per (start, context, features) with context-aware central values
        linear_scan_configs = [
            prune_config(_central_values_for_context(context))
            | {"context": context, "features": features, "job": start}
            for start, (context, features) in itt.product(
                starts, contexts_features
            )
        ]
    else:
        scan_params = (
            params_to_scan if params_to_scan is not None else scan_attributes
        )

        # Build configs per (start, context, features) so context-specific
        # central values are used while keeping the global scan range.
        linear_scan_configs = []
        for start, (context, features) in itt.product(
            starts, contexts_features
        ):
            ctx_central = _central_values_for_context(context)
            ctx_configs = [
                prune_config({**ctx_central, **{param: value}})
                | {"context": context, "features": features, "job": start}
                for param in scan_params
                if param in linear_hyperparameters
                for value in linear_hyperparameters[param][
                    "range"
                ]  # global range
                if _central_for_context(context, param) != value
            ] + [
                prune_config(ctx_central)
                | {"context": context, "features": features, "job": start}
            ]
            linear_scan_configs.extend(ctx_configs)

    # TODO: fix this, this does not work! (linear scans ok, cartesian product broken)
    # product expand for product hyperparameters
    for param in scan_attributes:
        if param in current_product_hyperparameters:
            linear_scan_configs = [
                {**config, **{param: value}}
                for config in linear_scan_configs
                for value in current_product_hyperparameters[param]
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
    splits: set = None,
):
    STARTS = [str(i) for i in range(n_starts)]
    return generate_linear_scan(
        contexts_features=contexts_features,
        starts=STARTS,
        select_central_values=select_central_values,
        params_to_scan=params_to_scan,
        splits=splits,
    )


if __name__ == "__main__":
    from training_configuration import (
        CONTEXTS_FEATURES_BY_FIGURE,
        PARAMS_TO_SCAN,
        PATHWAYS_BY_FIGURE,
        SELECT_CENTRAL_VALUES_BY_FIGURE,
        SPLITS_BY_FIGURE,
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
        splits = SPLITS_BY_FIGURE[figure_name]

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

        print(f"\n  Splits: {len(splits)}")
        for i, split in enumerate(sorted(splits)[:5], 1):
            print(f"    {i}. {split}")
        if len(splits) > 5:
            print(f"    ... and {len(splits) - 5} more")

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
            splits=splits,
        )

        print(f"\n  Total configurations generated: {len(configs)}")
        print(f"  Configurations per start: {len(configs) // N_STARTS}")

        # Show unique values for key hyperparameters
        if configs:
            print("\n  Unique values scanned:")
            for param in scan_attributes:
                if param in configs[0]:
                    unique_vals = sorted({cfg[param] for cfg in configs})
                    if len(unique_vals) > 1:
                        print(f"    {param}: {unique_vals}")

    print(f"\n{'=' * 80}")
    print("SUMMARY COMPLETE")
    print(f"{'=' * 80}\n")
