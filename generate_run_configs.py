import itertools as itt
import pandas as pd

from training_configuration import (
    CONTEXTS_FEATURES, FEATURES_TRANSFORM,
    SPLITS, PRETRAIN, MEDIAN_INIT,
    LATENT_DIMS, NN_STRUCTURE_MULTIPLIER, NETWORK_LAYOUT, USE_BIAS, LAST_LAYER_ACTIVATION,
    NN_INIT_FN, RECONSTRUCT, ACTIVATION_FNS, OPTIMISERS,
    # Regularisation
    ORTH_REG_STRATEGIES, ALPHAS, BETAS, GAMMAS, DELTAS, OMEGAS, EPSILONS, ZETAS,
    # Learning rate scheduling
    LRATE_PRETRAINING_RATIO,
    MAX_LEARNING_RATES, LEARNING_RATE_SPANS, LEARNING_RATE_DECAYS, WARMUP_FCTS, OPT_STEPS, OPT_MULT,
    WEIGHT_DECAY, MOMENTUM,
    LINEAR_SCHEDULE,
    # OTHER OPTIONS
    USE_EARLY_STOP, DROP_REG_POST_PRETRAIN, SPARSITY_THRESHOLD,
    HP_RUN_MODE, REFINE_HPS
)


def make_configs_unique(run_configs: list[dict]) -> list[dict]:
    return [
        dict(t) for t in set(
            tuple(d.items()) for d in run_configs
        )
    ]


def prune_config(run_config: dict):
    prune = False
    hps_to_prune = []

    # Learning-rate scheduling
    if run_config["optimiser"] == "adamw_sf":
        # remove schedule hyperparams when using schedule-free
        # 0 is not an otherwise valid value (used for regressors and pretraining baselines)
        hps_to_prune.extend(["lrate_span", "lrate_decay", "opt_steps", "opt_mult"])
        run_config["use_simple_linear_schedule"] = False  # also remove linear schedule
        prune = True
    elif run_config["use_simple_linear_schedule"]:  # only with adam or adamw
        hps_to_prune.extend(["opt_steps", "opt_mult", "momentum"])  # use default momentum value
        if run_config["optimiser"] == "adam":
            hps_to_prune.append("weight_decay")  # no weight decay for regular Adam, but keep it for AdamW
        prune = True

    # Network structure - linear benchmark
    if run_config["linear_benchmark"] == 'True':
        # remove network layout hyperparams when using linear benchmark
        hps_to_prune.extend(["nn_structure_multiplier", "depth"])
        run_config["last_layer_activation"] = False  # also remove non-linearities
        prune = True

    # Reconstruction/decoder head
    if not run_config["reconstruct"]:
        # force reconstruction loss and symmetry loss/regularisation params to zero if no decoder head
        hps_to_prune.extend(["recon_loss", "symm_reg"])
        prune = True

    if prune:
        for hp in hps_to_prune:
            run_config[hp] = 0


def generate_linear_scan(STARTS: list[str]):

    hyperparameters = {
        "n_hidden": LATENT_DIMS,
        "network_layout": NETWORK_LAYOUT,
        "l1reg_inflate": ALPHAS,
        "oreg_inflate": BETAS,
        "l1reg_encode": GAMMAS,
        "oreg_encode": DELTAS,
        "l1reg_inflater_output": OMEGAS,
        "recon_loss": EPSILONS,
        "symm_reg": ZETAS,
        "max_lrate": MAX_LEARNING_RATES,
        "lrate_span": LEARNING_RATE_SPANS,
        "lrate_decay": LEARNING_RATE_DECAYS,
        "warmup_fct": WARMUP_FCTS,
        "opt_steps": OPT_STEPS,
        "opt_mult": OPT_MULT,
        "weight_decay": WEIGHT_DECAY,
        "momentum": MOMENTUM,
    }

    # Check that all hyperparameter options are dicts (central value, range)
    if not all(isinstance(hyperparam, dict) for hyperparam in hyperparameters.values()):
        raise TypeError("Inconsistent typing for linear scans!")

    # Get central values
    central_values = {
        hyperparam: hyperparameters[hyperparam]['central_value']
        for hyperparam in hyperparameters.keys()
    }

    # Compute all possible combinations of hyperparameters for the linear scans
    linear_scan_configs = [
        {
            **central_values,  # every other hyperparameter is fixed to the assigned central value
            hyperparam: hp_value,  # specific hyperparameter is varied within its range
            "context": context,
            "features": features,
            "features_transform": features_transform,
            "samples": split,
            "pretrain": pretrain,
            "median_init": median_init,
            "nn_structure_multiplier": NN_STRUCTURE_MULTIPLIER,  # fixed, not tuning it
            "use_layer_bias": use_layer_bias,
            "last_layer_activation": last_layer_activation,
            "nn_init_fn": nn_init_fn,
            "reconstruct": reconstruct,
            "activation_fn_name": activation_fn_name,
            "optimiser": optimiser,
            "orth_reg_strategy": orth_reg_strategy,
            "lrate_pretraining_ratio": lrate_pretraining_ratio,
            "use_simple_linear_schedule": use_simple_linear_schedule,
            "use_early_stopping": use_early_stopping,
            "drop_reg_after_pretrain": drop_reg_after_pretrain,
            "sparsity_threshold": sparsity_threshold,
            "job": job,
        }
        for hyperparam, details in hyperparameters.items()
        for hp_value in details['range']
        for (
            (context, features), features_transform, split, pretrain, median_init,
            use_layer_bias, last_layer_activation, nn_init_fn, reconstruct, activation_fn_name, optimiser,
            orth_reg_strategy, lrate_pretraining_ratio, use_simple_linear_schedule, use_early_stopping,
            drop_reg_after_pretrain, sparsity_threshold, job
        ) in itt.product(
            CONTEXTS_FEATURES, FEATURES_TRANSFORM, SPLITS, PRETRAIN, MEDIAN_INIT,
            USE_BIAS, LAST_LAYER_ACTIVATION, NN_INIT_FN, RECONSTRUCT, ACTIVATION_FNS, OPTIMISERS,
            ORTH_REG_STRATEGIES, LRATE_PRETRAINING_RATIO, LINEAR_SCHEDULE, USE_EARLY_STOP,
            DROP_REG_POST_PRETRAIN, SPARSITY_THRESHOLD, STARTS
        )
    ]

    # Unpack network layout into depth and linear_benchmark + drop the original network_layout key
    for linear_scan_config in linear_scan_configs:
        linear_scan_config["depth"] = linear_scan_config["network_layout"][0]
        linear_scan_config["linear_benchmark"] = linear_scan_config["network_layout"][1]
        linear_scan_config.pop("network_layout")
        prune_config(linear_scan_config)

    # Ensure configs are unique
    unique_linear_scan_configs = make_configs_unique(linear_scan_configs)

    return unique_linear_scan_configs


def generate_grid_search(STARTS: list[str]):
    hyperparameters = {
        "n_hidden": LATENT_DIMS,
        "network_layout": NETWORK_LAYOUT,
        "l1reg_inflate": ALPHAS,
        "oreg_inflate": BETAS,
        "l1reg_encode": GAMMAS,
        "oreg_encode": DELTAS,
        "l1reg_inflater_output": OMEGAS,
        "recon_loss": EPSILONS,
        "symm_reg": ZETAS,
        "max_lrate": MAX_LEARNING_RATES,
        "lrate_span": LEARNING_RATE_SPANS,
        "lrate_decay": LEARNING_RATE_DECAYS,
        "warmup_fct": WARMUP_FCTS,
        "opt_steps": OPT_STEPS,
        "opt_mult": OPT_MULT,
        "weight_decay": WEIGHT_DECAY,
        "momentum": MOMENTUM,
    }

    # Check hyperparameter options are not set up for linear scans by mistake
    if any(isinstance(hyperparam, dict) for hyperparam in hyperparameters.values()):
        raise TypeError("Inconsistent typing for grid search!")

    grid_search_configs = [
        {
            "context": context,
            "features": features,
            "features_transform": features_transform,
            "samples": split,
            "pretrain": pretrain,
            "median_init": median_init,
            "n_hidden": n_hidden,
            "nn_structure_multiplier": NN_STRUCTURE_MULTIPLIER,   # fixed
            "network_layout": network_layout,
            "use_layer_bias": use_layer_bias,
            "last_layer_activation": last_layer_activation,
            "nn_init_fn": nn_init_fn,
            "reconstruct": reconstruct,
            "activation_fn_name": activation_fn_name,
            "optimiser": optimiser,
            "orth_reg_strategy": orth_reg_strategy,
            "l1reg_inflate": l1reg_inflate,
            "oreg_inflate": oreg_inflate,
            "l1reg_encode": l1reg_encode,
            "oreg_encode": oreg_encode,
            "l1reg_inflater_output": l1reg_inflater_output,
            "recon_loss": recon_loss,
            "symm_reg": symm_reg,
            "lrate_pretraining_ratio": lrate_pretraining_ratio,
            "max_lrate": max_lrate,
            "lrate_span": lrate_span,
            "lrate_decay": lrate_decay,
            "warmup_fct": warmup_fct,
            "opt_steps": opt_steps,
            "opt_mult": opt_mult,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "use_simple_linear_schedule": use_simple_linear_schedule,
            "use_early_stopping": use_early_stopping,
            "drop_reg_after_pretrain": drop_reg_after_pretrain,
            "sparsity_threshold": sparsity_threshold,
            "job": job,
        }
        for (
            (context, features), features_transform, split, pretrain, median_init, n_hidden, network_layout,
            use_layer_bias, last_layer_activation, nn_init_fn, reconstruct, activation_fn_name, optimiser,
            orth_reg_strategy, l1reg_inflate, oreg_inflate, l1reg_encode, oreg_encode,
            l1reg_inflater_output, recon_loss, symm_reg,
            lrate_pretraining_ratio, max_lrate, lrate_span, lrate_decay, warmup_fct,
            opt_steps, opt_mult,
            weight_decay, momentum,
            use_simple_linear_schedule, use_early_stopping, drop_reg_after_pretrain,
            sparsity_threshold, job
        ) in itt.product(
            CONTEXTS_FEATURES, FEATURES_TRANSFORM, SPLITS, PRETRAIN, MEDIAN_INIT, LATENT_DIMS, NETWORK_LAYOUT,
            USE_BIAS, LAST_LAYER_ACTIVATION, NN_INIT_FN, RECONSTRUCT, ACTIVATION_FNS, OPTIMISERS,
            ORTH_REG_STRATEGIES, ALPHAS, BETAS, GAMMAS, DELTAS,
            OMEGAS, EPSILONS, ZETAS,
            LRATE_PRETRAINING_RATIO, MAX_LEARNING_RATES, LEARNING_RATE_SPANS, LEARNING_RATE_DECAYS, WARMUP_FCTS,
            OPT_STEPS, OPT_MULT,
            WEIGHT_DECAY, MOMENTUM,
            LINEAR_SCHEDULE, USE_EARLY_STOP, DROP_REG_POST_PRETRAIN,
            SPARSITY_THRESHOLD, STARTS
        )
    ]
    # Unpack network layout into depth and linear_benchmark + drop network_layout key
    for grid_search_config in grid_search_configs:
        grid_search_config["depth"] = grid_search_config["network_layout"][0]
        grid_search_config["linear_benchmark"] = grid_search_config["network_layout"][1]
        grid_search_config.pop("network_layout")
        prune_config(grid_search_config)

    # Ensure configs are unique -- removes combinations of scheduling hyperparams when using schedule-free
    unique_grid_search_configs = make_configs_unique(grid_search_configs)

    return unique_grid_search_configs


def generate_refined_tuning_configs(STARTS: list[str], filepath: str, hps_to_tune: dict):
    if not isinstance(hps_to_tune, dict):
        raise TypeError("Hyperparameters to tune must be specified as a dictionary!")
    df = pd.read_csv(filepath)
    # Subset to test/val (want to refine runs that perform best on unseen data)
    # and drop unnecessary columns + early-stopping (to override it)
    cols_to_drop = list(hps_to_tune.keys()) + ['Unnamed: 0', 'dataset', 'rmse mean', 'rmse std']
    cols_to_drop = [
        col for col in cols_to_drop if col in df.columns
    ]
    df_sub = df[df["dataset"] == "test"].drop(
        columns=cols_to_drop
    )
    # Make sure these columns are integers
    for column in ["n_hidden", "nn_structure_multiplier", "depth", "opt_steps", "opt_mult"]:
        if column in df_sub.columns:  # might be one of the hps to tune - might have been dropped!
            df_sub[column] = df_sub[column].astype(int)
    # Transform to dictionary
    config_dict = df_sub.to_dict(orient='records')
    # Prepare the refined hyperparameter tuning options
    hps_combinations = list(itt.product(*hps_to_tune.values()))
    # Initialise list of hyperparameter configurations
    refined_tuning_configs = []
    for config, job, split, hp_combo in itt.product(
            config_dict, STARTS, SPLITS, hps_combinations
    ):
        # Combine best performing configs with possible choices of hyperparameters
        refined_config = {**config, "samples": split, "job": job}
        refined_config.update(zip(hps_to_tune.keys(), hp_combo))
        refined_tuning_configs.append(refined_config)

    return refined_tuning_configs


def generate_run_configs(n_starts: int, hp_run_mode: str, refine_hps: dict = None):
    STARTS = [str(i) for i in range(n_starts)]
    if hp_run_mode == "linear_scans":
        return generate_linear_scan(STARTS=STARTS)
    elif hp_run_mode == "grid_search":
        return generate_grid_search(STARTS=STARTS)
    elif hp_run_mode == "refined_tuning":
        filepath = "EGFR_MAPK.dream_cytof.top_10_best_dmm.csv"  # TODO @GiacomoFabrini - do not hardcode this!
        if refine_hps is None:
            raise ValueError("Hyperparameters to tune must be specified! refine_hps cannot be None!")
        return generate_refined_tuning_configs(STARTS=STARTS, filepath=filepath, hps_to_tune=refine_hps)
    else:
        raise ValueError(f"Invalid run mode: {hp_run_mode}")


# generate_run_configs(
#     n_starts=10,
#     hp_run_mode="refined_tuning",
#     refine_hps={
#         "use_early_stopping": [True, False],
#         "last_layer_activation": [True, False],
#         "n_hidden": [8, 10, 12],
#     },
# )

generate_run_configs(
    n_starts=10,
    hp_run_mode=HP_RUN_MODE,
    refine_hps=REFINE_HPS,
)
