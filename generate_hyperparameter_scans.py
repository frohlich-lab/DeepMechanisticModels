import itertools as itt
from common import EVALUATION_TRAINING, SafeDict
from training_configuration import (
    CONTEXTS_FEATURES, SPLITS, PRETRAIN,
    LATENT_DIMS, NETWORK_LAYOUT, USE_BIAS, NN_INIT_FN,
    RECONSTRUCT, ACTIVATION_FNS, OPTIMISERS,
    ORTH_REG_STRATEGIES, ALPHAS, BETAS, GAMMAS, DELTAS, EPSILONS, ZETAS,
    MAX_LEARNING_RATES, LEARNING_RATE_SPANS, LEARNING_RATE_DECAYS, WARMUP_FCTS, OPT_STEPS, OPT_MULT, LINEAR_SCHEDULE,
    USE_EARLY_STOP, DROP_REG_POST_PRETRAIN, SPARSITY_THRESHOLD, FEATURES_TRANSFORM
)


def generate_hyperparam_scans(n_starts: int):
    STARTS = [str(i) for i in range(n_starts)]

    hyperparameters = {
        "n_hidden": LATENT_DIMS,
        "network_layout": NETWORK_LAYOUT,
        "l1reg_inflate": ALPHAS,
        "oreg_inflate": BETAS,
        "l1reg_encode": GAMMAS,
        "oreg_encode": DELTAS,
        "recon_loss": EPSILONS,
        "symm_reg": ZETAS,
        "max_lrate": MAX_LEARNING_RATES,
        "lrate_span": LEARNING_RATE_SPANS,
        "lrate_decay": LEARNING_RATE_DECAYS,
        "warmup_fct": WARMUP_FCTS,
        "opt_steps": OPT_STEPS,
        "opt_mult": OPT_MULT,
    }

    if all(isinstance(hyperparam, tuple) for hyperparam in hyperparameters.values()):
        mode = "product_grid"
    elif all(isinstance(hyperparam, dict) for hyperparam in hyperparameters.values()):
        mode = "linear_scan"
    else:
        raise TypeError("Inconsistent typing for hyperparameter scans!")

    if mode == 'linear_scan':
        central_values = {
            hyperparam: hyperparameters[hyperparam]['central_value']
            for hyperparam in hyperparameters.keys()
        }
        hyperparam_configurations = [
            {
                **central_values,
                hyperparam: hp_value,
                "context": context,
                "features": features,
                "features_transform": features_transform,
                "samples": split,
                "pretrain": pretrain,
                "use_layer_bias": use_layer_bias,
                "nn_init_fn": nn_init_fn,
                "reconstruct": reconstruct,
                "activation_fn_name": activation_fn_name,
                "optimiser": optimiser,
                "orth_reg_strategy": orth_reg_strategy,
                "use_simple_linear_schedule": use_simple_linear_schedule,
                "use_early_stopping": use_early_stopping,
                "drop_reg_after_pretrain": drop_reg_after_pretrain,
                "sparsity_threshold": sparsity_threshold,
                "job": job,
            }
            for hyperparam, details in hyperparameters.items()
            for hp_value in details['range']
            for (
                (context, features), features_transform, split, pretrain,
                use_layer_bias, nn_init_fn, reconstruct, activation_fn_name, optimiser,
                orth_reg_strategy, use_simple_linear_schedule, use_early_stopping, drop_reg_after_pretrain,
                sparsity_threshold, job
            ) in itt.product(
                CONTEXTS_FEATURES, FEATURES_TRANSFORM, SPLITS, PRETRAIN,
                USE_BIAS, NN_INIT_FN, RECONSTRUCT, ACTIVATION_FNS, OPTIMISERS,
                ORTH_REG_STRATEGIES, LINEAR_SCHEDULE, USE_EARLY_STOP, DROP_REG_POST_PRETRAIN,
                SPARSITY_THRESHOLD, STARTS
            )
        ]
    elif mode == 'product_grid':
        hyperparam_configurations = [
            {
                "context": context,
                "features": features,
                "features_transform": features_transform,
                "samples": split,
                "pretrain": pretrain,
                "n_hidden": n_hidden,
                "network_layout": network_layout,
                "use_layer_bias": use_layer_bias,
                "nn_init_fn": nn_init_fn,
                "reconstruct": reconstruct,
                "activation_fn_name": activation_fn_name,
                "optimiser": optimiser,
                "l1reg_inflate": l1reg_inflate,
                "oreg_inflate": oreg_inflate,
                "l1reg_encode": l1reg_encode,
                "oreg_encode": oreg_encode,
                "recon_loss": recon_loss,
                "symm_reg": symm_reg,
                "max_lrate": max_lrate,
                "lrate_span": lrate_span,
                "lrate_decay": lrate_decay,
                "warmup_fct": warmup_fct,
                "opt_steps": opt_steps,
                "opt_mult": opt_mult,
                "orth_reg_strategy": orth_reg_strategy,
                "use_simple_linear_schedule": use_simple_linear_schedule,
                "use_early_stopping": use_early_stopping,
                "drop_reg_after_pretrain": drop_reg_after_pretrain,
                "sparsity_threshold": sparsity_threshold,
                "job": job,
            }
            for ((context, features), features_transform, split, pretrain, n_hidden, network_layout,
                 use_layer_bias, nn_init_fn, reconstruct, activation_fn_name, optimiser,
                 l1reg_inflate, oreg_inflate, l1reg_encode, oreg_encode, recon_loss, symm_reg,
                 max_lrate, lrate_span, lrate_decay, warmup_fct, opt_steps, opt_mult,
                 orth_reg_strategy, use_simple_linear_schedule, use_early_stopping, drop_reg_after_pretrain,
                 sparsity_threshold, job) in itt.product(
                CONTEXTS_FEATURES, FEATURES_TRANSFORM, SPLITS, PRETRAIN, LATENT_DIMS, NETWORK_LAYOUT,
                USE_BIAS, NN_INIT_FN, RECONSTRUCT, ACTIVATION_FNS, OPTIMISERS,
                ALPHAS, BETAS, GAMMAS, DELTAS, EPSILONS, ZETAS,
                MAX_LEARNING_RATES, LEARNING_RATE_SPANS, LEARNING_RATE_DECAYS, WARMUP_FCTS, OPT_STEPS, OPT_MULT,
                ORTH_REG_STRATEGIES, LINEAR_SCHEDULE, USE_EARLY_STOP, DROP_REG_POST_PRETRAIN,
                SPARSITY_THRESHOLD, STARTS
            )
        ]
    # Unpack network layout
    for hyperparam_configuration in hyperparam_configurations:
        hyperparam_configuration["depth"] = hyperparam_configuration["network_layout"][0]
        hyperparam_configuration["linear_benchmark"] = hyperparam_configuration["network_layout"][1]
        hyperparam_configuration.pop("network_layout")

    training_evaluations = [
        EVALUATION_TRAINING.format_map(SafeDict(**hyperparam_configuration, dataset=dataset))
        for hyperparam_configuration in hyperparam_configurations
        for dataset in ['train', 'test']
    ]
    return hyperparam_configurations

generate_hyperparam_scans(10)
