PATHWAYS = ("EGFR_MAPK",)

DATASETS = ("dream_cytof",)
# DATASETS = {
#    'synthetic_16_0.2_0.0',
#     'synthetic_32_0.2_0.0',
#     'synthetic_64_0.2_0.0',
#     'synthetic_128_0.2_0.0',
# }

# Input contexts/features & feature selection strategy
CONTEXTS_FEATURES = (
    # ("cytof_init", "all"),
    # ("cytof_init", "rfe"),
    # ("cytof_init", "lasso"),
    # ("cytof_init", "elastic"),
    # ("cytof_init", "sequential"),
    # ("cytof_init", "RFE_5_permute"),
    ("cytof_init", "RFE_10_permute"),
    # ("cytof_init", "RFE_10_tree"),
    # ("cytof_init", "RFE_15_permute"),
    # ("cytof_init", "RFE_20_permute"),
    # ("cytof_dynamic", "all"),  # only observables that are part of the model (for EGFR_MAPK: ERK, MEK)
    # ("cytof_dynamic_full", "all"),  # all observables
    # ("proteomics", "all"),
    # ("proteomics", "rfe"),
    # ("proteomics", "lasso"),
    # ("proteomics", "elastic"),
    # ("proteomics", "sequential"),
    # ("proteomics", "HVGRFE_5_permute"),
    # ("proteomics", "HVGRFE_10_permute"),
    # ("proteomics", "HVGRFE_10_tree"),
    # ("proteomics", "RFE_10_tree"),
    # ("proteomics", "HVGRFE_15_permute"),
    # ("proteomics", "HVGRFE_20_permute"),
    # ("transcriptomics", "all"),
    # ("transcriptomics", "rfe"),
    # ("transcriptomics", "lasso"),
    # ("transcriptomics", "elastic"),
    # ("transcriptomics", "sequential"),
    # ("transcriptomics", "HVGRFE_5_permute"),
    # ("transcriptomics", "HVGRFE_10_permute"),
    # ("transcriptomics", "HVGRFE_10_tree"),
    # ("transcriptomics", "RFE_10_tree"),
    # ("transcriptomics", "HVGRFE_15_permute"),
    # ("transcriptomics", "HVGRFE_20_permute"),
    # ("cytof_init+proteomics+transcriptomics", "all"),  # multi-modal, crude - horizontal stacking
    # ("MOSA", "all"),
)

# Cross-validation splits
SPLITS = {
    "0of5",
    "1of5",
    "2of5",
    "3of5",
    "4of5",
}

PRETRAIN = {
    "True",
}

STANDARDISE_FEATURES = {
    True,
    # False,
}

# INITIALISATION STRATEGY FOR MEDIAN KINETIC PARAMETERS
MEDIAN_INIT = {
    # "per_sample",
    "avg_model",
}

# Train/freeze median kinetic parameters
FREEZE_MEDIANS = {
    # True,
    False,
}

# Network Structure and Initialisation Hyperparameters

# n_hidden: dimension of latent/bottleneck representation to which input features are encoded.
# From W&B, it does not seem to matter much, but it is slightly positively correlated with rmse_val.min - lower
# appears to be better? -- stat tests show the same.
# LATENT_DIMS = (
#     2,
#     3,
#     4,
#     6,
#     8,
#     10,
#     # 14  # inflater does not inflate, rather simply processes same shape input
# )
# Define linear scan for LATENT_DIMS
LATENT_DIMS = {
    "range": (
        2,
        # 3,
        # 4,
        # 6,
        # 8,
        # 10,
    ),
    "central_value": 2,
}

# Network Layout/Architecture
NN_STRUCTURE_MULTIPLIER = 2

# Define network depths
NETWORK_DEPTH = {
    "range": (
        0,
        # 1,
        # 2
    ),
    "central_value": 0,  # no hidden layers
}

# NETWORK_DEPTH = (
#     0,
#     1,
#     2,
# )

# For now: encoder_layer_biases, inflater_layer_biases and decoder_layer_biases all take from a single USE_BIAS
# hyperparameter
USE_BIAS = (
    "True",
    # "False",
)

# last_layer_activation: use the activation function in the last layer as well (default: not used in output layer)
LAST_LAYER_ACTIVATION = (
    # "True",
    "False",
)

# For now: encoder_weight/bias_init_fn, inflater_weight/bias_init_fn, decoder_weight/bias_init_fn all take from a single
# NN_INIT_FN hyperparameter
NN_INIT_FN = (
    # "eqx_default",
    "custom",  # custom initialisation with small scale (0.01)
    # "HN",  # He Normal
    # "HU",  # He Uniform
    # "LN",  # LeCun Normal
    # "LU",  # LeCun Uniform
    # "XN",  # Xavier/Glorot Normal
    # "XU",  # Xavier/Glorot Uniform
)


# RECONSTRUCT: whether to add a second head to the autoencoder or not
RECONSTRUCT = (
    True,
    # False,
)


# Training Hyperparameters
# Activation Functions: activation_fn_name
ACTIVATION_FNS = (
    # "tanh",
    # "relu",
    # "leaky_relu",
    "swish",
    # "softplus",
)

# optimiser to use
OPTIMISERS = {
    "adam",
    # "adamw",
}


# REGULARISATION HYPERPARAMETERS
# ORTHOGONAL REGULARISATION STRATEGIES: L1 vs L2
ORTH_REG_STRATEGIES = (
    # "L1",
    "L2",
)


# Define common linear scan range for regularisation scaling hyperparameters
LINEAR_SCAN_RANGE_L1REG = (
    0,
    # 1e-6,
    # 1e-5,
    # 1e-4,
    # 1e-3,
    # 10**(-2.5),
    # 1e-2,
    # 10**(-1.5),
    # 1e-1,
    # 10**(-0.5),
    # 1e0
)

LINEAR_SCAN_RANGE_OREG = (
    0,
    # 1e-3,
    # 1e-2,
    # 1e-1,
    # 1e0,
    # 1e1,
    # 10**(1.5),
    # 1e2,
    # 10**(2.5),
    # 1e3,
)
LINEAR_SCAN_CENTRAL = 0  # previously 1e2

# ALPHAS: l1reg_inflate, l1 regularisation for inflater network.
# From W&B, it seems that rmse_val.min is positively correlated with `l1reg` params
# i.e. the lower the regularisation, the lower the rmse -- this does not hold for transcriptomics.
# ALPHAS = (
#     # 0,  # tested
#     # 1e1,
#     # 5e1,
#     # 1e2, # tested
#     1e3,
#     # 1e4, # tested
#     1e5,  # reenable
#     # 1e6,
#     # 1e8,
#     # 1e10,  # increasing values
# )
ALPHAS = {
    "range": (0, ),
    "central_value": 0,
}

# BETAS: oreg_inflate, orthogonal regularisation for inflater network.
# From W&B, it seems like oreg params are negatively correlated with rmse_val.min, i.e. the higher the params,
# the lower the rmse_val.min
# BETAS = (
#     # 0,  # tested
#     1e2,  # reenable -- restricting to 1e2 only as it does not seem to have much of an impact!
#     # 1e4, # tested
#     # 1e5,
#     # 1e6, # tested
#     1e7,
#     # 1e8,
# )
BETAS = {
    "range": (0, ),
    "central_value": 0
}
# previously centred at 1e7, but now excluded from scanned values

# GAMMAS: l1reg_encode, l1 regularisation of encoder network
# GAMMAS = (
#     # 0,  # tested
#     # 1e1,
#     # 5e1,
#     # 1e2,  # tested
#     1e3,
#     # 1e4,  # tested
#     # 1e5,
#     # 1e6,  # tested
#     # 1e8,
#     # 1e10,  # increasing values
# )
GAMMAS = {
    "range": (0, ),
    "central_value": 0,
}

# DELTAS: oreg_encode, orthogonal regularisation of encoder network
# DELTAS = (
#     # 0,  # tested
#     # 1e2, # tested
#     # 1e4, # tested
#     # 1e6,  # tested
#     1e7,
#     # 1e8,
#     # 1e10,  # increasing values
# )
DELTAS = {
    "range": (0, ),
    "central_value": 0,
}
# previously centered at 1e7, but now excluded from scanned values

# OMEGAS: l1reg_inflater_output -- directly penalises the number of non-negative cell-specific deviations
# 1e-4 seems to help with both rmse_train and rmse_val on all contexts -> using this as central value
# to scan switching epoch
# 09.01.2024 - added pre-multiplier in DMM = 1e-6. Therefore, 1 -> 1e-6; 1e2 -> 1e-4
# OMEGAS = {'range': (0, 1e-4, 1e-3), 'central_value': 0}
# OMEGAS = {'range': (1e0, 1e1, 1e2, 1e3, ), 'central_value': 1e2}
# OMEGAS = {'range': (0, 1e0, 1e1, 1e2, 1e3, ), 'central_value': 1e2}
OMEGAS = {
    "range": (62.5, ),
    "central_value": 62.5,
}

# THETAS: l2reg_inflater_output -- directly penalises the magnitude of non-negative cell-specific deviations
THETAS = {
    "range": (0, ),
    "central_value": 0,
}

# EPSILONS: recon_loss, reconstruction loss scale hyperparameter
# EPSILONS = (
#     # 0,
#     # 1.0,
#     1e5,
#     1e7,
# )
EPSILONS = {
    "range": (0, 1e-3, 1e-2, 1e-1, 1e0),
    "central_value": 0
}

# ZETAS: symm_reg, encoder-decoder symmetry regularisation scale hyperparameter
# ZETAS = (
#     # 0,
#     # 1.0,
#     1e5,
#     # 1e8,
# )
ZETAS = {"range": (0,), "central_value": 0}

# ETAS: median_reg, median kinetic parameter regularisation scale hyperparameter
# ETAS = {'range': (0, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100), 'central_value': 0}
ETAS = {"range": (0,), "central_value": 0}

# Epoch at which to disable OMEGA regularisation (l1reg_inflater_output)
# Default: mid-training
# INFLATER_OUTPUT_REG_EPOCHS = {'range': (50, 100, 200, 300, 500), 'central_value': 100}
INFLATER_OUTPUT_REG_EPOCHS = {"range": (100,), "central_value": 100}

# Percentage thresholds for sparsity
# SPARSE_THRESH_PERCS = {'range': (5, 10, 25, 50, 75, 100), 'central_value': 50}
# SPARSE_THRESH_PERCS = {'range': (25, 50, 75, 100), 'central_value': 50}
SPARSE_THRESH_PERCS = {"range": ("gmm",), "central_value": "gmm"}

# LEARNING SCHEDULE HYPERPARAMETERS
# MAX_LEARNING_RATES: max_lrate, maximum learning rate at the start of the learning schedule
# MAX_LEARNING_RATES = {
#     1e-1,
#     1e-2,
#     # 1e-3,
# }
# Linear scan range for max_lrate
MAX_LEARNING_RATES = {
    "range": (
        # 1e-3,
        # 5e-3,
        1e-2,
    ),
    "central_value": 1e-2,
}  # increased central value by one OOM

# LEARNING_RATE_SPANS: lrate_span, ratio between learning rate after warm-up and before warm-up within a schedule
# LEARNING_RATE_SPANS = {
#     1e0,
#     # 1e1,  # ratio of 10
#     # 1e2,  # ratio of 100
#     # 1e3,
# }
# Linear scan range for lrate_span
LEARNING_RATE_SPANS = {"range": (1e0,), "central_value": 1e0}

# LEARNING_RATE_DECAYS: lrate_decay, decay factor between consecutive schedules
# LEARNING_RATE_DECAYS = {
#     0.9**0,  # no decay
#     # 0.9**1,
#     # 0.9**2,
#     # 0.9**3,
# }
# Linear scan range for lrate_decay
LEARNING_RATE_DECAYS = {
    "range": (
        0.9**0,
        # 0.9**1,
    ),
    "central_value": 0.9**0,
}

# WARMUP_FCTS: warmup_fct, fraction of epochs to be used for warmup within a given schedule
# WARMUP_FCTS = {
#     # 0.4,
#     # 0.2,
#     0.1,
#     # 0.05,
#     # 1e-2,
#     # 1e-3,
# }
# Linear scan range for warmup_fct
WARMUP_FCTS = {"range": (0.0,), "central_value": 0.0}

# OPT_STEPS: opt_steps, number of steps in the first schedule (they multiply each time in length by opt_mult)
# OPT_STEPS = {
#     # 1,
#     # 2,
#     10,
# }
# Linear scan range for opt_steps
OPT_STEPS = {"range": (1, 2, 5, 10, 100), "central_value": 10}

# OPT_MULT: opt_mult, multiplier for the number of steps in each schedule
# OPT_MULT = {
#     # 1,
#     2,
#     # 3,
# }
# Linear scan range for opt_mult
OPT_MULT = {"range": (1, 2, 5, 10), "central_value": 2}

# Weight-decay for AdamW / schedule-free AdamW
# WEIGHT_DECAY = {
#     1e-4, # default in AdamW - optax implementation
# }
WEIGHT_DECAY = {
    "range": (
        0.0,
        # 1e-2,
        # 1e-4,
    ),
    "central_value": 0.0,
}

# Momentum for AdamW / schedule-free AdamW
# MOMENTUM = {
#     0.9,
#     0.98,  # in Schedule-Free Learning paper they test 0.9 and 0.98
#     # 0.99,
# }
MOMENTUM = {
    "range": (
        0.9,
        # 0.98
    ),
    "central_value": 0.9,
}


# LINEAR_SCHEDULE: use_simple_linear_schedule, can override learning schedule and produce a single linear schedule
# with the given max learning rate, warm-up and decay
LINEAR_SCHEDULE = {
    True,  # simple linear learning rate schedule - warmup + decay, no cosine annealing schedules
    # False,
}


# EARLY-STOPPING HYPERPARAMETERS
# USE_EARLY_STOP: use_early_stopping, enables early-stopping via flax.training.early_stopping
USE_EARLY_STOP = {
    # True,
    False,  # disabled for now - allow to overfit
}

# PATIENCE: patience, number of consecutive epochs where we tolerate rmse_val not improving by at least min_improvement
PATIENCE = (
    9  # should be about 50 epochs in linear scale (unsure about log-scale!)
)

# MIN_IMPROVEMENT: min_improvement, absolute improvement in rmse_val to consider as improvement not to lose patience
MIN_IMPROVEMENT = 0

# Flag to enable/disable statistical tests
RETURN_STAT_TESTS = False

# Maximum number of epochs for training - not varied between individual runs, just globally set here
N_EPOCHS = 500
PRETRAIN_N_EPOCHS = 0

# Type of run
HP_RUN_MODE = "linear_scans"

# If HP_RUN_MODE = 'refined_tuning', need to specify REFINE_HPS
# can use options above and simply pack it into a dictionary
# If not, leave None
REFINE_HPS = None
# REFINE_HPS = {
#     "use_early_stopping": USE_EARLY_STOP,
#     "last_layer_activation": LAST_LAYER_ACTIVATION,
# }

N_ENSEMBLE_MEMBERS = 1  # number of ensemble members to average over (top N RMSE val across training) -- NOT IN USE
N_ENSEMBLE_EVALUATION = 1  # how many ensemble members to use during evaluation

SYNC_ENCODER_INFLATER_REG = True  # whether to synchronise encoder and inflater regularisation hyperparameters
