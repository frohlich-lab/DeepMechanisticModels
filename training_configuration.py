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
    ("cytof_init", "all"),
    # ("cytof_init", "rfe"),
    # ("cytof_init", "lasso"),
    # ("cytof_init", "elastic"),
    # ("cytof_init", "sequential"),
    # ("cytof_dynamic", "all"),  # only observables that are part of the model (for EGFR_MAPK: ERK, MEK)
    # ("cytof_dynamic_full", "all"),  # all observables
    ("proteomics", "all"),
    # ("proteomics", "rfe"),
    # ("proteomics", "lasso"),
    # ("proteomics", "elastic"),
    # ("proteomics", "sequential"),
    ("transcriptomics", "all"),
    # ("transcriptomics", "rfe"),
    # ("transcriptomics", "lasso"),
    # ("transcriptomics", "elastic"),
    # ("transcriptomics", "sequential"),
)

# input features transformation (e.g. PCA) -- keeps 95% variance components, uses whitening
FEATURES_TRANSFORM = {
    "pca",
    # "None",
}

# Cross-validation splits
SPLITS = {
    "0of5",
    # "1of5",
    # "2of5",
    # "3of5",
    # "4of5"
}

PRETRAIN = {
    "True",
}

# Network Structure and Initialisation Hyperparameters

# n_hidden: dimension of latent/bottleneck representation to which input features are encoded.
# From W&B, it does not seem to matter much, but it is slightly positively correlated with rmse_val.min - lower
# appears to be better? -- stat tests show the same.
# LATENT_DIMS = (
#     # 2,
#     # 3,
#     # 4,
#     # 6,
#     # 8,
#     10,
#     # 14  # inflater does not inflate, rather simply processes same shape input
# )
# Define linear scan for LATENT_DIMS
LATENT_DIMS = {'range': (2, 4, 6, 8, 10), 'central_value': 4}

# Network Layout/Architecture
NN_STRUCTURE_MULTIPLIER = 2

# Define network layouts for linear scans in modular fashion
NETWORK_LAYOUT = {
    'range': (
        (0, "False"),
        (1, "False"),
        (2, "False"),
        (3, "False"),
        (4, "False"),
        # (5, "False"),
    ),  # 0-4 hidden layers, no linear benchmark
    'central_value': (2, "False")  # 2 hidden layers, no linear benchmark
}

# NETWORK_LAYOUT = (
#     (2, "False"),
#     # (3, "False"),
# )

# For now: encoder_layer_biases, inflater_layer_biases and decoder_layer_biases all take from a single USE_BIAS
# hyperparameter
USE_BIAS = (
    # "True",
    "False",
)

# last_layer_activation: use the activation function in the last layer as well (default: not used in output layer)
LAST_LAYER_ACTIVATION = (
    "True",
    "False",
)

# For now: encoder_weight/bias_init_fn, inflater_weight/bias_init_fn, decoder_weight/bias_init_fn all take from a single
# NN_INIT_FN hyperparameter
NN_INIT_FN = (
    "eqx_default",
    # "HN",  # He Normal
    # "HU",  # He Uniform
    # "LN",  # LeCun Normal
    # "LU",  # LeCun Uniform
    # "XN",  # Xavier/Glorot Normal
    # "XU",  # Xavier/Glorot Uniform
)


# RECONSTRUCT: whether to add a second head to the autoencoder or not
RECONSTRUCT = (
    # True,
    False,
)


# Training Hyperparameters
# Activation Functions: activation_fn_name
ACTIVATION_FNS = (
    "tanh",
    # "relu",
    # "leaky_relu",
    "swish",
    # "softplus",
)

# optimiser to use
OPTIMISERS = {
    # "adam",
    "adamw",
    # "adamw_sf",
}


# REGULARISATION HYPERPARAMETERS
# ORTHOGONAL REGULARISATION STRATEGIES: L1 vs L2
ORTH_REG_STRATEGIES = (
    # "L1",
    "L2",
)


# Define common linear scan range for regularisation scaling hyperparameters
LINEAR_SCAN_RANGE = (0, 1e0, 1e1, 1e2, 1e3, 1e4)
LINEAR_SCAN_CENTRAL = 1e2

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
ALPHAS = {'range': LINEAR_SCAN_RANGE, 'central_value': LINEAR_SCAN_CENTRAL}

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
BETAS = {'range': (0, ), 'central_value': 0}
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
GAMMAS = {'range': LINEAR_SCAN_RANGE, 'central_value': LINEAR_SCAN_CENTRAL}

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
DELTAS = {'range': (0, ), 'central_value': 0}
# previously centered at 1e7, but now excluded from scanned values

# EPSILONS: recon_loss, reconstruction loss scale hyperparameter
# EPSILONS = (
#     # 0,
#     # 1.0,
#     1e5,
#     1e7,
# )
EPSILONS = {'range': LINEAR_SCAN_RANGE, 'central_value': 0}

# ZETAS: symm_reg, encoder-decoder symmetry regularisation scale hyperparameter
# ZETAS = (
#     # 0,
#     # 1.0,
#     1e5,
#     # 1e8,
# )
ZETAS = {'range': LINEAR_SCAN_RANGE, 'central_value': 0}


# LEARNING SCHEDULE HYPERPARAMETERS
LRATE_PRETRAINING_RATIO = {
    10.0,
}
# MAX_LEARNING_RATES: max_lrate, maximum learning rate at the start of the learning schedule
# MAX_LEARNING_RATES = {
#     1e-1,
#     1e-2,
#     # 1e-3,
# }
# Linear scan range for max_lrate
MAX_LEARNING_RATES = {'range': (1e-4, 1e-3, 1e-2, 1e-1), 'central_value': 1e-1}  # increased central value by one OOM

# LEARNING_RATE_SPANS: lrate_span, ratio between learning rate after warm-up and before warm-up within a schedule
# LEARNING_RATE_SPANS = {
#     1e0,
#     # 1e1,  # ratio of 10
#     # 1e2,  # ratio of 100
#     # 1e3,
# }
# Linear scan range for lrate_span
LEARNING_RATE_SPANS = {'range': (1e0, 1e1, 1e2, 1e3), 'central_value': 1e1}

# LEARNING_RATE_DECAYS: lrate_decay, decay factor between consecutive schedules
# LEARNING_RATE_DECAYS = {
#     0.9**0,  # no decay
#     # 0.9**1,
#     # 0.9**2,
#     # 0.9**3,
# }
# Linear scan range for lrate_decay
LEARNING_RATE_DECAYS = {'range': (0.9**0, 0.9**1, 0.9**2, 0.9**3), 'central_value': 0.9**1}

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
WARMUP_FCTS = {'range': (0.4, 0.2, 0.1, 0.05), 'central_value': 0.1}

# OPT_STEPS: opt_steps, number of steps in the first schedule (they multiply each time in length by opt_mult)
# OPT_STEPS = {
#     # 1,
#     # 2,
#     10,
# }
# Linear scan range for opt_steps
OPT_STEPS = {'range': (1, 2, 5, 10, 100), 'central_value': 10}

# OPT_MULT: opt_mult, multiplier for the number of steps in each schedule
# OPT_MULT = {
#     # 1,
#     2,
#     # 3,
# }
# Linear scan range for opt_mult
OPT_MULT = {'range': (1, 2, 5, 10), 'central_value': 2}

# Weight-decay for AdamW / schedule-free AdamW
# WEIGHT_DECAY = {
#     1e-4, # default in AdamW - optax implementation
# }
WEIGHT_DECAY = {'range': (1e-1, 1e-2, 1e-3, 1e-4), 'central_value': 1e-4}

# Momentum for AdamW / schedule-free AdamW
# MOMENTUM = {
#     0.9,
#     0.98,  # in Schedule-Free Learning paper they test 0.9 and 0.98
#     # 0.99,
# }
MOMENTUM = {'range': (0.9, 0.98), 'central_value': 0.9}


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
PATIENCE = 9  # should be about 50 epochs in linear scale (unsure about log-scale!)

# MIN_IMPROVEMENT: min_improvement, absolute improvement in rmse_val to consider as improvement not to lose patience
MIN_IMPROVEMENT = 0

# Drop regularisation after pretraining
DROP_REG_POST_PRETRAIN = {
    # "True",
    "False",  # disabled this - pretraining is NOT informative and is only helpful to AVOID bad param regimes
    # at the start of training
}

# Threshold to sparsify the model weights if dropping regularisation post pretraining (while keeping learnt sparsity)
SPARSITY_THRESHOLD = {
    # 1e-2,
    1e-3,  # NOT IN USE
}

# Flag to enable/disable statistical tests
RETURN_STAT_TESTS = False

# Maximum number of epochs for training - not varied between individual runs, just globally set here
N_EPOCHS = 1000
PRETRAIN_N_EPOCHS = 2000

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
