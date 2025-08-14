modifications = [
    # baselines
    # "begfr",
    # "berbb2",
    # "bmek",
    # "brps6ka1",
    # transcriptional individualisation
    # "tegfr",
    # "terbb2",
    # "ttgfa",
    # "tbtc",
    # "tereg",
    # "tnrg1",
    # "tnrg2",
    # mutations
    # "mbraf",
    # "mkras",
    # observable function
    # "logobs",
]

PATHWAYS = (
    [
        # "EGFR_MAPK",
        # "EGFR_MAPK__begfr_berbb2_bmek_brps6ka1",
        # "EGFR_MAPK__begfr_berbb2_bmek_brps6ka1_tegfr",
        # "EGFR_MAPK__begfr_berbb2_bmek_brps6ka1_tegfr_tereg_tnrg1",
        "EGFR_MAPK__logobs",
        # "EGFR_MAPK__logobs_tegfr",
        # "EGFR_MAPK__logobs_tegfr_terbb2",
        # "EGFR_MAPK__logobs_tegfr_aggavg_pobs",
        # "EGFR_MAPK__logobs_fegfr_aggavg_pobs",
        # "EGFR_MAPK__logobs_fegfr_aggavg",
        # "EGFR_MAPK__logobs_tegfr_terbb2_aggavg",
        "EGFR_MAPK__logobs_tegfr_aggavg",
        # "EGFR_MAPK__logobs_tegfr_terbb2_aggavg",
        # "EGFR_MAPK",
        # "EGFR_MAPK__tegfr",
        # "EGFR_MAPK__tegfr_aggavg",
        # "EGFR_MAPK__tegfr_terbb2_aggavg",
        # "EGFR_MAPK__begfr_berbb2_bmek_brps6ka1_logobs",
        # "EGFR_MAPK__begfr_berbb2_bmek_brps6ka1_tegfr_logobs",
        # "EGFR_MAPK__begfr_berbb2_bmek_brps6ka1_tegfr_tereg_tnrg1_logobs",
    ]
    # + [
    #     f"EGFR_MAPK__{'_'.join(sorted(['begfr', 'berbb2', 'bmek', 'brps6ka1', 'tegfr', 'ttgfa', 'tbtc', 'tereg', 'tnrg1', 'tnrg2'] + list(combo)))}"
    #     for r in range(1, len(modifications) + 1)
    #     for combo in combinations(modifications, r)
    #     # "EGFR_MAPK_her2",
    #     # "EGFR_MAPK_freeeq",
    #     # "EGFR_MAPK_freeeq_tobs",
    # ]
)

DATASETS = ("dream_cytof",)

# Input contexts/features & feature selection strategy
CONTEXTS_FEATURES = [
    # ("cytof_init", "all"),
    ("cytof_init", "RFE_10_permute"),
    # ("cytof_dynamic", "RFE_10_permute"),
    # ("cytof_dynamic_pca", "RFE_10_permute"),
    # ("cytof_dynamic", "all"),  # only observables that are part of the model (for EGFR_MAPK: ERK, MEK)
    # ("cytof_dynamic_full", "all"),  # all observables
    # ("proteomics", "HVGRFE_6_permute"),
    # ("transcriptomics", "all"),
    # ("transcriptomics", "HVGRFE_6_permute"),
    # ("transcriptomics", "PAM50"),
    # ("transcriptomics", "IHC"),
    # ("transcriptomics", "KRT"),
    # ("multimodal", "optimal"),
    # ("MOSA", "all"),
    # (context, genomic_features)
    # for genomic_features in [
    #     "MSIGDB_KEGG_ERBB",
    #     "MSIGDB_KEGG_MAPK",
    #     "MSIGDB_KEGG_EGFR",
    #     "MSIGDB_KEGG_RTK",
    #     "MSIGDB_KEGG_ERK",
    #     "MSIGDB_BIOCARTA_MAPK",
    #     "MSIGDB_BIOCARTA_EGF",
    #     "MSIGDB_BIOCARTA_ERK",
    #     "MSIGDB_BIOCARTA_RAS",
    #     "MSIGDB_BIOCARTA_P38",
    #     "MSIGDB_PID_ERBB_DOWNSTREAM",
    #     "MSIGDB_PID_ERBB_INTERN",
    #     "MSIGDB_PID_ERBB_PROXIMAL",
    #     "MSIGDB_PID_ERBB",
    #     "MSIGDB_PID_RAS",
    #     "MSIGDB_PID_MAPK",
    #     "MSIGDB_PID_P38_DOWNSTREAM",
    #     "MSIGDB_PID_P38",
    #     "MSIGDB_REACTOME_EGFR_CANCER_VARIANTS",
    #     "MSIGDB_REACTOME_EGFR_DOWNREGULATION",
    #     "MSIGDB_REACTOME_EGFR",
    #     "MSIGDB_REACTOME_EGFR_CANCER",
    #     "MSIGDB_REACTOME_ERBB2_OVEREXPRESSION",
    #     "MSIGDB_REACTOME_ERBB2",
    #     "MSIGDB_REACTOME_ERBB2_CANCER",
    #     "MSIGDB_REACTOME_ERK_TARGETS",
    #     "MSIGDB_REACTOME_ERK",
    #     "MSIGDB_REACTOME_MAPK",
    #     "MSIGDB_REACTOME_MAPK_CANCER",
    #     "MSIGDB_REACTOME_P38",
    #     "MSIGDB_WP_EGFR",
    #     "MSIGDB_WP_EGFR_RESISTANCE",
    #     "MSIGDB_WP_MAPK",
    #     "MSIGDB_WP_P38",
    #     "PAM50",
    #     "MEKFA",
    #     "CompRes",
    #     "MPAS",
    #     "CSC",
    #     "IHC",
    #     "HVGRFE_5_permute",
    #     "HVGRFE_10_permute",
    #     "HVGRFE_15_permute",
    #     "HVGRFE_20_permute",
    #     "HVGRFE_5_tree",
    #     "HVGRFE_10_tree",
    #     "HVGRFE_15_tree",
    #     "HVGRFE_20_tree",
    #     "RFE_5_permute",
    #     "RFE_10_permute",
    #     "RFE_15_permute",
    #     "RFE_20_permute",
    #     "RFE_5_tree",
    #     "RFE_10_tree",
    #     "RFE_15_tree",
    #     "RFE_20_tree",
    # ]
    # for context in ["transcriptomics", "proteomics"]
    # if not (
    #     context == "proteomics" and genomic_features == "MPAS"
    # )  # not enough features
    # (
    #     context,
    #     f"{'' if context in ('cytof_init', 'cytof_dynamic') else 'HVG'}RFE_{n_features}_permute",
    # )
    # for n_features in range(4, 4, 4)
    # for context in [
    #     # "transcriptomics",
    #     # "proteomics",
    #     "cytof_init",
    #     # "cytof_dynamic",
    # ]
]

# Cross-validation splits
SPLITS = {
    "MCF7",
    "BT20",
    "HCC1500",
    "EVSAT",
    "UACC3199",
}

STANDARDISE_FEATURES = {
    # True,
    False,
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
        # 2,
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
    # "True",
    "False",
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
    "range": (0,),
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
BETAS = {"range": (0,), "central_value": 0}
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
    "range": (0,),
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
    "range": (0,),
    "central_value": 0,
}
# previously centered at 1e7, but now excluded from scanned values

# OMEGAS: l1reg_inflater_output -- directly penalises the number of non-negative cell-specific deviations
OMEGAS = {
    "range": (
        0,
        1e-4,
    ),
    "central_value": 1e-4,
}

# THETAS: l2reg_inflater_output -- directly penalises the magnitude of non-negative cell-specific deviations
THETAS = {
    "range": (0,),
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
    "range": (1e-2,),
    "central_value": 1e-2,
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
INFLATER_OUTPUT_REG_EPOCHS = {
    "range": (200,),
    "central_value": 200,
}

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
OPT_STEPS = {"range": (10,), "central_value": 10}

# OPT_MULT: opt_mult, multiplier for the number of steps in each schedule
# OPT_MULT = {
#     # 1,
#     2,
#     # 3,
# }
# Linear scan range for opt_mult
OPT_MULT = {"range": (2,), "central_value": 2}

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

NEPOCH = {
    "range": (500,),
    "central_value": 500,
}

INFLATER_BOUND = {
    "range": (
        # 2,
        3,
        # 4,
        # 5,
    ),
    "central_value": 3,
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

SYNC_ENCODER_INFLATER_REG = True  # whether to synchronise encoder and inflater regularisation hyperparameters


# Figure 1A
CONTEXTS_FEATURES_1A = [
    ("cytof_init", "RFE_10_permute"),
    ("proteomics", "HVGRFE_10_permute"),
    ("transcriptomics", "HVGRFE_10_permute"),
    ("multimodal", "best_RFE_10_permute"),
    ("multimodal", "RFE_10_permute"),
]

PATHWAYS_1A = (
    [
        "EGFR_MAPK__logobs",
    ]
)


# Figure 1B
CONTEXTS_FEATURES_1B = CONTEXTS_FEATURES_1A

PATHWAYS_1B = PATHWAYS_1A


# Figure 2
CONTEXTS_FEATURES_2 = [
    ("cytof_init", "RFE_10_permute"),
    ("cytof_init_plus_tEGFR", "RFE_10_permute"),
    ("cytof_init_plus_pEGFR", "RFE_10_permute"),
]

PATHWAYS_2 = (
    [
        "EGFR_MAPK__logobs",
        "EGFR_MAPK__logobs_fegfr_aggavg",
    ]
)


# Figure 3
CONTEXTS_FEATURES_3 = [
    ("cytof_init", "RFE_10_permute"),
]

PATHWAYS_3 = (
    [
        "EGFR_MAPK__logobs",
        "EGFR_MAPK__logobs_tegfr_aggavg",  # or do we want _pobs? Does that help?
    ]
)


# Master Suite for figures
CONTEXTS_FEATURES_BY_FIGURE = {
    "default": CONTEXTS_FEATURES,
    "figure1a": CONTEXTS_FEATURES_1A,
    "figure1b": CONTEXTS_FEATURES_1B,
    "figure2": CONTEXTS_FEATURES_2,
    "figure3": CONTEXTS_FEATURES_3,
}

PATHWAYS_BY_FIGURE = {
    "default": PATHWAYS,
    "figure1a": PATHWAYS_1A,
    "figure1b": PATHWAYS_1B,
    "figure2": PATHWAYS_2,
    "figure3": PATHWAYS_3,
}

SELECT_CENTRAL_VALUES_BY_FIGURE = {
    "default": False,  # ML param scans
    "figure1a": True,
    "figure1b": False,  # ML param scans
    "figure2": True,
    "figure3": True,
}
