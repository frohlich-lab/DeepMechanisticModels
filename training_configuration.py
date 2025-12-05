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
        "EGFR_MAPK__logobs_tegfr_aggavg",
        "EGFR_MAPK__logobs_tegfr_tbtc_aggavg",
        "EGFR_MAPK__logobs_tegfr_terbb2_tbtc_aggavg",
        "EGFR_MAPK__logobs_tegfr_terbb2_terbb3_tbtc_aggavg",
        "EGFR_MAPK__logobs_tegfr_terbb2_terbb3_tbtc_tnrg1_tnrg2_aggavg",
        "EGFR_MAPK__logobs_terbb3_aggavg",
        "EGFR_MAPK__logobs_terbb3_tnrg1_tnrg2_aggavg",
        # "EGFR_MAPK__logobs_tegfr_terbb2_aggavg",
        # "EGFR_MAPK_AKT__logobs",
        # "EGFR_MAPK_P38__logobs",
        # "EGFR_MAPK_P38_AKT__logobs",
        # "EGFR_MAPK__logobs_tegfr",
        # "EGFR_MAPK__logobs_tegfr_terbb2",
        # "EGFR_MAPK__logobs_tegfr_aggavg_pobs",
        # "EGFR_MAPK__logobs_fegfr_aggavg_pobs",
        # "EGFR_MAPK__logobs_fegfr_aggavg",
        # "EGFR_MAPK__logobs_tegfr_terbb2_aggavg",
        # "EGFR_MAPK__logobs_tegfr_aggavg",
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
    # ("cytof_init_pca", "RFE_10_permute"),
    # ("cytof_dynamic", "RFE_10_permute"),
    # ("cytof_dynamic_pca", "RFE_10_permute"),
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
    # "all",
    "BT20",
    "HCC1500",
    "HCC2185",  # re-added 15.10.2025
    "MCF7",
    "UACC3199",
    # Remaining cell-lines for LOOCV
    # Subchallenge IV - complete cell lines
    # "184A1",
    # "BT474",
    # "BT549",
    # "CAL148",
    # "CAL851",
    # "CAL51",  # microsatellite instability (MSI)
    # "DU4475",
    # "EFM192A",
    # "EVSAT",  # originally in validation set, removed after re-running Cytof Data Analysis on all cell-lines
    # "HBL100",
    # "HCC1187",
    # "HCC1395",
    # "HCC1419",
    # "HCC1569",
    # # "HCC1599",  # outlier
    # "HCC1937",
    # "HCC1954",
    # "HCC2157",  # no transcriptomic data
    # "HCC3153",
    # "HCC38",
    # "HCC70",
    # "HDQP1",
    # "JIMT1",
    # "MCF10A",
    # "MCF10F",  # no transcriptomic data
    # "MDAMB134VI",
    # "MDAMB157",
    # "MDAMB175VII",
    # "MDAMB361",
    # "MDAMB415",
    # "MDAMB453",
    # "MDAkb2",  # no transcriptomic data
    # "MFM223",
    # "MPE600",
    # "MX1",
    # "OCUBM",
    # "T47D",
    # "UACC812",
    # "UACC893",  # added 15.10.2025, removed on 02.12.2025 after re-running Cytof Data Analysis -> still present for LOOCV
    # "ZR7530",
    # Subchallenge 2
    # "184B5",
    # "BT483",
    # "HCC1428",
    # "HCC1806",
    # "HCC202",
    # "Hs578T",
    # "MCF12A",
    # "MDAMB231",
    # "MDAMB468",
    # "SKBR3",
    # "ZR751",
    # Subchallenge 1
    # "AU565",
    # "EFM19",
    # "HCC2218",
    # "LY2",
    # "MACLS2",
    # "MDAMB436",
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
LATENT_DIMS = {
    "range": (
        1,
        2,
        3,
        4,
        6,
        8,
        10,
        # 12,
        15,
        20,
        25,
        30,
    ),
    "central_value": 8,  # updated after v71 and v72 / then after v73 (samples + all, base & tEGFR)
}

# Network Layout/Architecture
NN_STRUCTURE_MULTIPLIER = 2

# Define network depths
NETWORK_DEPTH = {
    "range": (
        0,
        1,
        2,
        3,
        # 4,
        # 5
    ),
    "central_value": 0,  # no hidden layers
}

MULTIHEADED = {
    True,
    # False
}

# Encoder_layer_biases, inflater_layer_biases and decoder_layer_biases all take from a single USE_BIAS hyperparameter
USE_BIAS = (
    # "True",
    "False",
)

# last_layer_activation: use the activation function in the last layer as well (default: not used in output layer)
LAST_LAYER_ACTIVATION = (
    # "True",
    "False",
)

# Encoder_weight/bias_init_fn, inflater_weight/bias_init_fn, decoder_weight/bias_init_fn all take from a single
# NN_INIT_FN hyperparameter
NN_INIT_FN = (
    # "eqx_default",
    "custom",  # custom initialisation with small scale (0.1)
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

DROPOUT_RATES = {
    "range": (
        0,
        0.02,
        0.05,
        0.1,
        0.15,
        0.2,
        0.35,
        0.5,
    ),
    "central_value": 0.1,  # updated after v71_fig1b_p38
}

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

# ALPHAS: l1reg_inflate, l1 regularisation for inflater network.
ALPHAS = {
    "range": (0,),
    "central_value": 0,
}

# BETAS: oreg_inflate, orthogonal regularisation for inflater network.
BETAS = {"range": (0,), "central_value": 0}

# GAMMAS: l1reg_encode, l1 regularisation of encoder network
GAMMAS = {
    "range": (0,),
    "central_value": 0,
}

# DELTAS: oreg_encode, orthogonal regularisation of encoder network
DELTAS = {
    "range": (0,),
    "central_value": 0,
}


# OMEGAS: l1reg_inflater_output -- directly penalises the number of non-negative cell-specific deviations
OMEGAS = {
    "range": (
        # 0,
        # 1e-4,
        # 1e-3,
        1e-2,
        # 1e-1,
        # 1e0,
        # 1e1
    ),
    "central_value": 1e-2,  # re-centering after v70
}

# THETAS: l2reg_inflater_output -- directly penalises the magnitude of non-negative cell-specific deviations
THETAS = {
    "range": (
        0,
        # 1e2,  # just enough to match 0.1 l1reg_inflater_output magnitude
        # 1e3,
        # 1e4,
        # 1e5,
    ),
    "central_value": 0,
}

# EPSILONS: recon_loss, reconstruction loss scale hyperparameter
EPSILONS = {
    "range": (
        # 0,
        1e-4,
        # 1e-3,
        # 1e-2,
        # 1e-1,
        # 1e0,
        # 1e1
    ),
    "central_value": 1e-4,
}

# ZETAS: symm_reg, encoder-decoder symmetry regularisation scale hyperparameter
ZETAS = {"range": (0,), "central_value": 0}

# ETAS: median_reg, median kinetic parameter regularisation scale hyperparameter
ETAS = {"range": (0,), "central_value": 0}

# Epoch at which to disable OMEGA regularisation (l1reg_inflater_output)
# Default: mid-training
# INFLATER_OUTPUT_REG_EPOCHS = {'range': (50, 100, 200, 300, 500), 'central_value': 100}
INFLATER_OUTPUT_REG_EPOCHS = {
    "range": (200,),
    "central_value": 200,
}

# Includes both smaller and larger scales to probe sensitivity of training dynamics.
NN_INIT_SCALES = {
    "range": (0.01, 0.1, 1.0, 10.0),
    "central_value": 0.1,
}

# Percentage thresholds for sparsity
# SPARSE_THRESH_PERCS = {'range': (5, 10, 25, 50, 75, 100), 'central_value': 50}
# SPARSE_THRESH_PERCS = {'range': (25, 50, 75, 100), 'central_value': 50}
SPARSE_THRESH_PERCS = {"range": ("gmm",), "central_value": "gmm"}

# LEARNING SCHEDULE HYPERPARAMETERS
MAX_LEARNING_RATES = {
    "range": (
        # 1e-3,
        # 5e-3,
        1e-2,
    ),
    "central_value": 1e-2,
}

# LEARNING_RATE_SPANS: lrate_span, ratio between learning rate after warm-up and before warm-up within a schedule
LEARNING_RATE_SPANS = {"range": (1e0,), "central_value": 1e0}

# LEARNING_RATE_DECAYS: lrate_decay, decay factor between consecutive schedules
LEARNING_RATE_DECAYS = {
    "range": (
        0.9**0,
        # 0.9**1,
    ),
    "central_value": 0.9**0,
}

# WARMUP_FCTS: warmup_fct, fraction of epochs to be used for warmup within a given schedule
WARMUP_FCTS = {"range": (0.0,), "central_value": 0.0}

# OPT_STEPS: opt_steps, number of steps in the first schedule (they multiply each time in length by opt_mult)
OPT_STEPS = {"range": (10,), "central_value": 10}

# OPT_MULT: opt_mult, multiplier for the number of steps in each schedule
OPT_MULT = {"range": (2,), "central_value": 2}

# Weight-decay for AdamW / schedule-free AdamW
WEIGHT_DECAY = {
    "range": (
        0.0,
        # 1e-2,
        # 1e-4,
    ),
    "central_value": 0.0,
}

# Momentum for AdamW / schedule-free AdamW
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
    ("cytof_init", "RFE_15_permute"),
    ("proteomics", "HVGRFE_10_permute"),
    ("transcriptomics", "HVGRFE_15_permute"),
    # ("multimodal", "best_RFE_10_permute"),
    ("multimodal", "RFE_10_permute"),
]

PATHWAYS_1A = [
    "EGFR_MAPK__logobs",
]


# Figure 1B
CONTEXTS_FEATURES_1B = CONTEXTS_FEATURES_1A

PATHWAYS_1B = PATHWAYS_1A


# Figure 1C
CONTEXTS_FEATURES_1C = (
    [
        (context, features)
        for N in [
            10,
            15,
            20,
            25,
            30,
        ]
        for context, features in zip(
            ["cytof_init", "proteomics", "transcriptomics", "multimodal"],
            [
                f"RFE_{N}_permute",
                f"HVGRFE_{N}_permute",
                f"HVGRFE_{N}_permute",
                # f"best_RFE_{N}_permute",
                f"RFE_{N}_permute",
            ],
            strict=True,
        )
        if not (
            context in ("cytof_init", "multimodal") and N > 37
        )  # only 37 features available
    ]
    + [
        # Curated feature sets
        # MPAS
        ("transcriptomics", "MPAS"),
    ]
    + [
        (context, genomic_features)
        # All MAPK (KEGG, BIOCARTA, PID, REACTOME, WP) + PAM50
        for genomic_features in [
            "MSIGDB_KEGG_MAPK",
            "MSIGDB_BIOCARTA_MAPK",
            "MSIGDB_PID_MAPK",
            "MSIGDB_REACTOME_MAPK",
            "MSIGDB_REACTOME_MAPK_CANCER",
            "MSIGDB_WP_MAPK",
            "PAM50",
        ]
        for context in ["transcriptomics", "proteomics"]
    ]
)

PATHWAYS_1C = PATHWAYS_1A


# Figure 2
CONTEXTS_FEATURES_2 = [
    ("cytof_init", "RFE_15_permute"),
    ("cytof_init_plus_tEGFR", "RFE_15_permute"),
    ("cytof_init_plus_pEGFR", "RFE_15_permute"),
    # ("cytof_init_plus_tEGFR_pEGFR", "RFE_10_permute"),
    (
        "cytof_init_plus_lb",
        "RFE_15_permute",
    ),  # one-hot-encoded luminal/basal subtype from Marcotte et al.
    (
        "cytof_init_plus_intr",
        "RFE_15_permute",
    ),  # one-hot-encoded intrinsic subtype (PAM50-like) from Marcotte et al.
    ("multimodal", "RFE_10_permute"),  # multiheaded
]


PATHWAYS_2 = [
    "EGFR_MAPK__logobs",
    "EGFR_MAPK__logobs_fegfr_aggavg",
]

# Figure 2B -- dropped
# CONTEXTS_FEATURES_2B = [
#     ("cytof_init", "RFE_15_permute"),
#     ("cytof_init_plus_tERBB2", "RFE_15_permute"),
#     ("cytof_init_plus_pERBB2", "RFE_15_permute"),
#     # ("cytof_init_plus_tERBB2_pERBB2", "RFE_10_permute"),
#     ("cytof_init_plus_lb", "RFE_15_permute"),  # one-hot-encoded luminal/basal subtype from Marcotte et al.
#     ("cytof_init_plus_intr", "RFE_15_permute"),  # one-hot-encoded intrinsic subtype (PAM50-like) from Marcotte et al.
#     ("multimodal", "RFE_10_permute"),  # multiheaded
# ]
#
# PATHWAYS_2B = (
#     [
#         "EGFR_MAPK__logobs",
#         # ERBB2 models
#         "EGFR_MAPK__logobs_ferbb2_aggavg",
#         # "EGFR_MAPK__logobs_ferbb2_aggavg_pobs",
#     ]
# )


# Figure 3
CONTEXTS_FEATURES_3 = [
    ("cytof_init", "RFE_15_permute"),
    (
        "cytof_init_plus_lb",
        "RFE_15_permute",
    ),  # one-hot-encoded luminal/basal subtype from Marcotte et al.
    (
        "cytof_init_plus_intr",
        "RFE_15_permute",
    ),  # one-hot-encoded intrinsic subtype (PAM50-like) from Marcotte et al.
    # ("multimodal", "best_RFE_10_permute"),
    # ("multimodal", "best_RFE_15_permute"),
    ("multimodal", "RFE_10_permute"),  # multiheaded
]

PATHWAYS_3 = [
    # base model
    "EGFR_MAPK__logobs",
    # EGFR models
    "EGFR_MAPK__logobs_tegfr_aggavg",
    "EGFR_MAPK__logobs_pegfr_aggavg",
    # ERBB2 models
    "EGFR_MAPK__logobs_terbb2_aggavg",
    "EGFR_MAPK__logobs_perbb2_aggavg",
]

# Figure 3B - scanning n_hidden (need to set PARAMS_TO_SCAN below)
CONTEXTS_FEATURES_3B = CONTEXTS_FEATURES_3
PATHWAYS_3B = [
    "EGFR_MAPK__logobs_tegfr_aggavg",
]

# Figure 3C - run base and tEGFR models on "all" splits with both 6 and 8 hidden units
CONTEXTS_FEATURES_3C = [
    ("cytof_init", "RFE_15_permute"),
    ("multimodal", "RFE_10_permute"),  # multiheaded
]
PATHWAYS_3C = PATHWAYS_3

# Figure 4
CONTEXTS_FEATURES_4 = [
    ("cytof_init", "RFE_10_permute"),
]

PATHWAYS_4 = [
    # Base
    "EGFR_MAPK__logobs",
    "EGFR_MAPK__logobs_tegfr_aggavg",
    # # Baselines
    # "EGFR_MAPK__logobs_begfr_berbb2_bmek_brps6ka1",
    # "EGFR_MAPK__logobs_tegfr_begfr_berbb2_bmek_brps6ka1_aggavg",
    # # Growth Factors
    # "EGFR_MAPK__logobs_ttgfa_tbtc_tereg_tnrg1_tnrg2",
    # "EGFR_MAPK__logobs_tegfr_ttgfa_tbtc_tereg_tnrg1_tnrg2_aggavg",
    # # Baselines and Growth Factors
    # "EGFR_MAPK__logobs_begfr_berbb2_bmek_brps6ka1_ttgfa_tbtc_tereg_tnrg1_tnrg2",
    # "EGFR_MAPK__logobs_tegfr_begfr_berbb2_bmek_brps6ka1_ttgfa_tbtc_tereg_tnrg1_tnrg2_aggavg",
    # Mutations
    "EGFR_MAPK__logobs_mbraf_mkras",
    "EGFR_MAPK__logobs_tegfr_mbraf_mkras_aggavg",
    # # All components
    # "EGFR_MAPK__logobs_begfr_berbb2_bmek_brps6ka1_ttgfa_tbtc_tereg_tnrg1_tnrg2_mbraf_mkras",
    # "EGFR_MAPK__logobs_tegfr_begfr_berbb2_bmek_brps6ka1_ttgfa_tbtc_tereg_tnrg1_tnrg2_mbraf_mkras_aggavg",
]

# Figure 5 - LOOCV on tEGFR model with cytof_init and multimodal contexts
CONTEXTS_FEATURES_5 = [
    ("cytof_init", "RFE_15_permute"),
    ("multimodal", "RFE_10_permute"),  # multiheaded
]
PATHWAYS_5 = [
    "EGFR_MAPK__logobs_tegfr_aggavg",  # tEGFR model only
]

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


# Master Suite for figures
CONTEXTS_FEATURES_BY_FIGURE = {
    "default": CONTEXTS_FEATURES,
    "figure1a": CONTEXTS_FEATURES_1A,
    "figure1b": CONTEXTS_FEATURES_1B,
    "figure1c": CONTEXTS_FEATURES_1C,
    "figure2": CONTEXTS_FEATURES_2,
    # "figure2b": CONTEXTS_FEATURES_2B,
    "figure3": CONTEXTS_FEATURES_3,
    "figure3b": CONTEXTS_FEATURES_3B,
    "figure3c": CONTEXTS_FEATURES_3C,
    "figure4": CONTEXTS_FEATURES_4,
    "figure5": CONTEXTS_FEATURES_5,
}

PATHWAYS_BY_FIGURE = {
    "default": PATHWAYS,
    "figure1a": PATHWAYS_1A,
    "figure1b": PATHWAYS_1B,
    "figure1c": PATHWAYS_1C,
    "figure2": PATHWAYS_2,
    # "figure2b": PATHWAYS_2B,
    "figure3": PATHWAYS_3,
    "figure3b": PATHWAYS_3B,
    "figure3c": PATHWAYS_3C,
    "figure4": PATHWAYS_4,
    "figure5": PATHWAYS_5,
}

SELECT_CENTRAL_VALUES_BY_FIGURE = {
    "default": False,  # ML param scans
    "figure1a": True,
    "figure1b": False,  # ML param scans
    "figure1c": True,  # feature scan only
    "figure2": True,
    # "figure2b": True,
    "figure3": True,
    "figure3b": False,  # scan (but subset to params below)
    "figure3c": False,  # scan (but subset to params below)
    "figure4": True,
    "figure5": True,
}

PARAMS_TO_SCAN = {
    "default": None,
    "figure1a": None,
    "figure1b": None,
    "figure1c": None,
    "figure2": None,
    # "figure2b": None,
    "figure3": None,
    "figure3b": ["n_hidden"],  # only n_hidden
    "figure3c": ["n_hidden", "depth"],  # only n_hidden
    "figure4": None,
    "figure5": None,
}


# Whether to drop p.HER2 from cytof features - default: False (keep)
DROP_HER2_FROM_FEATURES = False
