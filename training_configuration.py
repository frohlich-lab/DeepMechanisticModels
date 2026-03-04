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

BEST_FEATURE_SETS = {
    "cytof_init": "RFE_8_permute",
    "multimodal": "RFE_10_permute",
    "proteomics": "HVGRFE_8_permute",
    "transcriptomics": "HVGRFE_12_permute",
}


PATHWAYS = [
    "EGFR_MAPK__logobs",
    "EGFR_MAPK__logobs_tegfr_aggavg",
    "EGFR_MAPK__logobs_tegfr_tbtc_aggavg",
    "EGFR_MAPK__logobs_tegfr_terbb2_tbtc_aggavg",
    "EGFR_MAPK__logobs_tegfr_terbb2_terbb3_tbtc_aggavg",
    "EGFR_MAPK__logobs_tegfr_terbb2_terbb3_tbtc_tnrg1_tnrg2_aggavg",
    "EGFR_MAPK__logobs_terbb3_aggavg",
    "EGFR_MAPK__logobs_terbb3_tnrg1_tnrg2_aggavg",
]

DATASETS = ("dream_cytof",)

# Input contexts/features & feature selection strategy
CONTEXTS_FEATURES = [
    ("cytof_init", BEST_FEATURE_SETS["cytof_init"]),
    ("proteomics", BEST_FEATURE_SETS["proteomics"]),
    ("transcriptomics", BEST_FEATURE_SETS["transcriptomics"]),
    ("multimodal", BEST_FEATURE_SETS["multimodal"]),
]

# Cross-validation splits
SPLITS = {
    # "all",
    "BT20",
    "HCC1500",
    "HCC2185",  # re-added 15.10.2025
    "MCF7",
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
LATENT_DIMS = {
    "range": (
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
    ),
    "central_value": 4,  # updated in v79
}

# Context-specific overrides for the *central value* of n_hidden (LATENT_DIMS).
# Keys are context names; values are the central value to use for that context.
# The scan range always comes from the global LATENT_DIMS["range"].
# When a context is not listed the global LATENT_DIMS["central_value"] is used.
LATENT_DIMS_BY_CONTEXT: dict[str, int] = {
    "multimodal": 6,
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
        4,
        5,
    ),
    "central_value": 0,  # no hidden layers
}

# Context-specific overrides for the *central value* of depth (NETWORK_DEPTH).
# Same format as LATENT_DIMS_BY_CONTEXT.
NETWORK_DEPTH_BY_CONTEXT: dict[str, int] = {
    "transcriptomics": 3,
    "multimodal": 1,
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
        0.05,
        0.1,
        0.2,
        0.4,
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
L1_INFLATE_REGS = {
    "range": (0,),
    "central_value": 0,
}

# BETAS: oreg_inflate, orthogonal regularisation for inflater network.
OREG_INFLATE_REGS = {
    "range": (0,),
    "central_value": 0,
}

# GAMMAS: l1reg_encode, l1 regularisation of encoder network
L1_ENCODE_REGS = {
    "range": (0,),
    "central_value": 0,
}

# DELTAS: oreg_encode, orthogonal regularisation of encoder network
OREG_ENCODE_REGS = {
    "range": (0,),
    "central_value": 0,
}


# OMEGAS: l1reg_inflater_output -- directly penalises the number of non-negative cell-specific deviations
L1_INFLATE_OUTPUT_REGS = {
    "range": (0, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1),
    "central_value": 1e-1,  # updated post v79
}

# THETAS: l2reg_inflater_output -- directly penalises the magnitude of non-negative cell-specific deviations
L2_INFLATE_OUTPUT_REGS = {
    "range": (
        0,
        1e1,
        1e2,  # just enough to match 0.1 l1reg_inflater_output magnitude
        1e3,
        1e4,
        1e5,
    ),
    "central_value": 0,
}

# EPSILONS: recon_loss, reconstruction loss scale hyperparameter
RECON_REGS = {
    "range": (0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1),
    "central_value": 1e-2,
}

# ZETAS: symm_reg, encoder-decoder symmetry regularisation scale hyperparameter
SYMMETRY_REGS = {
    "range": (0,),
    "central_value": 0,
}

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

# Figure 1A
CONTEXTS_FEATURES_1A = CONTEXTS_FEATURES

PATHWAYS_1A = [
    "EGFR_MAPK__logobs",
    "EGFR_MAPK__logobs_tegfr_aggavg",
]

SPLITS_1A = {
    "BT20",
    "HCC1500",
    "HCC2185",
    "MCF7",
    "UACC3199",
    "all",
}


# Figure 1B
CONTEXTS_FEATURES_1B = CONTEXTS_FEATURES_1A

PATHWAYS_1B = [
    "EGFR_MAPK__logobs",
]

SPLITS_1B = {
    "BT20",
    "HCC1500",
    "HCC2185",
    "MCF7",
    "UACC3199",
}


# Figure 1C
CONTEXTS_FEATURES_1C = (
    [
        (context, features)
        for N in [
            4,
            6,
            8,
            10,
            12,
            16,
            20,
            24,
            28,
            32,
            64,
            128,
            256,
            512,
            1024,
            2048,
            4096,
        ]
        for context, features in zip(
            ["cytof_init", "proteomics", "transcriptomics", "multimodal"],
            [
                f"RFE_{N}_permute",
                f"HVGRFE_{N}_permute",
                f"HVGRFE_{N}_permute",
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

PATHWAYS_1C = PATHWAYS_1B

SPLITS_1C = SPLITS_1B


# Figure 2
CONTEXTS_FEATURES_2 = [
    ("cytof_init", BEST_FEATURE_SETS["cytof_init"]),
    ("cytof_init_plus_tEGFR", BEST_FEATURE_SETS["cytof_init"]),
    ("cytof_init_plus_pEGFR", BEST_FEATURE_SETS["cytof_init"]),
    # one-hot-encoded luminal/basal subtype from Marcotte et al.
    ("cytof_init_plus_lb", BEST_FEATURE_SETS["cytof_init"]),
    # one-hot-encoded intrinsic subtype (PAM50-like) from Marcotte et al.
    ("cytof_init_plus_intr", BEST_FEATURE_SETS["cytof_init"]),
]


PATHWAYS_2 = [
    "EGFR_MAPK__logobs",
    "EGFR_MAPK__logobs_fegfr_aggavg",
]

SPLITS_2 = SPLITS_1B


# Figure 3
CONTEXTS_FEATURES_3 = [
    ("cytof_init", BEST_FEATURE_SETS["cytof_init"]),
    ("multimodal", BEST_FEATURE_SETS["multimodal"]),
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
    # ERBB3 models
    "EGFR_MAPK__logobs_terbb3_aggavg",
    "EGFR_MAPK__logobs_perbb3_aggavg",
]

SPLITS_3 = {"all"}

# Figure 3B - scanning n_hidden (need to set PARAMS_TO_SCAN below)
CONTEXTS_FEATURES_3B = CONTEXTS_FEATURES_3
PATHWAYS_3B = [
    # base model
    "EGFR_MAPK__logobs",
    # EGFR models
    "EGFR_MAPK__logobs_tegfr_aggavg",
]

SPLITS_3B = SPLITS_3

# Figure 3C - run base and tEGFR models on "all" splits with both 6 and 8 hidden units
CONTEXTS_FEATURES_3C = [
    ("cytof_init", BEST_FEATURE_SETS["cytof_init"]),
]
PATHWAYS_3C = PATHWAYS_3

SPLITS_3C = SPLITS_3

# Figure 4
CONTEXTS_FEATURES_4 = [
    ("cytof_init", BEST_FEATURE_SETS["cytof_init"]),
]

PATHWAYS_4 = [
    # Base
    "EGFR_MAPK__logobs_tegfr_aggavg",
    # Growth Factors
    "EGFR_MAPK__logobs_tegfr_ttgfa_tbtc_tereg_tnrg1_tnrg2_aggavg",
    "EGFR_MAPK__logobs_tegfr_ttgfa_aggavg",
    "EGFR_MAPK__logobs_tegfr_tbtc_aggavg",
    "EGFR_MAPK__logobs_tegfr_tereg_aggavg",
    "EGFR_MAPK__logobs_tegfr_tnrg1_aggavg",
    "EGFR_MAPK__logobs_tegfr_tnrg2_aggavg",
    # Mutations
    "EGFR_MAPK__logobs_tegfr_mbraf_mkras_aggavg",
    "EGFR_MAPK__logobs_tegfr_mkras_aggavg",
    "EGFR_MAPK__logobs_tegfr_mbraf_aggavg",
]

SPLITS_4 = SPLITS_3

# Figure 5 - LOOCV on tEGFR model with cytof_init and multimodal contexts
CONTEXTS_FEATURES_5 = [
    ("cytof_init", BEST_FEATURE_SETS["cytof_init"]),
]
PATHWAYS_5 = [
    "EGFR_MAPK__logobs_tegfr_aggavg",  # tEGFR model only
]

SPLITS_5 = {
    "all",
    "BT20",
    "HCC1500",
    "HCC2185",  # re-added 15.10.2025
    "MCF7",
    "UACC3199",
    # Remaining cell-lines for LOOCV
    # Subchallenge IV - complete cell lines
    "184A1",
    "BT474",
    "BT549",
    "CAL148",
    "CAL851",
    "CAL51",
    "DU4475",
    "EFM192A",
    "EVSAT",
    "HBL100",
    "HCC1187",
    "HCC1395",
    "HCC1419",
    "HCC1569",
    # "HCC1599",  # outlier
    "HCC1937",
    "HCC1954",
    # "HCC2157",  # no transcriptomic data
    "HCC3153",
    "HCC38",
    "HCC70",
    "HDQP1",
    "JIMT1",
    "MCF10A",
    # "MCF10F",  # no transcriptomic data
    "MDAMB134VI",
    "MDAMB157",
    "MDAMB175VII",
    "MDAMB361",
    "MDAMB415",
    "MDAMB453",
    # "MDAkb2",  # no transcriptomic data
    "MFM223",
    "MPE600",
    "MX1",
    "OCUBM",
    "T47D",
    "UACC812",
    "UACC893",
    "ZR7530",
    # Subchallenge 2
    "184B5",
    "BT483",
    "HCC1428",
    "HCC1806",
    "HCC202",
    "Hs578T",
    "MCF12A",
    "MDAMB231",
    "MDAMB468",
    "SKBR3",
    "ZR751",
    # Subchallenge 1
    "AU565",
    "EFM19",
    "HCC2218",
    "LY2",
    "MACLS2",
    "MDAMB436",
}

EXTRA_MARKERS_5B_PROT = (
    # generic >0.5 corr with RMSE
    "PDIA3",
    # STRING enriched
    "MAP2K1",
    "PAK4",
    "ARAF",
    # parameter devs
    "STMN1",
    "ARF6",
)

EXTRA_MARKERS_5B_TX = (
    # generic >0.5 corr with RMSE
    "CASK",
    "SMARCD3",
    "CYP24A1",
    "IQGAP2",
    "PLOD2",
    # STRING enriched
    "ERBB3",
    "ERBB2",
    "EGFR",
    # parameter devs
    "JAK2",
    "BTC",
    "MAP3K8",
    "CDK5R1",
    "PRKCD",
    "RPS6KA3",
)

# Gene groups for figure 5b – used for separate sub-plots and per-group
# multiple-testing correction.
EXTRA_MARKERS_5B_GROUPS = {
    "RMSE-correlated": {
        "prot": ("PDIA3",),
        "tx": ("CASK", "SMARCD3", "CYP24A1", "IQGAP2", "PLOD2"),
    },
    "STRING-enriched": {
        "prot": ("MAP2K1", "PAK4", "ARAF"),
        "tx": ("ERBB3", "ERBB2", "EGFR"),
    },
    "Parameter deviations": {
        "prot": ("STMN1", "ARF6"),
        "tx": ("JAK2", "BTC", "MAP3K8", "CDK5R1", "PRKCD", "RPS6KA3"),
    },
}

CONTEXTS_FEATURES_5B = (
    [
        ("cytof_init", BEST_FEATURE_SETS["cytof_init"]),
    ]
    + [
        (f"cytof_init_plus_t{marker}", BEST_FEATURE_SETS["cytof_init"])
        for marker in EXTRA_MARKERS_5B_TX
    ]
    + [
        (f"cytof_init_plus_p{marker}", BEST_FEATURE_SETS["cytof_init"])
        for marker in EXTRA_MARKERS_5B_PROT
    ]
)


SPLITS_5B = {"all"}

CONTEXTS_FEATURES_6 = [
    ("cytof_init", BEST_FEATURE_SETS["cytof_init"]),
]
PATHWAYS_6 = [
    "EGFR_MAPK__logobs_tegfr_aggavg",  # tEGFR model only
]
SPLITS_6 = [
    f"all_{pct}pct_{seed}"
    for pct in (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
    for seed in (0, 1, 2, 3, 4)
] + ["all"]

modifications = [
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
    "figure3": CONTEXTS_FEATURES_3,
    "figure3b": CONTEXTS_FEATURES_3B,
    "figure3c": CONTEXTS_FEATURES_3C,
    "figure4": CONTEXTS_FEATURES_4,
    "figure5": CONTEXTS_FEATURES_5,
    "figure5b": CONTEXTS_FEATURES_5B,
    "figure6": CONTEXTS_FEATURES_6,
}

PATHWAYS_BY_FIGURE = {
    "default": PATHWAYS,
    "figure1a": PATHWAYS_1A,
    "figure1b": PATHWAYS_1B,
    "figure1c": PATHWAYS_1C,
    "figure2": PATHWAYS_2,
    "figure3": PATHWAYS_3,
    "figure3b": PATHWAYS_3B,
    "figure3c": PATHWAYS_3C,
    "figure4": PATHWAYS_4,
    "figure5": PATHWAYS_5,
    "figure5b": PATHWAYS_5,
    "figure6": PATHWAYS_6,
}

SPLITS_BY_FIGURE = {
    "default": SPLITS,
    "figure1a": SPLITS_1A,
    "figure1b": SPLITS_1B,
    "figure1c": SPLITS_1C,
    "figure2": SPLITS_2,
    "figure3": SPLITS_3,
    "figure3b": SPLITS_3B,
    "figure3c": SPLITS_3C,
    "figure4": SPLITS_4,
    "figure5": SPLITS_5,
    "figure5b": SPLITS_5B,
    "figure6": SPLITS_6,
}

SELECT_CENTRAL_VALUES_BY_FIGURE = {
    "default": False,  # ML param scans
    "figure1a": True,
    "figure1b": False,  # ML param scans
    "figure1c": True,  # feature scan only
    "figure2": True,
    "figure3": True,
    "figure3b": False,  # scan (but subset to params below)
    "figure3c": False,  # scan (but subset to params below)
    "figure4": True,
    "figure5": True,
    "figure5b": True,
    "figure6": True,
}

PARAMS_TO_SCAN = {
    "default": None,
    "figure1a": None,
    "figure1b": None,
    "figure1c": None,
    "figure2": None,
    "figure3": None,
    "figure3b": ["n_hidden"],  # only n_hidden
    "figure3c": ["n_hidden", "depth"],  # only n_hidden
    "figure4": None,
    "figure5": None,
    "figure5b": None,
    "figure6": None,
}


# Whether to drop p.HER2 from cytof features - default: False (keep)
DROP_HER2_FROM_FEATURES = False
