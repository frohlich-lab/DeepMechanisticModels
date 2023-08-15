ALPHAS = (
    1e-1,
    1e-0,
    1e1,
    1e2,
    1e3,
    1e4,
)
LATENT_DIMS = (
    2,
    4,
    6,
    8,
    # 10,
    # 12
)
CONTEXTS_FEATURES = (
    ("cytof_init", "all"),
    ("cytof_init", "rfe"),
    ("cytof_init", "lasso"),
    ("cytof_init", "elastic"),
    ("cytof_init", "sequential"),
    ("cytof_dynamic", "all"),
    ("proteomics", "all"),
    ("proteomics", "rfe"),
    ("proteomics", "lasso"),
    ("proteomics", "elastic"),
    # ("proteomics", "sequential"),
    ("transcriptomics", "all"),
    ("transcriptomics", "rfe"),
    ("transcriptomics", "lasso"),
    ("transcriptomics", "elastic"),
    # ("transcriptomics", "sequential"),
)
CONTEXTS = (
    # "proteomics",
    # "transcriptomics",
    "cytof_init",
    "cytof_dynamic",
)
FEATURES = (
    "all",
    "rfe",
    "lasso",
    "elastic",
    "sequential",
)
PATHWAYS = ("EGFR_MAPK",)
DATASETS = ("dream_cytof",)
# DATASETS = {
#    'synthetic_16_0.2_0.0',
#     'synthetic_32_0.2_0.0',
#     'synthetic_64_0.2_0.0',
#     'synthetic_128_0.2_0.0',
# }
SPLITS = {
    "0_5",
    # '1_5',
    # '2_5',
    # '3_5',
    # '4_5'
}
PRETRAIN = {
    "True",
}
