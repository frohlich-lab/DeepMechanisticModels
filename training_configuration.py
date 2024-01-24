# ORTHOGONAL REGULARISATION STRATEGIES: L1 vs L2
ORTH_REG_STRATEGIES = (
    "L1",
    "L2",
)

# ALPHAS: values for l1reg_inflate: l1 regularisation of layers that inflate from latent/bottleneck to mechanistic parameters
# from wandb.ai, it seems that rmse_val.min (best value for rmse in validation) is positively correlated with l1reg params
# i.e. the lower the regularisation, the lower the rmse -- this holds for cytof_init and proteomics, not for transcriptomics
ALPHAS = (
    #1e1,
    #5e1,
    1e2,
    #1e3,
    1e4,
    1e6,
    #1e8,
)

# BETAS: values for oreg_inflate: orthogonal regularisation for layers that inflate from latent/bottleneck to mechanistic parameters
# from wandb.ai, it seems like oreg params are negatively correlated with rmse_val.min, i.e. the higher the params,
# the lower the rmse_val.min
BETAS = (
    1e2,
    1e4,
    1e6,
    #1e7,
    #1e8,
    )

# GAMMAS: values for l1reg_encode: l1 regularisation of encoder network (from inputs to latent/bottleneck)
# same as above - trying to lower this value
GAMMAS = (
    #1e1,
    #5e1,
    1e2,
    #1e3,
    1e4,
    1e6,
    #1e8,
)

# DELTAS: values for oreg_encode: orthogonal regularisation of encoder network (from inputs to latent/bottleneck)
DELTAS = (
    1e2,
    1e4,
    1e6,
    #1e7,
    #1e8,
)

# n_hidden: number of dimensions of bottleneck\latent representation obtained using the encoder
# from wandb.ai, it does not seem to matter much, but it is slightly positively correlated with rmse_val.min - lower
# appears to be better? Might be worth trying 3-dim, as it could still be visualised but is in between the values of
# 2 and 4 that have been tried so far?
# From wandb, it looks like values above 4 were barely attempted: there is a single run with 6 for proteomics with weird looking results
# FOR NEXT TIME: try higher values of n_hidden? e.g. 8, 10, 12?
LATENT_DIMS = (
    2,
    #3,
    4,
    6,
    8,
    # 10,
    # 12
)

CONTEXTS_FEATURES = (
    ("cytof_init", "all"),
    # ("cytof_init", "rfe"),
    # ("cytof_init", "lasso"),
    # ("cytof_init", "elastic"),
    # ("cytof_init", "sequential"),
    #("cytof_dynamic", "all"),
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
# NOT CURRENTLY IN USE, REPLACED BY CONTEXTS_FEATURES
#CONTEXTS = (
    #"proteomics",
    #"transcriptomics",
    #"cytof_init",
    #"cytof_dynamic",
#)
# NOT CURRENTLY IN USE, REPLACED BY CONTEXTS_FEATURES
#FEATURES = (
    #"pca",
    #"all",
    #"rfe",
    #"lasso",
    #"elastic",
    # "sequential",
#)
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
