# ORTHOGONAL REGULARISATION STRATEGIES: L1 vs L2
ORTH_REG_STRATEGIES = (
    # "L1",
    "L2",  # restricting to L2 only post stat tests
)

# ALPHAS: values for l1reg_inflate: l1 regularisation of layers that inflate from latent/bottleneck to kinetic params.
# From W&B, it seems that rmse_val.min is positively correlated with l1reg params
# i.e. the lower the regularisation, the lower the rmse -- this does not hold for transcriptomics.
ALPHAS = (
    0,  # tested
    # 1e1,
    # 5e1,
    # 1e2, # tested
    # 1e3,
    # 1e4, # tested
    1e6,  # reenable
    1e8,
    1e10,  # increasing values
)

# BETAS: values for oreg_inflate: orthogonal regularisation for layers that inflate from bottleneck to kinetic params.
# From W&B, it seems like oreg params are negatively correlated with rmse_val.min, i.e. the higher the params,
# the lower the rmse_val.min
BETAS = (
    0,  # tested
    1e2,  # reenable -- restricting to 1e2 only as it does not seem to have much of an impact!
    # 1e4, # tested
    # 1e6, # tested
    # 1e7,
    # 1e8,
    )

# GAMMAS: values for l1reg_encode: l1 regularisation of encoder network (from inputs to bottleneck).
GAMMAS = (
    0,  # tested
    # 1e1,
    # 5e1,
    # 1e2,  # tested
    # 1e3,
    # 1e4,  # tested
    1e6,  # tested
    1e8,
    1e10,  # increasing values
)

# DELTAS: values for oreg_encode: orthogonal regularisation of encoder network (from inputs to bottleneck)
DELTAS = (
    0,  # tested
    # 1e2, # tested
    # 1e4, # tested
    1e6,  # tested
    # 1e7,
    1e8,
    1e10,  # increasing values
)

# n_hidden: number of dimensions of bottleneck representation obtained using the encoder.
# From W&B, it does not seem to matter much, but it is slightly positively correlated with rmse_val.min - lower
# appears to be better? -- stat tests show the same.
LATENT_DIMS = (
    2,
    # 3,
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

OPTIMISERS = {
    "adam",
    "adamw",
}

# DEFAULT_LINEAR_SCHEDULE = dict(
#     init_value=1e-2,
#     transition_steps=100,
#     end_value=1e-3,
# )

PATIENCE = 19  # before it was 9 - should correspond to (19+1)*5 = 100 epochs overall

MIN_IMPROVEMENT = 0
