ALPHAS = (
    1e-2,
    # 1e-1,
    1e-0,
    # 1e1,
    1e2,
    # 1e3,
    1e4,
    # 1e5,
    1e6,
)
LATENT_DIMS = (
    2,
    4,
    6,
    # 8,
    # 10,
    # 12
)
CONTEXTS = (
    "proteomics_pca",
    "transcriptomics_pca",
    "cytof_init_pca",
    "proteomics_zpca",
    "transcriptomics_zpca",
    "cytof_init_zpca",
    "cytof_dynamic_pca",
    # "proteomics_spca_10.0",
    # "proteomics_spca_1.0",
    # "proteomics_spca_0.1",
    # "transcriptomics_spca_10.0",
    # "transcriptomics_spca_1.0",
    # "transcriptomics_spca_0.1",
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
