ALPHAS = {1e-4, 1e-3, 1e-2, 1e-1, 1e-0, 1e1, 1e2}
LATENT_DIMS = {2, 4, 8}
# CONTEXTS = {'baseline', 'init', 'dynamic', }
CONTEXTS = {'baseline', }
PATHWAYS = {'EGFR_MAPK', }
#DATASETS = {'dream_cytof', }
DATASETS = {
#    'synthetic_16_0.5', 'synthetic_16_0.1', 'synthetic_16_0.05', 'synthetic_16_0.01',
#     'synthetic_32_0.5', 'synthetic_32_0.1', 'synthetic_32_0.05', 'synthetic_32_0.01',
    'synthetic_64_1.0_1.0', 'synthetic_64_0.1_1.0', 'synthetic_64_0.01_1.0',
    'synthetic_64_1.0_0.1', 'synthetic_64_0.1_0.1', 'synthetic_64_0.01_0.1',
    'synthetic_64_1.0_0.01', 'synthetic_64_0.1_0.01', 'synthetic_64_0.01_0.01',
#     'synthetic_128_0.5', 'synthetic_128_0.1', 'synthetic_128_0.05', 'synthetic_128_0.01',
}
SPLITS = {'0_5', }
