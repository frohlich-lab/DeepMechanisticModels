"""
Pretraining of population + individual parameters based on per sample
pretraining
"""

import sys
import os

import pandas as pd
import numpy as np
import scipy.linalg as la
import fides

from pypesto.optimize import FidesOptimizer

from mEncoder.autoencoder import MechanisticAutoEncoder
from process_data import training_samples, Wildcards
from mEncoder.pretraining import (
    generate_cross_sample_pretraining_problem,
    pretrain,
    store_and_plot_pretraining,
)
from mEncoder import (
    MODEL_FEATURE_PREFIX,
    parameter_boundaries_scales,
    pretrain_dir,
    data_dir,
    ESTIMATION_OUTFILE_TEMP,
)

from jax.config import config
config.update("jax_enable_x64", True)
# config.update("jax_disable_jit", True)

MODEL = sys.argv[1]
DATA = sys.argv[2]
CONTEXT = sys.argv[3]
SAMPLES = sys.argv[4]
N_HIDDEN = int(sys.argv[5])
ALPHA = float(sys.argv[6])
JOB = int(sys.argv[7])

samples = training_samples(Wildcards(DATA, SAMPLES))
mae = MechanisticAutoEncoder(
    N_HIDDEN,
    (
        data_dir / f"{DATA}__{MODEL}__measurements.tsv",
        data_dir / f"{DATA}__{MODEL}__conditions.tsv",
        data_dir / f"{DATA}__{MODEL}__observables.tsv",
    ),
    pathway_name=MODEL,
    samples=samples,
    l1reg=ALPHA,
    contextualization=CONTEXT,
    n_threads=4,
)

problem = generate_cross_sample_pretraining_problem(mae)
pretrained_samples = {}

outdir = pretrain_dir / MODEL / DATA
output_prefix = os.path.splitext(
    ESTIMATION_OUTFILE_TEMP.format(
        context=CONTEXT, samples=SAMPLES, n_hidden=N_HIDDEN, alpha=ALPHA, job=JOB
    )
)[0]

for sample in samples:
    df = pd.read_csv(pretrain_dir / MODEL / DATA / f"{sample}.csv", index_col=[0])
    pretrained_samples[sample] = df[
        [col for col in df.columns if not col.startswith(MODEL_FEATURE_PREFIX)]
    ]


def startpoints(**kwargs):
    """
    Custom startpoint routine for cross sample pretraining. This function
    uses the results computed for the completely unconstrained problem
    where the model is just fitted to each individual sample. For each
    sample, a random local optimization result is picked. Then
    shared population parameters are computed as mean over all samples and
    sample specific input parameter are computed by substracting this mean
    from the local solution.
    """
    n_starts = kwargs["n_starts"]
    lb = kwargs["lb"]

    dim = lb.size
    xs = np.empty((n_starts, dim))

    for istart in range(n_starts):
        # use parameter values from random start for each sample
        par_combo = pd.concat(
            [
                pretraining[
                    pretraining.index
                    == np.min([np.random.poisson(2, 1)[0], len(pretraining) - 1])
                ]
                for pretraining in pretrained_samples.values()
            ]
        )
        par_combo.index = list(pretrained_samples.keys())
        par_combo = par_combo.reindex(mae.sample_names)
        means = par_combo.mean(skipna=True)
        par_combo -= means

        inputs = [
            "__".join(p.split("__")[:-1]).replace(MODEL_FEATURE_PREFIX, "")
            for p in mae.petab_importer.petab_problem.parameter_df.index
            if p.startswith(MODEL_FEATURE_PREFIX) and p.endswith(par_combo.index[0])
        ]

        # use this code to use reference inputs for initialization
        # if DATA.startswith("synthetic"):
        #     reference_inputs = pd.read_csv(
        #         data_dir / f"{DATA}__{MODEL}__reference_inputs.csv", index_col=[0]
        #     )
        #     for col in par_combo.columns:
        #         means[col] = reference_inputs.loc[col.replace("_obs", "")].values[0]
        #         if col not in inputs:
        #             continue
        #         for sample in par_combo.index:
        #             par_combo.loc[sample, col] = reference_inputs.loc[
        #                 f"{MODEL_FEATURE_PREFIX}{col}_{sample}"
        #             ].values[0, 0]
        w = la.lstsq(mae.data_pca[:, : mae.n_latent], par_combo[inputs].values)[
            0
        ].flatten()
        assert f"inflate_{len(w)-1}_weight" in problem.x_names
        # compute INPUT parameters as difference to mean
        for ix, xname in enumerate(problem.x_names):
            if xname.startswith("inflate") and xname.endswith("weight"):
                xi = w[int(xname.split("_")[1])]
            else:
                xi = means[xname]
            lb, ub, _ = parameter_boundaries_scales[xname.split("_")[-1]]
            xs[istart, ix] = (
                xi if not np.isnan(xi) else np.random.random() * (ub - lb) + lb
            )

    return xs


'''
elif INIT == 'sampling':
    def startpoints(**kwargs):
        """
        Custom startpoint routine for cross sample pretraining. This function
        uses the results computed for the completely unconstrained problem
        where the model is just fitted to each individual sample. For each
        sample, a random local optimization result is picked. Then
        shared population parameters are computed as mean over all samples and
        sample specific input parameter are computed by substracting this mean
        from the local solution.
        """
        n_starts = kwargs['n_starts']
        lb = kwargs['lb']

        dim = lb.size
        xs = np.empty((n_starts, dim))

        for istart in range(n_starts):
            for ix, xname in enumerate(
                    problem.get_reduced_vector(np.asarray(problem.x_names),
                                               problem.x_free_indices)
            ):

                xs[istart, ix] = np.random.random()*(ub-lb) + lb

        return xs
'''

optimizer = FidesOptimizer(
    hessian_update=fides.HybridFixed(),
    options={
        fides.Options.FATOL: 0,
        fides.Options.FRTOL: 0,
        fides.Options.XTOL: 1e-8,
        fides.Options.MAXTIME: 3600 * 10,
        fides.Options.MAXITER: 1e3,
    },
)
np.random.seed(JOB)
result = pretrain(problem, startpoints, 1, optimizer)
store_and_plot_pretraining(
    result, outdir=outdir, prefix=output_prefix, plot_waterfall=False
)
