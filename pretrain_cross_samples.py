"""
Pretraining of population + individual parameters based on per sample
pretraining
"""

import sys

import pandas as pd
import numpy as np
import scipy.linalg as la
import fides

from pypesto.optimize import FidesOptimizer
from pathlib import Path

from mEncoder.pretraining import generate_cross_sample_pretraining_problem, pretrain, store_and_plot_pretraining
from mEncoder import MODEL_FEATURE_PREFIX
from common import  CROSS_SAMPLE_OUTFILE_PARS, CROSS_SAMPLE_OUTFILE_RESULTS, PER_SAMPLE_OUTFILE_PARS
from util import load_from_argv

from jax.config import config
config.update("jax_enable_x64", True)
# config.update("jax_disable_jit", True)

conf, mae, problem = load_from_argv(sys.argv, dataset='train', n_threads=4)

pypesto_problem = generate_cross_sample_pretraining_problem(mae, problem)
pretrained_samples = {}

for sample in mae.sample_names:
    df = pd.read_csv(PER_SAMPLE_OUTFILE_PARS.format(**conf.__dict__, sample=sample), index_col=[0])
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
        assert f"inflate_{len(w)-1}_weight" in pypesto_problem.x_names
        # compute INPUT parameters as difference to mean
        for ix, xname in enumerate(pypesto_problem.x_names):
            if xname.startswith("inflate") and xname.endswith("weight"):
                xi = w[int(xname.split("_")[1])]
            else:
                xi = means[xname]
            lb, ub, _ = problem.bounds[xname.split("_")[-1]]
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
        fides.Options.MAXITER: 25,
    },
)
np.random.seed(conf.job)
result = pretrain(pypesto_problem, startpoints, 1, optimizer)

rfile = Path(CROSS_SAMPLE_OUTFILE_RESULTS.format(**conf.__dict__))
pfile = Path(CROSS_SAMPLE_OUTFILE_PARS.format(**conf.__dict__))
store_and_plot_pretraining(
    result, pfile=pfile, rfile=rfile, plot_waterfall=False
)
