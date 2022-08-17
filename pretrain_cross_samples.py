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
    generate_cross_sample_pretraining_problem, pretrain,
    store_and_plot_pretraining
)
from mEncoder import (
    MODEL_FEATURE_PREFIX, apply_objective_settings,
    parameter_boundaries_scales, pretrain_dir, data_dir, fig_dir,
    ESTIMATION_OUTFILE_TEMP
)

MODEL = sys.argv[1]
DATA = sys.argv[2]
SAMPLES = sys.argv[3]
N_HIDDEN = int(sys.argv[4])
ALPHA = float(sys.argv[5])
JOB = int(sys.argv[6])

samples = training_samples(Wildcards(DATA, SAMPLES))
mae = MechanisticAutoEncoder(
    N_HIDDEN, (
        data_dir / f'{DATA}__{MODEL}__measurements.tsv',
        data_dir / f'{DATA}__{MODEL}__conditions.tsv',
        data_dir / f'{DATA}__{MODEL}__observables.tsv',
    ),
    pathway_name=MODEL, samples=samples, par_modulation_scale=ALPHA
)

problem = generate_cross_sample_pretraining_problem(mae)
pretrained_samples = {}

outdir = pretrain_dir / MODEL / DATA
figdir = fig_dir / MODEL / DATA / 'pretraining_cross'
output_prefix = os.path.splitext(ESTIMATION_OUTFILE_TEMP.format(
    samples=SAMPLES, n_hidden=N_HIDDEN, alpha=ALPHA, job=JOB
))[0]

for sample in samples:
    df = pd.read_csv(
        pretrain_dir / MODEL / DATA / f'{sample}.csv',
        index_col=[0]
    )
    pretrained_samples[sample] = df[[
        col for col in df.columns
        if not col.startswith(MODEL_FEATURE_PREFIX)
    ]]


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
        # use parameter values from random start for each sample
        par_combo = pd.concat([
            pretraining[
                pretraining.index == np.min([np.random.poisson(2, 1)[0],
                                             len(pretraining)-1])
            ]
            for pretraining in pretrained_samples.values()
        ])
        par_combo.index = samples
        means = par_combo.mean(skipna=True)
        par_combo -= means
        inputs = [
            '__'.join(p.split('__')[:-1]).replace(MODEL_FEATURE_PREFIX, '')
            for p in mae.petab_importer.petab_problem.parameter_df.index
            if p.startswith(MODEL_FEATURE_PREFIX)
            and p.endswith(par_combo.index[0])
        ]
        w = la.lstsq(mae.data_pca, par_combo[inputs].values)[0].flatten()
        # compute INPUT parameters as as difference to mean
        for ix, xname in enumerate(
                problem.get_reduced_vector(np.asarray(problem.x_names),
                                           problem.x_free_indices)
        ):
            if xname.startswith('inflate') and xname.endswith('weight'):
                xi = w[int(xname.split('_')[1])]
            else:
                xi = means[xname]
            lb, ub, _ = parameter_boundaries_scales[
                xname.split('_')[-1]
            ]
            xs[istart, ix] = xi if not np.isnan(xi) else \
                np.random.random() * (ub - lb) + lb

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

apply_objective_settings(problem, MODEL)

optimizer = FidesOptimizer(
    hessian_update=fides.HybridFixed(),
    options={
        fides.Options.FATOL: 0,
        fides.Options.FRTOL: 0,
        fides.Options.XTOL: 1e-8,
        fides.Options.MAXTIME: 3600 * 10,
        fides.Options.MAXITER: 2e2,
    }
)
np.random.seed(JOB)
result = pretrain(problem, startpoints, 1, optimizer)
store_and_plot_pretraining(result, outdir=outdir, prefix=output_prefix,
                           plot_waterfall=False)
