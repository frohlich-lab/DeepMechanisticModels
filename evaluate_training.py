import sys
import pandas as pd
import numpy as np
import itertools as itt
import matplotlib.pyplot as plt
import seaborn as sns
import petab
from pypesto.store import OptimizationResultHDF5Reader
from pypesto import OptimizeResult, Result
from pypesto.C import MODE_RES
from pypesto.visualize import waterfall
from petab import get_simulation_conditions

from mEncoder.autoencoder import MechanisticAutoEncoder
from process_data import training_samples, test_samples, Wildcards
from mEncoder.training import create_pypesto_problem
from mEncoder.petab_subproblem import load_petab
from mEncoder import (
    pretrain_dir, data_dir, fig_dir, apply_objective_settings,
    ESTIMATION_OUTFILE_TEMP
)
from mEncoder.analysis import process_simulation
from training_configuration import ALPHAS, HIDDEN_LAYERS

from pathlib import Path

MODEL = sys.argv[1]
DATA = sys.argv[2]
SAMPLES = sys.argv[3]

samples_training = training_samples(Wildcards(DATA, SAMPLES))
samples_test = test_samples(Wildcards(DATA, SAMPLES))

outdir = fig_dir / MODEL / DATA
indir = pretrain_dir / MODEL / DATA

evaluations = []

datafiles = (
    data_dir / f'{DATA}__{MODEL}__measurements.tsv',
    data_dir / f'{DATA}__{MODEL}__conditions.tsv',
    data_dir / f'{DATA}__{MODEL}__observables.tsv',
)

for dataset, samples in zip(
    ('train', 'test'),
    (samples_training, samples_test)
):
    for alpha, hidden_layers in itt.product(ALPHAS, HIDDEN_LAYERS):
        # par_modulation_scale = 0 => deactivate prior
        mae = MechanisticAutoEncoder(
            hidden_layers, datafiles,
            pathway_name=MODEL, samples=samples, par_modulation_scale=0.0
        )
        problem = create_pypesto_problem(mae)
        apply_objective_settings(problem, MODEL)

        infile = result_path / COLLECTED_ESTIMATION_OUTFILE_TEMP.format(
            samples=SAMPLES, n_hidden=hidden_layers, alpha=alpha
        )

        reader = OptimizationResultHDF5Reader(infile)
        result = pypesto.Result(problem)
        result.optimize_result = reader.read().optimize_result

        x = problem.get_reduced_vector(result.optimize_result.list[0]['x'],
                                       problem.x_free_indices)

        conditions = get_simulation_conditions(
            mae.petab_importer.petab_problem.measurement_df
        )

        res = obj(x, mode=MODE_RES, return_dict=True)

        for sample in samples_training:
            process_simulation(evaluations, res, conditions, sample,
                               'full', alpha, hidden_layers)

    df = pd.DataFrame(evaluations)
    df.to_csv(outdir / f'{SAMPLES}_evaluate_training_{dataset}.csv')

    g = sns.FacetGrid(data=df[df['sample'].apply(lambda x: x.endswith('_dyn'))],
                      col='sample', hue='layers',
                      palette='Blues', col_wrap=5)
    g.map_dataframe(sns.lineplot, x='alpha', y='chi2')
    [ax.set(yscale='log') for ax in g.axes]
    plt.tight_layout()
    plt.savefig(outdir / f'{SAMPLES}_evaluate_training_{dataset}.pdf')
