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
from process_data import training_samples, Wildcards
from mEncoder.pretraining import (
    generate_cross_sample_pretraining_problem,
    generate_per_sample_pretraining_problems
)
from mEncoder.petab_subproblem import load_petab
from mEncoder import (
    pretrain_dir, data_dir, apply_objective_settings, ESTIMATION_OUTFILE_TEMP
)
from training_configuration import ALPHAS, HIDDEN_LAYERS

from pathlib import Path


def process_simulation(evaluations, res, conditions, sample, type,
                       alpha, hidden_layers):
    splits = {
        'dyn': (conditions[petab.PREEQUILIBRATION_CONDITION_ID] == sample) &
        (conditions[petab.SIMULATION_CONDITION_ID] != sample),
        'stat': (conditions[petab.PREEQUILIBRATION_CONDITION_ID] == sample) &
        (conditions[petab.SIMULATION_CONDITION_ID] == sample),
    }
    for name, split in splits.items():
        ics = np.where(split)[0]
        chi2 = 0
        for ic in ics:
            chi2 += res['rdatas'][ic].chi2
        evaluations.append({
            'chi2': chi2,
            'sample': f'{sample}_{name}',
            'type': type,
            'alpha': alpha,
            'layers': hidden_layers,
        })


MODEL = sys.argv[1]
DATA = sys.argv[2]
SAMPLES = sys.argv[3]

samples = training_samples(Wildcards(DATA, SAMPLES))

outdir = pretrain_dir / MODEL / DATA

evaluations = []

datafiles = (
    data_dir / f'{DATA}__{MODEL}__measurements.tsv',
    data_dir / f'{DATA}__{MODEL}__conditions.tsv',
    data_dir / f'{DATA}__{MODEL}__observables.tsv',
)

for alpha, hidden_layers in itt.product(ALPHAS, HIDDEN_LAYERS):
    mae = MechanisticAutoEncoder(
        hidden_layers, datafiles,
        pathway_name=MODEL, samples=samples, par_modulation_scale=alpha
    )

    if hidden_layers == HIDDEN_LAYERS[0]:
        for sample in samples:
            importer = generate_per_sample_pretraining_problems(
                load_petab(datafiles, 'pw_' + MODEL, 0.0, [sample]),
                MODEL, f'{DATA}__{MODEL}', sample
            )
            problem_sample = importer.create_problem()
            df = pd.read_csv(outdir / f'{sample}.csv', index_col=[0])
            apply_objective_settings(problem_sample, MODEL)

            conditions = get_simulation_conditions(
                importer.petab_problem.measurement_df
            )
            ress = []
            fvals = []
            for ipar in range(len(df)):
                x = problem_sample.get_reduced_vector(
                    df.values[ipar, :],
                    problem_sample.x_free_indices
                )
                res = problem_sample.objective(x, return_dict=True)
                ress.append(res)
                fvals.append(res['fval'])

            process_simulation(evaluations, ress[np.argmin(fvals)],
                               conditions, sample, 'per_sample', alpha, 0)

    problem_cross_sample = generate_cross_sample_pretraining_problem(mae)
    apply_objective_settings(problem_cross_sample, MODEL)

    def result_file(job_id) -> Path:
        output_prefix = Path(ESTIMATION_OUTFILE_TEMP.format(
            samples=SAMPLES, n_hidden=hidden_layers, alpha=alpha, job=job_id
        )).stem
        return outdir / f'{output_prefix}.hdf5'

    result = OptimizeResult()
    for job in range(100):
        if not result_file(job).exists():
            continue
        r = OptimizationResultHDF5Reader(str(result_file(job))).read(). \
            optimize_result.list[0]
        r['id'] = str(job)
        result.append(r)

    result.sort()
    if not result.list:
        continue

    r = Result()
    r.optimize_result = result

    waterfall(r)
    plt.tight_layout()
    plt.savefig(
        outdir /
        f'{SAMPLES}_pretraining_cross_sample_a{alpha}_n{hidden_layers}_waterfall.pdf'
    )

    res = problem_cross_sample.objective(result.list[0]['x'],
                                         return_dict=True)

    conditions = get_simulation_conditions(
        mae.petab_importer.petab_problem.measurement_df
    )

    for sample in samples:
        process_simulation(evaluations, res, conditions, sample,
                           'cross_sample', alpha, hidden_layers)

    x_inner = problem_cross_sample.objective.infun(result.list[0]['x'])
    obj = problem_cross_sample.objective.base_objective._objectives[0]
    chi2prior = obj(x_inner, mode=MODE_RES, return_dict=True)['chi2']

    evaluations.append({
        'chi2': chi2prior,
        'sample': 'prior',
        'type': 'cross_sample',
        'alpha': alpha,
        'layers': hidden_layers,
    })

df = pd.DataFrame(evaluations)
df.to_csv(outdir / f'{SAMPLES}_evaluation_pretraining.csv')

g = sns.FacetGrid(data=df, col='sample', hue='layers', hue_order=HIDDEN_LAYERS,
                  palette='Blues', col_wrap=5)
g.map_dataframe(sns.lineplot, x='alpha', y='chi2').set(yscale='log')
plt.tight_layout()
plt.savefig(outdir / f'{SAMPLES}_evaluate_pretraining.pdf')
