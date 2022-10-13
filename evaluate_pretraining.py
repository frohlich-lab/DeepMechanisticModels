import sys
import pandas as pd
import numpy as np
import itertools as itt
import matplotlib.pyplot as plt
import seaborn as sns

from pypesto.store import OptimizationResultHDF5Reader
from pypesto import OptimizeResult, Result
from pypesto.C import MODE_RES
from pypesto.visualize import waterfall
from petab import get_simulation_conditions
from amici.petab_objective import rdatas_to_simulation_df

from mEncoder.autoencoder import MechanisticAutoEncoder
from process_data import training_samples, Wildcards
from mEncoder.pretraining import (
    generate_cross_sample_pretraining_problem,
    generate_per_sample_pretraining_problems
)
from mEncoder.petab_subproblem import load_petab
from mEncoder import (
    pretrain_dir, data_dir, fig_dir, apply_objective_settings,
    ESTIMATION_OUTFILE_TEMP
)
from mEncoder.plotting import plot_cross_samples
from mEncoder.analysis import process_simulation
from training_configuration import ALPHAS, HIDDEN_LAYERS

from pathlib import Path

MODEL = sys.argv[1]
DATA = sys.argv[2]
SAMPLES = sys.argv[3]

samples = training_samples(Wildcards(DATA, SAMPLES))

outdir = fig_dir / MODEL / DATA
indir = pretrain_dir / MODEL / DATA

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
            df = pd.read_csv(indir / f'{sample}.csv', index_col=[0])
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
        return indir / f'{output_prefix}.hdf5'


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

    x_inner = problem_cross_sample.objective.infun(result.list[0]['x'])

    conditions = get_simulation_conditions(
        mae.petab_importer.petab_problem.measurement_df
    )

    if alpha != 0.0:
        obj, obj_prior = problem_cross_sample.objective.base_objective._objectives
        chi2prior = obj_prior(x_inner, mode=MODE_RES, return_dict=True)['chi2']
    else:
        obj = problem_cross_sample.objective.base_objective
        chi2prior = 0.0

    res = obj(x_inner, mode=MODE_RES, return_dict=True)

    for sample in samples:
        process_simulation(evaluations, res, conditions, sample,
                           'cross_sample', alpha, hidden_layers)

    simulation_df = rdatas_to_simulation_df(
        res['rdatas'],
        model=obj.amici_model,
        measurement_df=mae.petab_importer.petab_problem.measurement_df,
    )

    plot_cross_samples(
        mae.petab_importer.petab_problem.measurement_df,
        simulation_df,
        outdir / 'pretrain_cross_samples',
        '__'.join([SAMPLES, hidden_layers, alpha])
    )

    evaluations.append({
        'chi2': chi2prior,
        'sample': 'prior',
        'type': 'cross_sample',
        'alpha': alpha,
        'layers': hidden_layers,
    })


df = pd.DataFrame(evaluations)
df.to_csv(outdir / f'{SAMPLES}_evaluate_pretraining.csv')

g = sns.FacetGrid(data=df[df['sample'].apply(lambda x: x.endswith('_dyn'))],
                  col='sample', hue='layers',
                  palette='Blues', col_wrap=5)
g.map_dataframe(sns.lineplot, x='alpha', y='chi2')
[ax.set(yscale='log') for ax in g.axes]
plt.tight_layout()
plt.savefig(outdir / f'{SAMPLES}_evaluate_pretraining.pdf')
