import sys
import pandas as pd
import numpy as np
import itertools as itt
import matplotlib.pyplot as plt
import petab

from petab import get_simulation_conditions
from amici.petab_objective import rdatas_to_simulation_df

from process_data import training_samples, test_samples, Wildcards
from mEncoder.pretraining import (
    generate_cross_sample_pretraining_problem,
    generate_per_sample_pretraining_problems
)
from mEncoder.petab_subproblem import load_petab
from mEncoder import (
    pretrain_dir, data_dir, fig_dir, apply_objective_settings,
)
from mEncoder.analysis import (
    process_simulation, load_mae, plot_loss_vs_regularization,
    load_optimize_result_pretraining_cross_samples, evaluate_simulations
)
from mEncoder.plotting import plot_single_sample, plot_cross_samples
from training_configuration import ALPHAS, HIDDEN_LAYERS

MODEL = sys.argv[1]
DATA = sys.argv[2]
SAMPLES = sys.argv[3]

outdir = fig_dir / MODEL / DATA
indir = pretrain_dir / MODEL / DATA

datafiles = (
    data_dir / f'{DATA}__{MODEL}__measurements.tsv',
    data_dir / f'{DATA}__{MODEL}__conditions.tsv',
    data_dir / f'{DATA}__{MODEL}__observables.tsv',
)

samples = {
    'train': training_samples(Wildcards(DATA, SAMPLES)),
    'test': test_samples(Wildcards(DATA, SAMPLES)),
}


def evaluate_pretraining_per_sample(dataset):
    evaluations = []

    for sample in samples[dataset]:
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
                           conditions, sample, 'per_sample', 0.0, 0)

        # Convert the simulation to PEtab format.
        simulation_df = rdatas_to_simulation_df(
            ress[np.argmin(fvals)]['rdatas'],
            model=problem_sample.objective.amici_model,
            measurement_df=importer.petab_problem.measurement_df,
        )
        plot_single_sample(importer.petab_problem.measurement_df,
                           simulation_df,
                           outdir / 'pretrain_per_sample',
                           '')
    return pd.DataFrame(evaluations)


def evaluate_petraining_cross_sample(dataset):
    evaluations = []
    for l2reg, latent_dim in itt.product(ALPHAS, HIDDEN_LAYERS):
        mae = load_mae(dataset, DATA, MODEL, SAMPLES, latent_dim, l2reg)

        problem_cross_sample = generate_cross_sample_pretraining_problem(mae)
        result = load_optimize_result_pretraining_cross_samples(
            MODEL, DATA, SAMPLES, latent_dim, l2reg
        )

        x = problem_cross_sample.objective.infun(result.list[0]['x'])

        obj = problem_cross_sample.objective.base_objective

        evaluate_simulations(
            obj, x, samples, mae.petab_importer.petab_problem, SAMPLES,
            dataset, l2reg, latent_dim, outdir / 'pretrain_cross_samples',
            evaluations
        )

    return pd.DataFrame(evaluations)


def evaluate_average(dataset):
    df_meas = pd.read_csv(
        data_dir / f'{DATA}__{MODEL}__measurements.tsv', sep='\t', index_col=0
    )
    df_obs = pd.read_csv(
        data_dir / f'{DATA}__{MODEL}__observables.tsv', sep='\t', index_col=0
    )
    df_meas = df_meas[
        df_meas[petab.OBSERVABLE_ID].apply(lambda x: x in df_obs.index)
    ]

    df_train = df_meas[
        df_meas[petab.PREEQUILIBRATION_CONDITION_ID].apply(
            lambda x: x in samples['train'])
    ]

    df_train['condition'] = df_train[petab.SIMULATION_CONDITION_ID].apply(
        lambda x: x.split('__')[1]
    )

    avg_model = df_train.groupby(
        [petab.OBSERVABLE_ID, petab.TIME, 'condition']
    ).agg(np.nanmean)

    df_sim = df_meas.copy()
    for ir, r in df_meas.iterrows():
        df_sim.loc[ir, petab.MEASUREMENT] = avg_model.loc[
            (r.observableId,
             r.time,
             r[petab.SIMULATION_CONDITION_ID].split('__')[1]),
            petab.MEASUREMENT
        ]

    df_sim[petab.SIMULATION] = df_sim[petab.MEASUREMENT]

    plot_cross_samples(df_meas, df_sim, outdir / 'average', 'avg')

    evaluations = []

    for sample in samples[dataset]:

        s = df_meas[df_meas[petab.PREEQUILIBRATION_CONDITION_ID] == sample]

        chi2 = 0
        for (observable, time, cond), m in avg_model.iterrows():
            d = s[
                (s[petab.OBSERVABLE_ID] == observable) &
                (s[petab.TIME] == time) &
                (s[petab.SIMULATION_CONDITION_ID] == f'{sample}__{cond}')
                ]
            r = (d[petab.MEASUREMENT] - m[petab.MEASUREMENT]) / d[
                petab.NOISE_PARAMETERS]
            chi2 += np.nansum(np.power(r, 2))

        evaluations.append({
            'chi2': chi2,
            'sample': f'{sample}_dyn',
            'type': 'average',
            'alpha': 0.0,
            'layers': 0,
        })

    return pd.DataFrame(evaluations)


for dataset in ['train', 'test']:
    # cross sample
    df = evaluate_petraining_cross_sample(dataset)
    df.to_csv(
        outdir / f'{SAMPLES}_evaluate_pretrain_cross_sample_{dataset}.csv'
    )
    plot_loss_vs_regularization(df)
    plt.savefig(
        outdir / f'{SAMPLES}_evaluate_pretrain_cross_sample_{dataset}.pdf'
    )

    # average
    df = evaluate_average(dataset)
    df.to_csv(outdir / f'{SAMPLES}_evaluate_average_{dataset}.csv')

    # per sample
    df = evaluate_pretraining_per_sample(dataset)
    df.to_csv(
        outdir / f'{SAMPLES}_evaluate_pretrain_per_sample_{dataset}.csv'
    )
