import petab
import numpy as np
import pypesto.objective
import seaborn as sns
import matplotlib.pyplot as plt

from mEncoder import (
    data_dir, pretrain_dir, ESTIMATION_OUTFILE_TEMP
)
from mEncoder.autoencoder import MechanisticAutoEncoder
from mEncoder.plotting import plot_cross_samples

from pypesto.store import OptimizationResultHDF5Reader
from pypesto.objective.aesara import AesaraObjective
from pypesto.C import MODE_FUN, MODE_RES
from pypesto import OptimizeResult
from amici.petab_objective import rdatas_to_simulation_df
from petab import get_simulation_conditions
from pathlib import Path

from process_data import training_samples, test_samples, Wildcards


def process_simulation(evaluations, res, conditions, sample, model_type,
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
        llh = 0
        for ic in ics:
            chi2 += res['rdatas'][ic].chi2
            llh += res['rdatas'][ic].llh
        evaluations.append({
            'chi2': chi2,
            'llh': llh,
            'sample': f'{sample}_{name}',
            'type': model_type,
            'alpha': alpha,
            'layers': hidden_layers,
        })


def load_mae(dataset, data, model, samples, hidden_layers, alpha):
    samples_train = training_samples(Wildcards(data, samples))

    datafiles = (
        data_dir / f'{data}__{model}__measurements.tsv',
        data_dir / f'{data}__{model}__conditions.tsv',
        data_dir / f'{data}__{model}__observables.tsv',
    )

    mae_train = MechanisticAutoEncoder(
        hidden_layers, datafiles,
        pathway_name=model, samples=samples_train, l2reg=alpha
    )

    if dataset == 'train':
        return mae_train

    samples_test = test_samples(Wildcards(data, samples))

    return MechanisticAutoEncoder(
        hidden_layers, datafiles,
        pathway_name=model, samples=samples_test, l2reg=alpha,
        features=mae_train.features,
        imputer=mae_train.imputer,
        scaler=mae_train.scaler,
        pca=mae_train.pca
    )


def result_file_pretraining_cross_sample(
        job_id, model, data, samples, hidden_layers, alpha
) -> Path:
    indir = pretrain_dir / model / data
    output_prefix = Path(ESTIMATION_OUTFILE_TEMP.format(
        samples=samples, n_hidden=hidden_layers, alpha=alpha, job=job_id
    )).stem
    return indir / f'{output_prefix}.hdf5'


def load_optimize_result_pretraining_cross_samples(
        model, data, samples, hidden_layers, alpha
):
    result = OptimizeResult()
    for job in range(100):
        rfile = result_file_pretraining_cross_sample(
            job, model, data, samples, hidden_layers, alpha
        )
        if not rfile.exists():
            continue
        r = OptimizationResultHDF5Reader(
            str(rfile)
        ).read().optimize_result.list[0]
        r['id'] = str(job)
        result.append(r)

    if result.list is not None:
        result.sort()

    return result


def evaluate_simulations(obj, x, samples, petab_problem, SAMPLES, dataset,
                         l2reg, latent_dim, outdir, evaluations):
    if isinstance(obj, AesaraObjective):
        res = obj(x, mode=MODE_FUN, return_dict=True)
        inner_obj = obj.base_objective
    else:
        res = obj(x, mode=MODE_RES, return_dict=True)
        inner_obj = obj

    conditions = get_simulation_conditions(
        petab_problem.measurement_df
    )

    for sample in samples:
        process_simulation(evaluations, res, conditions, sample,
                           'cross_sample', l2reg, latent_dim)

    if isinstance(inner_obj, pypesto.objective.AggregatedObjective):
        amici_model = inner_obj._objectives[0].amici_model
    else:
        amici_model = inner_obj.amici_model

    simulation_df = rdatas_to_simulation_df(
        res['rdatas'],
        model=amici_model,
        measurement_df=petab_problem.measurement_df,
    )

    plot_cross_samples(
        petab_problem.measurement_df,
        simulation_df,
        outdir,
        '__'.join([SAMPLES, str(latent_dim), str(l2reg), dataset])
    )


def plot_loss_vs_regularization(df):
    g = sns.FacetGrid(
        data=df[df['sample'].apply(lambda x: x.endswith('_dyn'))],
        col='sample', hue='layers', palette='Blues', col_wrap=5
    )
    g.map_dataframe(sns.lineplot, x='alpha', y='chi2')
    [ax.set(yscale='log') for ax in g.axes]
    plt.tight_layout()
