import sys
import pandas as pd
import itertools as itt
import matplotlib.pyplot as plt
import pypesto
from pypesto.store import OptimizationResultHDF5Reader


from process_data import training_samples, test_samples, Wildcards
from mEncoder.training import create_pypesto_problem
from mEncoder import (
    results_dir, data_dir, fig_dir,
    COLLECTED_ESTIMATION_OUTFILE_TEMP
)
from mEncoder.analysis import (
    load_mae, plot_loss_vs_regularization, evaluate_simulations
)
from training_configuration import ALPHAS, HIDDEN_LAYERS

MODEL = sys.argv[1]
DATA = sys.argv[2]
SAMPLES = sys.argv[3]

samples_training = training_samples(Wildcards(DATA, SAMPLES))
samples_test = test_samples(Wildcards(DATA, SAMPLES))

outdir = fig_dir / MODEL / DATA
indir = results_dir / MODEL / DATA
train_dir = outdir / 'full'

train_dir.mkdir(exist_ok=True, parents=True)


datafiles = (
    data_dir / f'{DATA}__{MODEL}__measurements.tsv',
    data_dir / f'{DATA}__{MODEL}__conditions.tsv',
    data_dir / f'{DATA}__{MODEL}__observables.tsv',
)

samples = {
    'train': training_samples(Wildcards(DATA, SAMPLES)),
    'test': test_samples(Wildcards(DATA, SAMPLES)),
}


def evaluate_training(dataset):
    evaluations = []
    for l2reg, latent_dim in itt.product(ALPHAS, HIDDEN_LAYERS):
        mae = load_mae(dataset, DATA, MODEL, SAMPLES, latent_dim, l2reg)
        problem = create_pypesto_problem(mae)

        infile = indir / COLLECTED_ESTIMATION_OUTFILE_TEMP.format(
            samples=SAMPLES, n_hidden=latent_dim, alpha=l2reg
        )

        reader = OptimizationResultHDF5Reader(infile)
        result = pypesto.Result(problem)
        result.optimize_result = reader.read().optimize_result

        x = problem.objective.infun(result.optimize_result.list[0]['x'])

        obj = problem.objective.base_objective

        evaluate_simulations(
            obj, x, samples, mae.petab_importer.petab_problem,
            SAMPLES, dataset, l2reg, latent_dim, outdir / 'simulation',
            evaluations, 'full'
        )

    return pd.DataFrame(evaluations)


for dataset in ('train', 'test'):
    df = evaluate_training(dataset)
    df.to_csv(outdir / f'{SAMPLES}_evaluate_training_{dataset}.csv')
    df.to_csv(outdir / f'_{SAMPLES}_evaluate_training_{dataset}.csv')
    plot_loss_vs_regularization(df)
    plt.savefig(outdir / f'{SAMPLES}_evaluate_training_{dataset}.pdf')
