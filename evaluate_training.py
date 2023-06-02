import itertools as itt
import sys

import matplotlib.pyplot as plt
import pandas as pd
import pypesto
from pypesto.store import OptimizationResultHDF5Reader

from common import (
    COLLECTED_TRAINING_RESULTS,
    EVALUATION_TRAINING,
    Wildcards,
    fig_dir,
    results_dir,
    test_samples,
    training_samples,
)
from mEncoder.analysis import evaluate_simulations, plot_loss_vs_regularization
from mEncoder.training import create_pypesto_problem
from training_configuration import ALPHAS, CONTEXTS, LATENT_DIMS
from util import load_mae

MODEL = sys.argv[1]
DATA = sys.argv[2]
SAMPLES = sys.argv[3]

samples_training = training_samples(Wildcards(DATA, SAMPLES))
samples_test = test_samples(Wildcards(DATA, SAMPLES))

outdir = fig_dir / MODEL / DATA
indir = results_dir / MODEL / DATA

samples = {
    "train": training_samples(Wildcards(DATA, SAMPLES)),
    "test": test_samples(Wildcards(DATA, SAMPLES)),
}


def evaluate_training(dataset):
    evaluations = []
    for l1reg, latent_dim, context in itt.product(
        ALPHAS, LATENT_DIMS, CONTEXTS
    ):
        conf, mae, problem = load_mae(
            model=MODEL,
            data=DATA,
            context=context,
            samples=SAMPLES,
            n_hidden=latent_dim,
            alpha=l1reg,
            dataset=dataset,
        )

        problem = create_pypesto_problem(mae, problem)

        infile = COLLECTED_TRAINING_RESULTS.format(**conf.__dict__)

        reader = OptimizationResultHDF5Reader(infile)
        result = pypesto.Result(problem)
        result.optimize_result = reader.read().optimize_result

        x = problem.objective.infun(result.optimize_result.list[0]["x"])

        obj = problem.objective.base_objective

        evaluate_simulations(
            obj,
            x,
            samples[dataset],
            mae.petab_importer.petab_problem,
            context,
            SAMPLES,
            dataset,
            l1reg,
            latent_dim,
            outdir / "simulation",
            evaluations,
            "full",
        )

    return pd.DataFrame(evaluations)


for dataset in ("train", "test"):
    df = evaluate_training(dataset)
    df.to_csv(
        EVALUATION_TRAINING.format(
            dataset=dataset, model=MODEL, data=DATA, samples=SAMPLES
        )
    )
    plot_loss_vs_regularization(df)
    plt.savefig(outdir / f"{SAMPLES}_evaluate_training_{dataset}.pdf")
