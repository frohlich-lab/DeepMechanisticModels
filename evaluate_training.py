import itertools as itt

import fire
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
from dmm.analysis import evaluate_simulations
from dmm.training import create_pypesto_problem
from util import Conf, load_models

conf = fire.Fire(Conf)

samples_training = training_samples(Wildcards(conf.data, conf.samples))
samples_test = test_samples(Wildcards(conf.data, conf.samples))

outdir = fig_dir / conf.model / conf.data
indir = results_dir / conf.model / conf.data

samples = {
    "train": training_samples(Wildcards(conf.data, conf.samples)),
    "test": test_samples(Wildcards(conf.data, conf.samples)),
}


def evaluate_training(dataset, conf):
    evaluations = []
    model, problem = load_models(conf, dataset)

    problem = create_pypesto_problem(model, problem)

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
        model.petab_importer.petab_problem,
        conf.context,
        conf.samples,
        dataset,
        conf.alpha,
        conf.n_hidden,
        outdir / "simulation",
        evaluations,
        "full",
    )

    return pd.DataFrame(evaluations)


for dataset in ("train", "test"):
    df = evaluate_training(dataset, conf)
    df.to_csv(EVALUATION_TRAINING.format(dataset=dataset, **conf.__dict__))
