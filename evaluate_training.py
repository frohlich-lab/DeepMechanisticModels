import fire
import jax
import pandas as pd
import pypesto

from common import (
    Conf,
    TRAINING_OUTFILE_RESULTS,
    EVALUATION_TRAINING,
    Wildcards,
    fig_dir,
    results_dir,
    test_samples,
    training_samples,
)
from dmm.analysis import evaluate_simulations
from dmm.training import create_pypesto_problem
from dmm.initialisation import load_models
from pypesto.store import OptimizationResultHDF5Reader


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
    # TODO @GiacomoFabrini THIS NEEDS TO CHANGE - THESE EVALUATIONS ARE NOT ACTUALLY USING THE MODEL,
    #  JUST THE LINKED PETAB IMPORTER and PROBLEM...
    model, problem = load_models(conf, dataset)

    problem = create_pypesto_problem(model)

    infile = TRAINING_OUTFILE_RESULTS.format(**conf.__dict__)

    reader = OptimizationResultHDF5Reader(infile)
    result = pypesto.Result(problem)
    result.optimize_result = reader.read().optimize_result

    x = problem.objective.infun(result.optimize_result.list[0]["x"])

    obj = problem.objective.base_objective

    evaluate_simulations(
        obj=obj,
        x=x,
        samples=samples[dataset],
        petab_problem=model.petab_importer.petab_problem,
        context=conf.context,
        split=conf.samples,
        dataset=dataset,
        job=conf.job,# adding job here to produce one plot per multistart
        orth_reg_strategy=conf.orth_reg_strategy,
        l1reg_inflate=conf.l1reg_inflate,
        oreg_inflate=conf.oreg_inflate,
        l1reg_encode=conf.l1reg_encode,
        oreg_encode=conf.oreg_encode,
        latent_dim=conf.n_hidden,
        features=conf.features,
        outdir=outdir / "simulation",
        evaluations=evaluations,
        model_type="full",
    )

    return pd.DataFrame(evaluations)


for dataset in ("train", "test"):
    # clear jax cache to avoid error where jitted function uses input with shape of train
    # which differs from test
    jax.clear_caches()
    df = evaluate_training(dataset, conf)
    df.to_csv(EVALUATION_TRAINING.format(dataset=dataset, **conf.__dict__))
