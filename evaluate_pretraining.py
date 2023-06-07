import fire
import matplotlib.pyplot as plt
import pandas as pd
from pypesto import Result
from pypesto.visualize import parameters, waterfall

from common import (
    CROSS_SAMPLE_OUTFILE_RESULTS,
    EVALUATION_PRETRAINING,
    Wildcards,
    fig_dir,
    pretrain_dir,
    test_samples,
    training_samples,
)
from dmm.analysis import (
    evaluate_simulations,
    load_optimize_result_pretraining_cross_samples,
)
from dmm.pretraining import generate_cross_sample_pretraining_problem
from util import Conf, load_models

conf = fire.Fire(Conf)

outdir = fig_dir / conf.model / conf.data
indir = pretrain_dir / conf.model / conf.data

cross_sample_dir = outdir / "pretrain_cross_sample"
cross_sample_dir.mkdir(exist_ok=True, parents=True)

samples = {
    "train": training_samples(Wildcards(conf.data, conf.samples)),
    "test": test_samples(Wildcards(conf.data, conf.samples)),
}


def evaluate_petraining_cross_sample(dataset, conf):
    evaluations = []
    model, problem = load_models(conf, dataset)

    problem_cross_sample = generate_cross_sample_pretraining_problem(
        model, problem
    )
    result = load_optimize_result_pretraining_cross_samples(
        CROSS_SAMPLE_OUTFILE_RESULTS.replace("{job}", "([0-9]+)").format(
            **conf.__dict__
        ),
        conf.n_starts,
    )

    r = Result(problem=problem_cross_sample)
    r.optimize_result = result

    waterfall(r)
    plt.tight_layout()
    run_name = f"{conf.samples}_a{conf.alpha}_n{conf.n_hidden}_c{conf.context}"
    plt.savefig(cross_sample_dir / f"{run_name}_waterfall.pdf")
    parameters(r)
    plt.tight_layout()
    plt.savefig(cross_sample_dir / f"{run_name}_parameters.pdf")

    x = problem_cross_sample.objective.infun(result.list[0]["x"])

    obj = problem_cross_sample.objective.base_objective

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
        "cross_sample",
    )

    return pd.DataFrame(evaluations)


for dataset in ["train", "test"]:
    # cross sample
    df = evaluate_petraining_cross_sample(dataset, conf)
    df.to_csv(
        EVALUATION_PRETRAINING.format(
            **conf.__dict__,
            dataset=dataset,
            mode="cross_sample",
        )
    )
