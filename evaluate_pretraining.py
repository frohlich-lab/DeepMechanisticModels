import sys
import pandas as pd
import numpy as np
import itertools as itt
import matplotlib.pyplot as plt
import petab

from amici.petab_objective import rdatas_to_simulation_df
from pypesto import Result
from pypesto.visualize import waterfall

from mEncoder.pretraining import (
    generate_cross_sample_pretraining_problem, generate_per_sample_pretraining_problems,
)
from mEncoder.petab_subproblem import load_petab
from mEncoder.analysis import (
    process_simulation, plot_loss_vs_regularization, load_optimize_result_pretraining_cross_samples,
    evaluate_simulations,
)
from mEncoder.plotting import plot_single_sample, plot_cross_samples
from common import (
    training_samples, test_samples, Wildcards, pretrain_dir, data_dir, fig_dir, tpl_evaluation_file,
    CROSS_SAMPLE_OUTFILE_RESULTS, MEASUREMENTS_FILE, CONDITIONS_FILE, OBSERVABLES_FILE
)
from util import load_petab_base_files, load_mae
from cytof.problem import CytofProblem

from training_configuration import ALPHAS, LATENT_DIMS, CONTEXTS

from jax.config import config
config.update("jax_enable_x64", True)


MODEL = sys.argv[1]
DATA = sys.argv[2]
SAMPLES = sys.argv[3]

outdir = fig_dir / MODEL / DATA
indir = pretrain_dir / MODEL / DATA

cross_sample_dir = outdir / "pretrain_cross_sample"
cross_sample_dir.mkdir(exist_ok=True, parents=True)

samples = {
    "train": training_samples(Wildcards(DATA, SAMPLES)),
    "test": test_samples(Wildcards(DATA, SAMPLES)),
}


def evaluate_pretraining_per_sample(dataset, model, data):
    evaluations = []
    problem = CytofProblem(model)
    petab_base_files = load_petab_base_files(model, data)
    for sample in samples[dataset]:
        petab_base_importer = load_petab(
            problem,
            data,
            0.0,
            **petab_base_files,
        )

        importer = generate_per_sample_pretraining_problems(
            petab_base_importer,
            problem,
            DATA,
            sample,
        )
        problem_sample = importer.create_problem()
        df = pd.read_csv(indir / f"{sample}.csv", index_col=[0])
        problem.apply_objective_settings(problem_sample.objective)

        ress = []
        fvals = []
        for ipar in range(len(df)):
            x = problem_sample.get_reduced_vector(
                df.values[ipar, :], problem_sample.x_free_indices
            )
            res = problem_sample.objective(x, return_dict=True)
            ress.append(res)
            fvals.append(res["fval"])

        # Convert the simulation to PEtab format.
        simulation_df = rdatas_to_simulation_df(
            ress[np.argmin(fvals)]["rdatas"],
            model=problem_sample.objective.amici_model,
            measurement_df=importer.petab_problem.measurement_df,
        )
        process_simulation(
            evaluations,
            importer.petab_problem.measurement_df,
            simulation_df,
            "none",
            sample,
            "per_sample",
            0.0,
            0,
        )

        plot_single_sample(
            importer.petab_problem.measurement_df,
            simulation_df,
            outdir / "simulation" / dataset / sample,
            sample,
            "per_sample",
        )
    return pd.DataFrame(evaluations)


def evaluate_petraining_cross_sample(dataset):
    evaluations = []
    for l1reg, latent_dim, context in itt.product(ALPHAS, LATENT_DIMS, CONTEXTS):
        conf, mae, problem = load_mae(
            model=MODEL,
            data=DATA,
            context=context,
            samples=SAMPLES,
            n_hidden=latent_dim,
            alpha=l1reg,
            dataset=dataset,
        )

        problem_cross_sample = generate_cross_sample_pretraining_problem(mae, problem)
        result = load_optimize_result_pretraining_cross_samples(
            CROSS_SAMPLE_OUTFILE_RESULTS.replace('{job}', '([0-9]+)').format(**conf.__dict__)
        )

        r = Result()
        r.optimize_result = result

        waterfall(r)
        plt.tight_layout()
        plt.savefig(
            cross_sample_dir / f"{SAMPLES}_a{l1reg}_n{latent_dim}_waterfall.pdf"
        )

        x = problem_cross_sample.objective.infun(result.list[0]["x"])

        obj = problem_cross_sample.objective.base_objective

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
            "cross_sample",
        )

    return pd.DataFrame(evaluations)


def evaluate_average(dataset, model, data):
    df_meas = pd.read_csv(MEASUREMENTS_FILE.format(model=model, data=data), sep="\t", index_col=0)
    df_obs = pd.read_csv(OBSERVABLES_FILE.format(model=model, data=data), sep="\t", index_col=0)
    df_meas = df_meas[df_meas[petab.OBSERVABLE_ID].apply(lambda x: x in df_obs.index)]

    df_train = df_meas[
        df_meas[petab.PREEQUILIBRATION_CONDITION_ID].apply(
            lambda x: x in samples["train"]
        )
    ]

    df_train["condition"] = df_train[petab.SIMULATION_CONDITION_ID].apply(
        lambda x: x.split("__")[1]
    )

    avg_model = df_train.groupby([petab.OBSERVABLE_ID, petab.TIME, "condition"]).agg(
        np.nanmean
    )

    df_sim = df_meas.copy()
    for ir, r in df_meas.iterrows():
        df_sim.loc[ir, petab.MEASUREMENT] = avg_model.loc[
            (r.observableId, r.time, r[petab.SIMULATION_CONDITION_ID].split("__")[1]),
            petab.MEASUREMENT,
        ]

    df_sim[petab.SIMULATION] = df_sim[petab.MEASUREMENT]

    plot_cross_samples(df_meas, df_sim, outdir / "simulation" / dataset, "avg")

    evaluations = []

    for sample in samples[dataset]:
        process_simulation(
            evaluations, df_meas, df_sim, "none", sample, "avg", 0.0, 0.0
        )

    return pd.DataFrame(evaluations)


for dataset in ["train", "test"]:
    # average
    df = evaluate_average(dataset, MODEL, DATA)
    df.to_csv(tpl_evaluation_file.format(samples=SAMPLES, model=MODEL, data=DATA, dataset=dataset, mode='average'))

    # per sample
    df = evaluate_pretraining_per_sample(dataset, MODEL, DATA)
    df.to_csv(tpl_evaluation_file.format(samples=SAMPLES, model=MODEL, data=DATA, dataset=dataset, mode='per_sample'))

    # cross sample
    df = evaluate_petraining_cross_sample(dataset)
    df.to_csv(tpl_evaluation_file.format(samples=SAMPLES, model=MODEL, data=DATA, dataset=dataset, mode='cross_sample'))
    plot_loss_vs_regularization(df)
    plt.savefig(outdir / f"{SAMPLES}_evaluate_pretrain_cross_sample_{dataset}.pdf")
