import itertools as itt

import fire
import numpy as np
import pandas as pd
import petab
from amici.petab_objective import rdatas_to_simulation_df

from common import (
    EVALUATION_REFERENCE,
    MEASUREMENTS_FILE,
    OBSERVABLES_FILE,
    Wildcards,
    fig_dir,
    pretrain_dir,
    test_samples,
    training_samples,
)
from cytof.problem import CytofProblem
from dmm.analysis import process_simulation
from dmm.petab_subproblem import load_petab
from dmm.plotting import plot_cross_samples, plot_single_sample
from dmm.pretraining import (
    generate_average_pretraining_problem,
    generate_per_sample_pretraining_problems,
)
from util import Conf, load_petab_base_files

conf = fire.Fire(Conf)

outdir = fig_dir / conf.model / conf.data
indir = pretrain_dir / conf.model / conf.data

cross_sample_dir = outdir / "pretrain_cross_sample"
cross_sample_dir.mkdir(exist_ok=True, parents=True)

samples = {
    "train": training_samples(Wildcards(conf.data, conf.samples)),
    "test": test_samples(Wildcards(conf.data, conf.samples)),
}


def evaluate_pretraining_per_sample(dataset, conf):
    evaluations = []
    problem = CytofProblem(conf.model)
    petab_base_files = load_petab_base_files(conf)
    for sample in samples[dataset]:
        rfile = indir / f"{sample}.csv"
        if not rfile.exists():
            continue

        petab_base_importer = load_petab(
            problem,
            conf.data,
            **petab_base_files,
        )

        importer = generate_per_sample_pretraining_problems(
            petab_base_importer,
            problem,
            conf.data,
            sample,
        )

        problem_sample = importer.create_problem()
        df = pd.read_csv(rfile, index_col=[0])
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
            "none",
        )

        plot_single_sample(
            importer.petab_problem.measurement_df,
            simulation_df,
            outdir / "simulation" / dataset / sample,
            sample,
            "per_sample",
        )
    return pd.DataFrame(evaluations)


def evaluate_average(dataset, conf):
    df_meas = pd.read_csv(
        MEASUREMENTS_FILE.format(**conf.__dict__), sep="\t", index_col=0
    )
    df_obs = pd.read_csv(
        OBSERVABLES_FILE.format(**conf.__dict__), sep="\t", index_col=0
    )
    df_meas = df_meas[
        df_meas[petab.OBSERVABLE_ID].apply(lambda x: x in df_obs.index)
    ]

    df_train = df_meas[
        df_meas[petab.PREEQUILIBRATION_CONDITION_ID].isin(samples["train"])
    ]

    df_train["condition"] = df_train[petab.SIMULATION_CONDITION_ID].apply(
        lambda x: x.split("__")[1]
    )
    gb_cols = [petab.OBSERVABLE_ID, "condition", petab.TIME]
    avg_model = (
        df_train[gb_cols + [petab.MEASUREMENT]]
        .groupby(gb_cols)
        .agg(np.nanmean)
    )

    df_sim = df_meas.copy()
    df_sim = df_sim.loc[
        df_sim[petab.PREEQUILIBRATION_CONDITION_ID].isin(samples[dataset]), :
    ]

    for ir, r in df_meas.iterrows():
        # pick closest time point to avoid issues with non-canonical time points
        candidates = avg_model.loc[
            (r.observableId, r[petab.SIMULATION_CONDITION_ID].split("__")[1]),
            petab.MEASUREMENT,
        ]
        df_sim.loc[ir, petab.MEASUREMENT] = candidates.iloc[
            np.argmin(np.abs(candidates.index - r.time))
        ]

    df_sim[petab.SIMULATION] = df_sim[petab.MEASUREMENT]

    plot_cross_samples(df_meas, df_sim, outdir / "simulation" / dataset, "avg")

    evaluations = []

    for sample in samples[dataset]:
        process_simulation(
            evaluations,
            df_meas,
            df_sim,
            "none",
            sample,
            "avg",
            0.0,
            0.0,
            "none",
        )

    return pd.DataFrame(evaluations)


def evaluate_average_model(dataset, conf):
    df_meas = pd.read_csv(
        MEASUREMENTS_FILE.format(**conf.__dict__), sep="\t", index_col=0
    )
    df_obs = pd.read_csv(
        OBSERVABLES_FILE.format(**conf.__dict__), sep="\t", index_col=0
    )
    df_meas = df_meas[
        df_meas[petab.OBSERVABLE_ID].apply(lambda x: x in df_obs.index)
    ]

    problem = CytofProblem(conf.model)
    petab_base_files = load_petab_base_files(conf)
    rfile = indir / f"model_average_{conf.samples}.csv"

    petab_base_importer = load_petab(
        problem,
        conf.data,
        **petab_base_files,
    )

    importer = generate_average_pretraining_problem(
        petab_base_importer,
        problem,
        conf.data,
        training_samples(Wildcards(conf.data, conf.samples))
        if dataset == "train"
        else test_samples(Wildcards(conf.data, conf.samples)),
    )
    problem_sample = importer.create_problem()
    df = pd.read_csv(rfile, index_col=[0])
    problem.apply_objective_settings(problem_sample.objective)

    ress = []
    fvals = []
    for ipar in range(len(df)):
        x = problem_sample.get_reduced_vector(
            df.values[0, :], problem_sample.x_free_indices
        )
        res = problem_sample.objective(x, return_dict=True)
        ress.append(res)
        fvals.append(res["fval"])

    # Convert the simulation to PEtab format.
    avg_model = rdatas_to_simulation_df(
        ress[np.argmin(fvals)]["rdatas"],
        model=problem_sample.objective.amici_model,
        measurement_df=importer.petab_problem.measurement_df,
    )

    avg_model[petab.SIMULATION_CONDITION_ID] = df_meas[
        petab.SIMULATION_CONDITION_ID
    ]
    avg_model[petab.PREEQUILIBRATION_CONDITION_ID] = df_meas[
        petab.PREEQUILIBRATION_CONDITION_ID
    ]

    df_meas = df_meas[
        df_meas[petab.PREEQUILIBRATION_CONDITION_ID].isin(samples[dataset])
    ]
    avg_model = avg_model[
        avg_model[petab.PREEQUILIBRATION_CONDITION_ID].isin(samples[dataset])
    ]

    plot_cross_samples(
        df_meas, avg_model, outdir / "simulation" / dataset, "avg_model"
    )

    evaluations = []

    for sample in samples[dataset]:
        process_simulation(
            evaluations,
            df_meas,
            avg_model,
            "none",
            sample,
            "avg_model",
            0.0,
            0.0,
            "none",
        )

    return pd.DataFrame(evaluations)


for dataset in ["train", "test"]:
    # model average
    df = evaluate_average_model(dataset, conf)
    df.to_csv(
        EVALUATION_REFERENCE.format(
            **conf.__dict__,
            dataset=dataset,
            mode="avg_model",
        )
    )

    # average
    df = evaluate_average(dataset, conf)
    df.to_csv(
        EVALUATION_REFERENCE.format(
            **conf.__dict__,
            dataset=dataset,
            mode="average",
        )
    )

    # per sample
    df = evaluate_pretraining_per_sample(dataset, conf)
    df.to_csv(
        EVALUATION_REFERENCE.format(
            **conf.__dict__,
            dataset=dataset,
            mode="per_sample",
        )
    )
