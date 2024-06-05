import fire
import numpy as np
import pandas as pd
import petab

from amici.petab_objective import rdatas_to_simulation_df
from common import (
    Conf,
    EVALUATION_REFERENCE,
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
from evaluation_utils import get_measurements_and_obervables
from typing import Dict
from util import load_petab_base_files


def process_per_sample_pretrain(
        sample: str,
        problem,
        conf: Conf,
        petab_base_files: Dict[str, pd.DataFrame]
):
    rfile = indir / f"{sample}.csv"
    if not rfile.exists():
        return None

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
    return importer, simulation_df


conf = fire.Fire(Conf)

outdir = fig_dir / conf.model / conf.data
indir = pretrain_dir / conf.model / conf.data

# cross_sample_dir = outdir / "pretrain_cross_sample"
# cross_sample_dir.mkdir(exist_ok=True, parents=True)

# TODO @GiacomoFabrini: NEED TO CHANGE "train" to encompass "train" and "validation" (currently called
#  "test") from the splits. Change "test" to be the untouched "test" set. This is to ensure
#  that MultiTaskLassoCV and MultiTaskElasticNetCV have the same learning opportunities in
#  CV than the full DMM (i.e. their CV should be performed on train+val, not on train only)
samples = {
    "train": training_samples(Wildcards(conf.data, conf.samples)),
    "test": test_samples(Wildcards(conf.data, conf.samples)),
}

# instantiate a replacement conf for references,
# only setting to 0 those parameters that are not already 0 by default
ref_conf = Conf(
    model=conf.model,
    data=conf.data,
    max_lrate=None,
    lrate_span=None,
    lrate_decay=None,
)


def evaluate_pretraining_per_sample(
        dataset: str,
        conf: Conf,
        samples: dict,
        petab_base_files: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    evaluations = []
    problem = CytofProblem(conf.model)

    # dictionary of samples - standard behaviour for `evaluate_reference`
    for sample in samples[dataset]:
        output = process_per_sample_pretrain(sample, problem, conf, petab_base_files)
        if output is None:
            # file not found
            continue
        importer, simulation_df = output
        process_simulation(
            evaluations=evaluations,
            measurement_df=importer.petab_problem.measurement_df,
            simulation_df=simulation_df,
            conf=ref_conf,
            sample=sample,
            model_type="per_sample",
        )

        plot_single_sample(
            importer.petab_problem.measurement_df,
            simulation_df,
            outdir / "simulation" / dataset / sample,
            sample,
            "per_sample",
        )
    return pd.DataFrame(evaluations)


def evaluate_average(
        dataset: str,
        conf: Conf,
        samples: dict,
) -> pd.DataFrame:
    df_meas, df_obs = get_measurements_and_obervables(conf)

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
        # pick the closest time point to avoid issues with non-canonical time points
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
            evaluations=evaluations,
            measurement_df=df_meas,
            simulation_df=df_sim,
            conf=ref_conf,
            sample=sample,
            model_type="avg",
        )
    return pd.DataFrame(evaluations)


def evaluate_average_model(
        dataset: str,
        conf: Conf,
        samples: dict,
        petab_base_files: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    df_meas, df_obs = get_measurements_and_obervables(conf)

    problem = CytofProblem(conf.model)
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
            evaluations=evaluations,
            measurement_df=df_meas,
            simulation_df=avg_model,
            conf=ref_conf,
            sample=sample,
            model_type="avg_model",
        )
    return pd.DataFrame(evaluations)


# Get petab_base_files
petab_base_files = load_petab_base_files(conf)

# Evaluate references/baselines
for dataset in ["train", "test"]:
    # model average ("avg_model")
    df = evaluate_average_model(dataset, conf, samples, petab_base_files)
    df.to_csv(
        EVALUATION_REFERENCE.format(
            **conf.__dict__,
            dataset=dataset,
            mode="avg_model",
        )
    )

    # average -- this looks NOT to be in use at the moment (only avg_model)
    # df = evaluate_average(dataset, conf, samples)
    # df.to_csv(
    #     EVALUATION_REFERENCE.format(
    #         **conf.__dict__,
    #         dataset=dataset,
    #         mode="average",
    #     )
    # )

    # per sample ("sample")
    df = evaluate_pretraining_per_sample(dataset, conf, samples, petab_base_files)
    df.to_csv(
        EVALUATION_REFERENCE.format(
            **conf.__dict__,
            dataset=dataset,
            mode="per_sample",
        )
    )
