import fire
import numpy as np
import pandas as pd
import petab

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
from dmm.plotting import plot_cross_samples, plot_single_sample
from evaluation_utils import (get_measurements_and_obervables,
                              process_per_sample_pretrain,
                              simulate_avg_model,
                              process_avg_model_simulation)
from typing import Dict
from util import load_petab_base_files


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
    max_lrate=0,
    lrate_span=0,
    lrate_decay=0,
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
        output = process_per_sample_pretrain(sample, problem, conf, indir, petab_base_files)
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

    # Simulate avg_model
    avg_model = simulate_avg_model(
        conf, indir, petab_base_files, dataset
    )

    # Prepare avg_model simulation for plotting and processing
    avg_model, df_meas = process_avg_model_simulation(avg_model, df_meas, dataset, samples)

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
