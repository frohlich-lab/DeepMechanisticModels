import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import petab.v1 as petab
import seaborn as sns
from amici.petab import rdatas_to_simulation_df
from pypesto import OptimizeResult
from pypesto.C import MODE_RES, RDATAS
from pypesto.store import OptimizationResultHDF5Reader

from .config_options import default_attributes
from .plotting import plot_cross_samples
from .training_helper_funcs import (
    model_output_to_petab_input,
    model_output_to_petab_input_nojit,
)


def process_simulation(
    evaluations,
    measurement_df,
    simulation_df,
    conf,
    sample,
):
    # Set columns for multi-index
    cols_to_check = [
        petab.OBSERVABLE_ID,
        petab.PREEQUILIBRATION_CONDITION_ID,
        petab.TIME,
        petab.SIMULATION_CONDITION_ID,
    ]
    # Sort by same observables and subset to same cell-line/sample
    mdf = (
        measurement_df[
            measurement_df[petab.PREEQUILIBRATION_CONDITION_ID] == sample
        ]
        .sort_values(by=cols_to_check)
        .set_index(cols_to_check)
    )
    sdf = (
        simulation_df[
            simulation_df[petab.PREEQUILIBRATION_CONDITION_ID] == sample
        ]
        .sort_values(by=cols_to_check)
        .set_index(cols_to_check)
    )
    # Subset to common observables (needed for regressors) -- alignment step
    sdf = sdf[sdf.index.isin(mdf.index)]

    # Compute residual dataframe
    res = mdf.copy()
    res["res"] = res[petab.MEASUREMENT] - sdf[petab.SIMULATION]
    res["sim"] = sdf[petab.SIMULATION]
    # Unpack multi-index
    res.reset_index(inplace=True)

    for _, r in res.iterrows():
        # re-defining condition in such a way that fits both avg and avg_model references and regression standards
        if len(r[petab.SIMULATION_CONDITION_ID].split("__")) > 1:
            condition = r[petab.SIMULATION_CONDITION_ID].split("__")[1]
        else:
            condition = r[petab.SIMULATION_CONDITION_ID]

        # Subset conf
        # TODO @GiacomoFabrini - are all the defaults needed?
        subset_hyperparams = default_attributes

        subset_conf_dict = {
            k: v for k, v in conf.to_dict().items() if k in subset_hyperparams
        }
        evaluations.append(
            {
                "res": r["res"],
                "sim": r["sim"],
                "obs": r[petab.MEASUREMENT],
                "sample": sample,
                "observable": r[petab.OBSERVABLE_ID],
                "condition": condition,
                "time": r[petab.TIME],
                **subset_conf_dict,
            }
        )


def load_optimize_result_pretraining_cross_samples(
    pattern: str, n_starts: int
):
    result = OptimizeResult()
    indir = Path(pattern).parent
    for file in os.listdir(indir):
        if not str(file).endswith(".hdf5"):
            continue

        m = re.match(str(Path(pattern).stem), str(file))
        if not m:
            continue

        if str(os.path.splitext(file)[0]).endswith("trace"):
            continue

        # ignore previous results with higher n_starts
        if int(str(m.group(1))) >= n_starts:
            continue

        r = (
            OptimizationResultHDF5Reader(str(indir / str(file)))
            .read()
            .optimize_result.list[0]
        )
        if r.x is None:
            continue

        r["id"] = m.group(1)
        result.append(r)

    if result.list is not None:
        result.sort()

    return result


def simulate_dmm(
    model, input_features, obj, petab_problem, conf, jit_fn: bool = True
) -> pd.DataFrame:
    # Generally use the jitted model_output_to_petab_input function
    if jit_fn:
        fn = model_output_to_petab_input
    else:
        fn = model_output_to_petab_input_nojit

    res = obj(fn(model, input_features), mode=MODE_RES, return_dict=True)

    amici_model = obj.amici_model

    try:
        simulation_df = rdatas_to_simulation_df(
            res[RDATAS],
            model=amici_model,
            measurement_df=petab_problem.measurement_df,
        )
    except Exception as e:
        print(f"Error occurred: {e}")
        # If there are NaNs in the simulation results (e.g. inf fval and zero grads during training),
        # simply return a dataframe with np.inf SIMULATION values
        if np.isnan(res["res"]).sum() > 0:
            # Create a DataFrame identical to petab_problem.measurement_df
            simulation_df = petab_problem.measurement_df.copy()
            # Rename the MEASUREMENT column to SIMULATION
            simulation_df.rename(
                columns={petab.MEASUREMENT: petab.SIMULATION}, inplace=True
            )
            # Set simulation values to np.inf to indicate that the simulation failed
            simulation_df[petab.SIMULATION] = np.inf
        else:
            # different error when trying to process simulation - raise and interrupt
            raise

    return simulation_df


def evaluate_simulations(
    model,
    input_features,
    obj,
    conf,
    samples,
    petab_problem,
    dataset,
    outdir,
    evaluations,
    plot_file_prefix: str,
):
    simulation_df = simulate_dmm(
        model, input_features, obj, petab_problem, conf, jit_fn=False
    )

    for sample in samples:
        process_simulation(
            evaluations=evaluations,
            measurement_df=petab_problem.measurement_df,
            simulation_df=simulation_df,
            conf=conf,
            sample=sample,
        )

    plot_cross_samples(
        petab_problem.measurement_df,
        simulation_df,
        figdir=outdir / dataset,
        prefix=plot_file_prefix,
    )


def plot_loss_vs_regularization(df):
    df["cf"] = df["context"] + "_" + df["features"]
    dfa = (
        df.groupby(
            ["l1reg_inflate", "n_hidden", "cf", "sample", "job"]
        )  # keep job-level info (one rmse value per job)
        .agg({"res": lambda x: np.sqrt(np.mean(np.power(x, 2)))})
        .rename(columns={"res": "rmse"})
        .reset_index()
    )

    g = sns.FacetGrid(data=dfa, col="sample", col_wrap=5)
    g.map_dataframe(
        sns.lineplot,
        x="l1reg_inflate",
        y="rmse",
        errorbar=lambda x: (
            x.min(),
            x.max(),
        ),  # display error bars from min rmse to max rmse across jobs
        hue="cf",
        palette="rocket",
        style="n_hidden",
        markers=True,
    )
    [ax.set(yscale="log", xscale="log") for ax in g.axes]
    plt.tight_layout()
