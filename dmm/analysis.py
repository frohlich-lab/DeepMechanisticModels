import amici
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import os
import re

import pandas as pd
import petab
import pypesto.objective
import seaborn as sns

from amici.petab_objective import rdatas_to_simulation_df
from common import default_attributes
from dmm.plotting import plot_cross_samples
from dmm.training_helper_funcs import model_output_to_petab_input
from pathlib import Path
from pypesto import OptimizeResult
from pypesto.C import MODE_RES, RDATAS
from pypesto.store import OptimizationResultHDF5Reader


def process_simulation(
    evaluations,
    measurement_df,
    simulation_df,
    conf,
    sample,
    model_type,
):
    idx = measurement_df[petab.PREEQUILIBRATION_CONDITION_ID] == sample
    mdf = measurement_df[idx]
    sdf = simulation_df[idx]
    # Reindex sdf to match mdf and check
    sdf = sdf.reindex(mdf.index)
    cols_to_check = [
        petab.OBSERVABLE_ID,
        petab.PREEQUILIBRATION_CONDITION_ID,
        petab.TIME,
        petab.SIMULATION_CONDITION_ID
    ]
    try:
        assert mdf[cols_to_check].equals(sdf[cols_to_check])
    except AssertionError:
        print("measurement and simulation dataframes are not identically ordered!")

    res = mdf.copy()
    res[petab.MEASUREMENT] -= sdf[petab.SIMULATION]

    for _, r in res.iterrows():
        # re-defining condition in such a way that fits both avg and avg_model references and regression standards
        if len(r[petab.SIMULATION_CONDITION_ID].split("__")) > 1:
            condition = r[petab.SIMULATION_CONDITION_ID].split("__")[1]
        else:
            condition = r[petab.SIMULATION_CONDITION_ID]

        # Subset conf
        # TODO @GiacomoFabrini - are all the defaults needed?
        subset_hyperparams = default_attributes

        subset_conf_dict = dict(
            (k, conf.__dict__[k])
            for k in subset_hyperparams
            if k in conf.__dict__
        )
        evaluations.append(
            {
                "res": r[petab.MEASUREMENT],
                "sample": sample,
                "type": model_type,
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
        model,
        input_features,
        obj,
        petab_problem
) -> pd.DataFrame:

    res = obj(
        model_output_to_petab_input(model, input_features),
        mode=MODE_RES,
        return_dict=True
    )

    amici_model = obj.amici_model

    # if isinstance(obj, pypesto.objective.AggregatedObjective):
    #     amici_model = obj._objectives[0].amici_model
    #     amici_solver = obj._objectives[0].amici_solver
    # else:
    #     amici_model = obj.amici_model
    #     amici_solver = obj.amici_solver

    # for r in res["rdatas"]:
    #     if r["status"] != amici.AMICI_SUCCESS:
    #         print(f'AMICI failed for {r["id"]}')
    #         x = jnp.ones((1,), dtype=jnp.float64)
    #         print(f"JAX dtype: {x.dtype} ")
    #         print(
    #             f"AMICI solver options: {amici_solver.getAbsoluteTolerance():.2e} atol, "
    #             f"{amici_solver.getRelativeTolerance():.2e} rtol"
    #         )
    #         return

    simulation_df = rdatas_to_simulation_df(
        res[RDATAS],
        model=amici_model,
        measurement_df=petab_problem.measurement_df,
    )
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
    model_type,
):
    # Simulate DMM model
    simulation_df = simulate_dmm(model, input_features, obj, petab_problem)

    for sample in samples:
        process_simulation(
            evaluations=evaluations,
            measurement_df=petab_problem.measurement_df,
            simulation_df=simulation_df,
            conf=conf,
            sample=sample,
            model_type=model_type,
        )

    plot_cross_samples(
        petab_problem.measurement_df,
        simulation_df,
        outdir / dataset,
        "__".join(
            [
                dataset,
                conf.samples,
                conf.context,
                conf.features,
                model_type,
                conf.__str__(),
            ]
        )
    )


def plot_loss_vs_regularization(df):
    df["cf"] = df["context"] + "_" + df["features"]
    dfa = (
        df.groupby(["l1reg_inflate", "n_hidden", "cf", "sample", "job"])  # keep job-level info (one rmse value per job)
        .agg({"res": lambda x: np.sqrt(np.mean(np.power(x, 2)))})
        .rename(columns={"res": "rmse"})
        .reset_index()
    )

    g = sns.FacetGrid(data=dfa, col="sample", col_wrap=5)
    g.map_dataframe(
        sns.lineplot,
        x="l1reg_inflate",
        y="rmse",
        errorbar=lambda x: (x.min(), x.max()),  # display error bars from min rmse to max rmse across jobs
        hue="cf",
        palette="rocket",
        style="n_hidden",
        markers=True,
    )
    [ax.set(yscale="log", xscale="log") for ax in g.axes]
    plt.tight_layout()
