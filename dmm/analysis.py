import jax
import os
import re
from pathlib import Path

import amici
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import petab
import pypesto.objective
import seaborn as sns
from amici.petab_objective import rdatas_to_simulation_df
from pypesto import OptimizeResult
from pypesto.C import MODE_RES
from pypesto.store import OptimizationResultHDF5Reader

from dmm.plotting import plot_cross_samples
from dmm.training import model_output_to_petab_input


def process_simulation(
    evaluations,
    measurement_df,
    simulation_df,
    context,
    job,
    sample,
    model_type,
    orth_reg_strategy,
    l1reg_inflate,
    oreg_inflate,
    l1reg_encode,
    oreg_encode,
    hidden_layers,
    features,
):
    idx = measurement_df[petab.PREEQUILIBRATION_CONDITION_ID] == sample
    mdf = measurement_df[idx]
    sdf = simulation_df[idx]

    res = mdf.copy()
    res[petab.MEASUREMENT] -= sdf[petab.SIMULATION]

    for _, r in res.iterrows():
        # re-defining condition in such a way that fits both avg and avg_model references and regression standards
        if len(r[petab.SIMULATION_CONDITION_ID].split("__")) > 1:
            condition = r[petab.SIMULATION_CONDITION_ID].split("__")[1]
        else:
            condition = r[petab.SIMULATION_CONDITION_ID]
        evaluations.append(
            {
                "res": r[petab.MEASUREMENT],
                "job": job,  # job-specific residuals
                "sample": sample,
                "type": model_type,
                "context": context,
                "orth_reg_strategy": orth_reg_strategy,
                "l1reg_inflate": l1reg_inflate,
                "oreg_inflate": oreg_inflate,
                "l1reg_encode": l1reg_encode,
                "oreg_encode": oreg_encode,
                "layers": hidden_layers,
                "features": features,
                "observable": r[petab.OBSERVABLE_ID],
                "condition": condition,
                "time": r[petab.TIME],
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


def evaluate_simulations(
    model,
    input_features,
    obj,
    samples,
    petab_problem,
    context,
    split,
    dataset,
    job,
    orth_reg_strategy,
    l1reg_inflate,
    oreg_inflate,
    l1reg_encode,
    oreg_encode,
    latent_dim,
    features,
    outdir,
    evaluations,
    model_type,
):
    res = obj(
        model_output_to_petab_input(model, input_features),
        mode=MODE_RES,
        return_dict=True
    )

    if isinstance(obj, pypesto.objective.AggregatedObjective):
        amici_model = obj._objectives[0].amici_model
        amici_solver = obj._objectives[0].amici_solver
    else:
        amici_model = obj.amici_model
        amici_solver = obj.amici_solver

    for r in res["rdatas"]:
        if r["status"] != amici.AMICI_SUCCESS:
            print(f'AMICI failed for {r["id"]}')
            x = jnp.ones((1,), dtype=jnp.float64)
            print(f"JAX dtype: {x.dtype} ")
            print(
                f"AMICI solver options: {amici_solver.getAbsoluteTolerance():.2e} atol, "
                f"{amici_solver.getRelativeTolerance():.2e} rtol"
            )
            return

    simulation_df = rdatas_to_simulation_df(
        res["rdatas"],
        model=amici_model,
        measurement_df=petab_problem.measurement_df,
    )

    for sample in samples:
        process_simulation(
            evaluations=evaluations,
            measurement_df=petab_problem.measurement_df,
            simulation_df=simulation_df,
            context=context,
            job=job,
            sample=sample,
            model_type=model_type,
            orth_reg_strategy=orth_reg_strategy,
            l1reg_inflate=l1reg_inflate,
            oreg_inflate=oreg_inflate,
            l1reg_encode=l1reg_encode,
            oreg_encode=oreg_encode,
            hidden_layers=latent_dim,
            features=features,
        )

    plot_cross_samples(
        petab_problem.measurement_df,
        simulation_df,
        outdir / dataset,
        "__".join(
            [
                split,
                context,
                str(job),  # include job number to produce one plot per multistart
                str(latent_dim),
                str(l1reg_inflate),
                dataset,
                model_type,
            ]
        ),
    )


def plot_loss_vs_regularization(df):
    df["cf"] = df["context"] + "_" + df["features"]
    dfa = (
        df.groupby(["l1reg_inflate", "layers", "cf", "sample", "job"])  # keep job-level info (one rmse value per job)
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
        style="layers",
        markers=True,
    )
    [ax.set(yscale="log", xscale="log") for ax in g.axes]
    plt.tight_layout()
