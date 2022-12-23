import os
import re

import petab
import numpy as np
import pypesto.objective
import seaborn as sns
import matplotlib.pyplot as plt

from mEncoder.plotting import plot_cross_samples

from pypesto.store import OptimizationResultHDF5Reader
from pypesto.C import MODE_RES
from pypesto import OptimizeResult
from amici.petab_objective import rdatas_to_simulation_df
from pathlib import Path


def process_simulation(
    evaluations,
    measurement_df,
    simulation_df,
    context,
    sample,
    model_type,
    alpha,
    hidden_layers,
):
    idx = measurement_df[petab.PREEQUILIBRATION_CONDITION_ID] == sample
    mdf = measurement_df[idx]
    sdf = simulation_df[idx]

    res = (mdf[petab.MEASUREMENT] - sdf[petab.SIMULATION]) / mdf[petab.NOISE_PARAMETERS]

    evaluations.append(
        {
            "rmse": np.sqrt(np.power(res.values, 2).mean()),
            "sample": sample,
            "type": model_type,
            "context": context,
            "alpha": alpha,
            "layers": hidden_layers,
        }
    )


def load_optimize_result_pretraining_cross_samples(pattern):
    result = OptimizeResult()
    indir = Path(pattern).parent
    for file in os.listdir(indir):
        if not str(file).endswith('.hdf5'):
            continue
        m = re.match(str(Path(pattern).stem), str(file))
        if not m:
            continue
        r = OptimizationResultHDF5Reader(str(indir / str(file))).read().optimize_result.list[0]
        r["id"] = m.group(1)
        result.append(r)

    if result.list is not None:
        result.sort()

    return result


def evaluate_simulations(
    obj,
    x,
    samples,
    petab_problem,
    context,
    SAMPLES,
    dataset,
    l1reg,
    latent_dim,
    outdir,
    evaluations,
    model_type,
):

    res = obj(x, mode=MODE_RES, return_dict=True)

    if isinstance(obj, pypesto.objective.AggregatedObjective):
        amici_model = obj._objectives[0].amici_model
    else:
        amici_model = obj.amici_model

    simulation_df = rdatas_to_simulation_df(
        res["rdatas"],
        model=amici_model,
        measurement_df=petab_problem.measurement_df,
    )

    for sample in samples:
        process_simulation(
            evaluations,
            petab_problem.measurement_df,
            simulation_df,
            context,
            sample,
            model_type,
            l1reg,
            latent_dim,
        )

    plot_cross_samples(
        petab_problem.measurement_df,
        simulation_df,
        outdir / dataset,
        "__".join([SAMPLES, context, str(latent_dim), str(l1reg), dataset, model_type]),
    )


def plot_loss_vs_regularization(df):
    g = sns.FacetGrid(data=df, col="sample", col_wrap=5)
    g.map_dataframe(
        sns.lineplot,
        x="alpha",
        y="rmse",
        hue="layers",
        palette="Blues",
        style="context",
    )
    [ax.set(yscale="log", xscale="log") for ax in g.axes]
    plt.tight_layout()
