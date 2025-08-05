"""Per sample pretraining."""

from logging import ERROR
from pathlib import Path

import amici.logging
import amici.petab.parameter_mapping
import fides
import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypesto
import seaborn as sns
from amici.petab import rdatas_to_simulation_df
from pypesto.optimize import FidesOptimizer

from common import (
    PER_SAMPLE_OUTFILE_PARS,
    PER_SAMPLE_OUTFILE_RESULTS,
    fig_dir,
    pretrain_dir,
)
from cytof.problem import CytofProblem
from dmm.config_options import Conf
from dmm.petab_subproblem import load_petab
from dmm.plotting import plot_single_sample
from dmm.pretraining import (
    generate_per_sample_pretraining_problems,
    pretrain,
    store_and_plot_pretraining,
)
from dmm.training_helper_funcs import Chi2Objective
from util import load_petab_base_files


def plot_sample_rates(x_full, problem, importer, sample, dir):
    x = problem.get_reduced_vector(x_full, problem.x_free_indices)
    res = problem.objective.base_objective(x, return_dict=True)
    amici_model = problem.objective.base_objective.amici_model
    simulation_df = rdatas_to_simulation_df(
        res["rdatas"],
        model=amici_model,
        measurement_df=importer.petab_problem.measurement_df,
    )

    plot_single_sample(
        importer.petab_problem.measurement_df,
        simulation_df,
        dir,
        sample,
        (sample + "_simulation"),
    )
    df = pd.concat(
        [
            pd.DataFrame(data=r.w, columns=amici_model.getExpressionNames())
            .assign(time=r.t)
            .assign(condition_id=r.id.split("__")[1])
            for r in res["rdatas"]
        ],
        axis=0,
    )

    df_rates = pd.melt(
        df,
        value_vars=[c for c in df.columns if c.endswith("_kr")],
        id_vars=["condition_id", "time"],
    )

    sns.FacetGrid(
        df_rates,
        col="variable",
        row="condition_id",
        sharey="col",
        margin_titles=True,
    ).map_dataframe(sns.lineplot, x="time", y="value")
    plt.tight_layout()
    plt.savefig(dir / f"{sample}_rates.pdf")

    df_factors = pd.melt(
        df,
        value_vars=[c for c in df.columns if c.endswith("_factor")],
        id_vars=["condition_id", "time"],
    )
    df_factors["effect"] = df_factors["variable"].apply(
        lambda x: x.split("_")[-2]
    )
    df_factors["effector"] = df_factors["variable"].apply(
        lambda x: "_".join(x.split("_")[2:-2])
        if x.split("_")[1] == "endo"
        else "_".join(x.split("_")[3:-2])
    )
    df_factors["target"] = df_factors["variable"].apply(
        lambda x: "_".join(x.split("_")[:2])
        if x.split("_")[1] == "endo"
        else "_".join(x.split("_")[:3])
    )

    df_factors["value"] = df_factors.apply(
        lambda r: r.value if r.effect == "activating" else -r.value, axis=1
    ).values

    sns.catplot(
        df_factors,
        kind="bar",
        col="target",
        row="condition_id",
        x="time",
        y="value",
        hue="effector",
        palette="tab20",
        fill=False,
        dodge=False,
        sharey="col",
        margin_titles=True,
    )
    plt.tight_layout()
    plt.savefig(dir / f"{sample}_factors.pdf")


np.random.seed(0)

conf = fire.Fire(Conf)

problem = CytofProblem(conf.model)

petab_base_importer = load_petab(
    problem=problem, dataset=conf.data, **load_petab_base_files(conf)
)

importer = generate_per_sample_pretraining_problems(
    importer=petab_base_importer,
    problem=problem,
    dataset=conf.data,
    sample=conf.sample,
)

outdir = pretrain_dir / conf.model / conf.data
figdir = fig_dir / conf.model / conf.data / "pretraining_sample"

factory = importer.create_objective_creator()
objective = Chi2Objective(factory.create_objective())
problem.apply_objective_settings(objective, n_threads=conf.threads)

pypesto_problem = importer.create_problem(
    objective=objective,
)

optimizer = FidesOptimizer(
    options={
        fides.Options.FATOL: 1e-8,
        fides.Options.XTOL: 1e-8,
        fides.Options.MAXTIME: 7200,
        fides.Options.MAXITER: 200,
    },
    # hessian_update=fides.hessian_approximation.BFGS()
)
amici.logging.get_logger("amici.swig_wrappers").setLevel(ERROR)
result = pretrain(
    problem=pypesto_problem,
    startpoint_method=pypesto.startpoint.UniformStartpoints(
        check_fval=True, check_grad=True
    ),
    nstarts=10,  # multistarts for pretraining (hard-coded)
    optimizer=optimizer,
)
results_file = Path(PER_SAMPLE_OUTFILE_RESULTS.format(**conf.__dict__))
pars_file = Path(PER_SAMPLE_OUTFILE_PARS.format(**conf.__dict__))
store_and_plot_pretraining(result, pfile=pars_file, rfile=results_file)
x_full = result.optimize_result.list[0].x
plot_sample_rates(
    x_full, pypesto_problem, importer, conf.sample, pars_file.parent
)
