import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import wandb
from common import EVALUATE_ALL_CSVS, fig_dir
from dmm.config_options import Conf
from dmm.wandb_init_log import init_wandb_ltt, log_performance
from evaluation_plotting import plot_parameter_heatmaps

conf = fire.Fire(Conf)

df = pd.read_csv(
    EVALUATE_ALL_CSVS.format(
        model=conf.model, data=conf.data, filename="by_cl_cond_obs"
    ),
    index_col=0,
)
df_rmse = pd.read_csv(
    EVALUATE_ALL_CSVS.format(
        model=conf.model, data=conf.data, filename="evaluate_all"
    ),
    index_col=0,
)
df_rmse = df_rmse[df_rmse.rmse.apply(np.isfinite)]
df_par_dev = pd.read_csv(
    EVALUATE_ALL_CSVS.format(
        model=conf.model, data=conf.data, filename="param_devs"
    ),
    index_col=0,
)

rmse_cols = [
    c
    for c in df_rmse.columns
    if c not in ["dataset", "samples", "job", "rmse", "context"]
    and df_rmse[c].nunique() > 1
]

df_rmse["method"] = df_rmse.apply(
    lambda r: "_".join(
        str(r[x])
        for x in rmse_cols
        if isinstance(r[x], str) or not np.isnan(r[x])
    ),
    axis=1,
)

# order by average performance on val
x_order = (
    df_rmse.loc[df_rmse.dataset == "val", ["ref", "rmse"]]
    .groupby("ref")
    .agg("mean")
    .sort_values("rmse")
    .index.tolist()
)

g = sns.FacetGrid(df_rmse, row="dataset", col="samples", margin_titles=True)
g.map_dataframe(
    sns.boxplot,
    data=df_rmse,
    y="method",
    x="rmse",
    order=x_order,
    hue="context",
)
outdir = fig_dir / conf.model / conf.data
outdir.mkdir(parents=True, exist_ok=True)
plt.savefig(outdir / "performance.pdf")

# average over jobs+samples
gb = [
    k
    for k in conf.__dict__.keys()
    if (k not in ["job", "samples", "sample"]) and (k in df.columns)
]

for group, df_run in df.groupby(["ref"] + gb, dropna=False):
    type = group[0]
    conf_run = Conf(
        **{"model": conf.model, "data": conf.data, **dict(zip(gb, group[1:]))}
    )

    if type == "DMM":
        gb_rmse = [
            g for g in gb if g not in ["sample", "observable", "condition"]
        ]
    elif type in ["sample", "avg_model"]:
        gb_rmse = ["ref", "context"]
    else:
        # regressors
        gb_rmse = ["ref", "context", "features"]
    rmses = df_rmse[(df_rmse[gb_rmse] == df_run[gb_rmse].iloc[0]).all(axis=1)]

    init_wandb_ltt(conf_run, type)
    for dataset in ["train", "val"]:
        rmse = rmses[rmses.dataset == dataset].rmse.mean()
        log_performance(df_run[df_run.dataset == dataset], dataset, rmse)

    if type == "DMM":
        par_dev = df_par_dev[
            (df_par_dev[gb_rmse] == df_run[gb_rmse].iloc[0]).all(axis=1)
        ]

        param_cols = [
            c
            for c in par_dev.columns
            if c
            not in ["cell_line", "samples", "dataset", "job", "median_init"]
            + gb_rmse
        ]
        gb_par_dev = ["cell_line", "dataset", "samples"]
        par_dev = (
            par_dev[param_cols + gb_par_dev].groupby(gb_par_dev).agg("mean")
        )

        plot_parameter_heatmaps(
            par_dev,
            param_cols,
            fig_dir
            / conf.model
            / conf.data
            / conf_run.context
            / conf_run.features
            / f"par_dev__{conf_run}",
        )

    wandb.finish()
