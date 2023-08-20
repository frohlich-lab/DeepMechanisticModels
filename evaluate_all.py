import itertools as itt
import os

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import wandb
from common import (
    EVALUATE_ALL,
    EVALUATION_PRETRAINING,
    EVALUATION_REFERENCE,
    EVALUATION_TRAINING,
    fig_dir,
)
from dmm.analysis import plot_loss_vs_regularization
from training_configuration import (
    ALPHAS,
    CONTEXTS_FEATURES,
    LATENT_DIMS,
    PRETRAIN,
    SPLITS,
)
from util import Conf

conf = fire.Fire(Conf)

outdir = fig_dir / conf.model / conf.data

METHODS = ("pca embedding", "end-to-end")

avgs = dict()
ps = dict()
dfs = []
for samples in SPLITS:
    for dataset in ["train", "test"]:
        # cross sample pretraining
        pretraining = pd.concat(
            (
                pd.read_csv(
                    EVALUATION_PRETRAINING.format(
                        **{
                            **conf.__dict__,
                            **dict(
                                alpha=alpha,
                                n_hidden=ldim,
                                context=ctxt,
                                samples=samples,
                                dataset=dataset,
                                pretrain=pretrain,
                                features=features,
                            ),
                        },
                        mode="cross_sample",
                    ),
                    index_col=0,
                )
            )
            for alpha, ldim, pretrain, (ctxt, features) in itt.product(
                ALPHAS, LATENT_DIMS, PRETRAIN, CONTEXTS_FEATURES
            )
        )
        plot_loss_vs_regularization(pretraining)
        plt.savefig(
            outdir / f"{samples}_evaluate_pretrain_cross_sample_{dataset}.pdf"
        )

        pretraining["ref"] = "meth"

        # training
        training = pd.concat(
            (
                pd.read_csv(
                    efile,
                    index_col=0,
                )
            )
            for alpha, ldim, pretrain, (ctxt, features) in itt.product(
                ALPHAS, LATENT_DIMS, PRETRAIN, CONTEXTS_FEATURES
            )
            if os.path.exists(
                efile := EVALUATION_TRAINING.format(
                    **{
                        **conf.__dict__,
                        **dict(
                            alpha=alpha,
                            n_hidden=ldim,
                            context=ctxt,
                            samples=samples,
                            dataset=dataset,
                            pretrain=pretrain,
                            features=features,
                        ),
                    },
                )
            )
        )
        plot_loss_vs_regularization(training)
        plt.savefig(outdir / f"{samples}_evaluate_training_{dataset}.pdf")
        training["ref"] = "meth"

        # # average
        # avg = pd.read_csv(
        #     EVALUATION_REFERENCE.format(
        #         **{
        #             **conf.__dict__,
        #             **dict(
        #                 samples=samples,
        #                 dataset=dataset,
        #             ),
        #         },
        #         mode="average",
        #     ),
        #     index_col=0,
        # )
        # avg["ref"] = "avg"

        # model average
        avg_model = pd.read_csv(
            EVALUATION_REFERENCE.format(
                **{
                    **conf.__dict__,
                    **dict(
                        samples=samples,
                        dataset=dataset,
                    ),
                },
                mode="avg_model",
            ),
            index_col=0,
        )
        avg_model["ref"] = "avg_model"

        # per sample
        ps = pd.read_csv(
            EVALUATION_REFERENCE.format(
                **{
                    **conf.__dict__,
                    **dict(
                        samples=samples,
                        dataset=dataset,
                    ),
                },
                mode="per_sample",
            ),
            index_col=0,
        )
        ps["ref"] = "sample"

        avg_ps_dfs = []
        # copy average/per sample
        for alpha, ldim, method, pretrain, (ctxt, features) in itt.product(
            ALPHAS, LATENT_DIMS, METHODS, PRETRAIN, CONTEXTS_FEATURES
        ):
            for rdf in [
                # avg,
                avg_model,
                ps,
            ]:
                avg_ps_df = rdf.copy()
                avg_ps_df["alpha"] = alpha
                avg_ps_df["layers"] = ldim
                avg_ps_df["context"] = ctxt
                avg_ps_df["type"] = method
                avg_ps_df["pretrain"] = pretrain
                avg_ps_df["features"] = features
                avg_ps_dfs.append(avg_ps_df)

        # dfd = pd.concat([training, pretraining])
        dfd = pd.concat([pretraining, training, *avg_ps_dfs])
        dfd["dataset"] = dataset
        dfd["samples"] = samples
        dfs.append(dfd)

df = pd.concat(dfs).reset_index()
df.rename(
    columns={
        "alpha": "l2 regularization",
        "layers": "latent dim",
        "type": "method",
    },
    inplace=True,
)
df.loc[df["method"] == "cross_sample", "method"] = "pca embedding"
df.loc[df["method"] == "full", "method"] = "end-to-end"

df["cf"] = df["context"] + "_" + df["features"]


def lineplot_ref_average(data, *args, **kwargs):
    sns.lineplot(data[data["ref"] == "avg_model"], *args, **kwargs)


def lineplot_methods(data, *args, **kwargs):
    sns.lineplot(data[data["ref"] == "meth"], *args, **kwargs)


def lineplot_ref_sample(data, *args, **kwargs):
    sns.lineplot(data[data["ref"] == "sample"], *args, **kwargs)


for gb in ("observable", "time", "condition", "sample", "all"):
    gbs = [
        "ref",
        "dataset",
        "method",
        "cf",
        "latent dim",
        "l2 regularization",
        "samples",
    ]
    if gb != "all":
        gbs = [gb, *gbs]
    df_gb = pd.DataFrame(
        [
            dict(
                zip(gbs, group),
                rmse=np.sqrt(np.square(group_df["res"]).mean()),
            )
            for group, group_df in df.groupby(gbs)
        ]
    )

    if gb == "time":
        # filter non-canonical timepoints (not enough datapoints)
        data = df_gb[
            np.logical_not(df_gb.time.isin([12, 14, 15, 16, 18, 25, 35]))
        ]
    else:
        data = df_gb

    kwargs_fg = dict()
    kwargs_lp = dict()

    if gb == "all":
        kwargs_fg["row_order"] = ("train", "test")
        if len(data["cf"].unique()) > 1:
            kwargs_lp["style"] = "latent dim"
    else:
        data = data[(data["cf"] == "cytof_init_all")]
        kwargs_lp["style"] = "dataset"
    kwargs_lp["palette"] = "tab10"

    fig = plt.figure()
    g = sns.FacetGrid(
        data=data,
        row=gb if gb != "all" else "dataset",
        col="method",
        **kwargs_fg,
    )

    g.map_dataframe(
        lineplot_methods,
        x="l1 regularization",
        y="rmse",
        hue="cf",
        errorbar="se",
        **kwargs_lp,
    )
    g.map_dataframe(
        lineplot_ref_average,
        x="l1 regularization",
        y="rmse",
        color="r",
        linestyle="--",
        errorbar=None,
    )
    g.map_dataframe(
        lineplot_ref_sample,
        x="l1 regularization",
        y="rmse",
        color="b",
        linestyle=":",
        errorbar=None,
    )
    g.set(xscale="log", ylim=(0, 1.5))
    g.add_legend()
    plt.tight_layout()
    rfile = EVALUATE_ALL.format(**conf.__dict__, group=gb)
    plt.savefig(rfile)
    efile = rfile.replace(".pdf", ".csv")
    data.to_csv(efile)

    if gb == "all":
        wandb.init(
            project=f"DeepMechanisticModels.{conf.data}.{conf.model}",
            config={
                **conf.__dict__,
            },
        )

        artifact = wandb.Artifact(
            name=f"evaluate_all_{conf.model}_{conf.data}",
            description="evaluate all",
            type="evaluation",
        )
        artifact.add(wandb.Table(dataframe=data), "evaluate_all.csv")
        wandb.log_artifact(artifact)
        wandb.finish()
