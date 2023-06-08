import itertools as itt

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common import (
    EVALUATE_ALL,
    EVALUATION_PRETRAINING,
    EVALUATION_REFERENCE,
    EVALUATION_TRAINING,
    fig_dir,
)
from dmm.analysis import plot_loss_vs_regularization
from training_configuration import ALPHAS, CONTEXTS, LATENT_DIMS, SPLITS
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
                            ),
                        },
                        mode="cross_sample",
                    ),
                    index_col=0,
                )
            )
            for alpha, ldim, ctxt in itt.product(ALPHAS, LATENT_DIMS, CONTEXTS)
        )
        plot_loss_vs_regularization(pretraining)
        plt.savefig(
            outdir
            / f"{conf.samples}_evaluate_pretrain_cross_sample_{dataset}.pdf"
        )

        pretraining["ref"] = "meth"

        # training
        training = pd.concat(
            (
                pd.read_csv(
                    EVALUATION_TRAINING.format(
                        **{
                            **conf.__dict__,
                            **dict(
                                alpha=alpha,
                                n_hidden=ldim,
                                context=ctxt,
                                samples=samples,
                                dataset=dataset,
                            ),
                        },
                    ),
                    index_col=0,
                )
            )
            for alpha, ldim, ctxt in itt.product(ALPHAS, LATENT_DIMS, CONTEXTS)
        )
        plot_loss_vs_regularization(training)
        plt.savefig(outdir / f"{conf.samples}_evaluate_training_{dataset}.pdf")
        training["ref"] = "meth"

        # average
        avg = pd.read_csv(
            EVALUATION_REFERENCE.format(
                **{
                    **conf.__dict__,
                    **dict(
                        samples=samples,
                        dataset=dataset,
                    ),
                },
                mode="average",
            ),
            index_col=0,
        )
        avg["ref"] = "avg"

        # model average
        # avg_model = pd.read_csv(
        #    EVALUATION_REFERENCE.format(samples=samples, model=MODEL, data=DATA, dataset=dataset, mode='avg_model'),
        #    index_col=0
        # )

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
        for alpha, ldim, ctxt, method in itt.product(
            ALPHAS, LATENT_DIMS, CONTEXTS, METHODS
        ):
            for rdf in [
                avg,
                # avg_model,
                ps,
            ]:
                avg_ps_df = rdf.copy()
                avg_ps_df["alpha"] = alpha
                avg_ps_df["layers"] = ldim
                avg_ps_df["context"] = ctxt
                avg_ps_df["type"] = method
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


def lineplot_ref_average(data, *args, **kwargs):
    sns.lineplot(data[data["ref"] == "avg"], *args, **kwargs)


def lineplot_methods(data, *args, **kwargs):
    sns.lineplot(data[data["ref"] == "meth"], *args, **kwargs)


def lineplot_ref_sample(data, *args, **kwargs):
    sns.lineplot(data[data["ref"] == "sample"], *args, **kwargs)


for gb in ("observable", "time", "condition", "sample", "all"):
    gbs = [
        "ref",
        "dataset",
        "method",
        "context",
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
                rmse=np.sqrt(
                    group_df["res"].apply(lambda x: np.power(x, 2)).mean()
                ),
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

    kwargs = dict()

    if gb == "all":
        kwargs["row_order"] = ("train", "test")
        if len(data.context.unique()) > 1:
            kwargs["style"] = "context"
    else:
        data = data[data["context"] == "baseline"]
        kwargs["style"] = "dataset"

    g = sns.FacetGrid(
        data=data, row=gb if gb != "all" else "dataset", col="method", **kwargs
    )

    g.map_dataframe(
        lineplot_methods,
        x="l2 regularization",
        y="rmse",
        hue="latent dim",
        errorbar="se",
    )
    g.map_dataframe(
        lineplot_ref_average,
        x="l2 regularization",
        y="rmse",
        color="k",
        linestyle="--",
        errorbar=None,
    )
    g.map_dataframe(
        lineplot_ref_sample,
        x="l2 regularization",
        y="rmse",
        color="k",
        linestyle=":",
        errorbar=None,
    )
    g.set(xscale="log", ylim=(0, 1.5))
    g.add_legend()
    plt.tight_layout()
    rfile = EVALUATE_ALL.format(**conf.__dict__, group=gb)
    plt.savefig(rfile)
    data.to_csv(rfile.replace(".pdf", ".csv"))
