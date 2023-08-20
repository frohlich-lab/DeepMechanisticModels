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
    BETAS,
    CONTEXTS_FEATURES,
    GAMMAS,
    LATENT_DIMS,
    PRETRAIN,
    SPLITS,
)
from util import Conf

conf = fire.Fire(Conf)

outdir = fig_dir / conf.model / conf.data

METHODS = ("pca embedding", "end-to-end")

dfs = []
for samples in SPLITS:
    for dataset in [
        # "train",
        "test"
    ]:
        # cross sample pretraining
        pretraining = pd.concat(
            pd.read_csv(efile, index_col=0)
            for alpha, beta, ldim, pretrain, (ctxt, features) in itt.product(
                ALPHAS, BETAS, LATENT_DIMS, PRETRAIN, CONTEXTS_FEATURES
            )
            if os.path.exists(
                efile := EVALUATION_PRETRAINING.format(
                    **{
                        **conf.__dict__,
                        **dict(
                            alpha=alpha,
                            beta=beta,
                            n_hidden=ldim,
                            context=ctxt,
                            samples=samples,
                            dataset=dataset,
                            pretrain=pretrain,
                            features=features,
                        ),
                    },
                    mode="cross_sample",
                )
            )
        )
        plot_loss_vs_regularization(pretraining)
        plt.savefig(
            outdir / f"{samples}_evaluate_pretrain_cross_sample_{dataset}.pdf"
        )
        pretraining["ref"] = "meth"

        # training
        training = pd.concat(
            pd.read_csv(efile, index_col=0)
            for alpha, beta, gamma, ldim, pretrain, (
                ctxt,
                features,
            ) in itt.product(
                ALPHAS, BETAS, GAMMAS, LATENT_DIMS, PRETRAIN, CONTEXTS_FEATURES
            )
            if os.path.exists(
                efile := EVALUATION_TRAINING.format(
                    **{
                        **conf.__dict__,
                        **dict(
                            alpha=alpha,
                            beta=beta,
                            gamma=gamma,
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
        for (
            alpha,
            beta,
            gamma,
            ldim,
            method,
            pretrain,
            (ctxt, features),
        ) in itt.product(
            ALPHAS,
            BETAS,
            GAMMAS,
            LATENT_DIMS,
            METHODS,
            PRETRAIN,
            CONTEXTS_FEATURES,
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
        "alpha": "l1 regularization",
        "layers": "latent dim",
        "type": "method",
    },
    inplace=True,
)
df.loc[df["method"] == "cross_sample", "method"] = "pca embedding"
df.loc[df["method"] == "full", "method"] = "end-to-end"


def lineplot_ref_average(data, *args, **kwargs):
    sns.lineplot(data[data["ref"] == "avg_model"], *args, **kwargs)


def lineplot_methods(data, *args, **kwargs):
    sns.lineplot(data[data["ref"] == "meth"], *args, **kwargs)


def lineplot_ref_sample(data, *args, **kwargs):
    sns.lineplot(data[data["ref"] == "sample"], *args, **kwargs)


gbs = [
    "ref",
    "dataset",
    "method",
    "context",
    "latent dim",
    "l1 regularization",
    "samples",
    "features",
]
data = pd.DataFrame(
    [
        dict(
            zip(gbs, group),
            rmse=np.sqrt(np.square(group_df["res"]).mean()),
        )
        for group, group_df in df.groupby(gbs)
    ]
)

fig = plt.figure()
g = sns.FacetGrid(
    data=data[data["dataset"] == "test"],
    row="method",
    col="context",
    row_order=("pca embedding", "end-to-end"),
)

g.map_dataframe(
    lineplot_methods,
    x="l1 regularization",
    y="rmse",
    hue="features",
    errorbar="se",
    style="latent dim",
    palette="tab10",
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
rfile = EVALUATE_ALL.format(**conf.__dict__, group="all")
plt.savefig(rfile)
efile = rfile.replace(".pdf", ".csv")
data.to_csv(efile)

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
