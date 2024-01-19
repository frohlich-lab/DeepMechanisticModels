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
    EVALUATION_REFERENCE,
    EVALUATION_TRAINING,
    fig_dir,
)
from dmm.analysis import plot_loss_vs_regularization
from training_configuration import (
    ORTH_REG_STRATEGIES,
    ALPHAS,
    BETAS,
    CONTEXTS_FEATURES,
    DELTAS,
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
        # training
        training = pd.concat(
            pd.read_csv(efile, index_col=0)
            for orth_reg_strategy, alpha, beta, gamma, delta, ldim, pretrain, (
                ctxt,
                features,
            ) in itt.product(
                ORTH_REG_STRATEGIES,
                ALPHAS,
                BETAS,
                GAMMAS,
                DELTAS,
                LATENT_DIMS,
                PRETRAIN,
                CONTEXTS_FEATURES,
            )
            if os.path.exists(
                efile := EVALUATION_TRAINING.format(
                    **{
                        **conf.__dict__,
                        **dict(
                            orth_reg_strategy=orth_reg_strategy,
                            l1reg_inflate=alpha,
                            oreg_inflate=beta,
                            l1reg_encode=gamma,
                            oreg_encode=delta,
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
            orth_reg_strategy,
            alpha,
            beta,
            gamma,
            delta,
            ldim,
            method,
            pretrain,
            (ctxt, features),
        ) in itt.product(
            ORTH_REG_STRATEGIES,
            ALPHAS,
            BETAS,
            GAMMAS,
            DELTAS,
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
                avg_ps_df["orth_reg_strategy"] = orth_reg_strategy
                avg_ps_df["l1reg_inflate"] = alpha
                avg_ps_df["oreg_inflate"] = beta
                avg_ps_df["l1reg_encode"] = gamma
                avg_ps_df["oreg_encode"] = delta
                avg_ps_df["layers"] = ldim
                avg_ps_df["context"] = ctxt
                avg_ps_df["type"] = method
                avg_ps_df["pretrain"] = pretrain
                avg_ps_df["features"] = features
                avg_ps_dfs.append(avg_ps_df)

        # dfd = pd.concat([training, pretraining])
        dfd = pd.concat([training, *avg_ps_dfs])
        dfd["dataset"] = dataset
        dfd["samples"] = samples
        dfs.append(dfd)

df = pd.concat(dfs).reset_index()
df.rename(
    columns={
        "layers": "latent dim",
        "type": "method",
    },
    inplace=True,
)


def lineplot_ref_average(data, *args, **kwargs):
    sns.lineplot(data[data["ref"] == "avg_model"], *args, **kwargs)


def lineplot_methods(data, *args, **kwargs):
    sns.lineplot(data[data["ref"] == "meth"], *args, **kwargs)


def lineplot_ref_sample(data, *args, **kwargs):
    sns.lineplot(data[data["ref"] == "sample"], *args, **kwargs)


gbs = [
    "ref",
    "dataset",
    "context",
    "latent dim",
    "orth_reg_strategy",
    "l1reg_inflate",
    "oreg_inflate",
    "l1reg_encode",
    "oreg_encode",
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

for group in (
    "orth_reg_strategy",
    "l1reg_inflate",
    "oreg_inflate",
    "l1reg_encode",
    "oreg_encode",
):
    fig = plt.figure()
    g = sns.FacetGrid(
        data=data,
        row="dataset",
        col="context",
        row_order=("train", "test"),
    )

    g.map_dataframe(
        lineplot_methods,
        x=group,
        y="rmse",
        hue="features",
        errorbar="se",
        style="latent dim",
        palette="tab10",
    )
    g.map_dataframe(
        lineplot_ref_average,
        x=group,
        y="rmse",
        color="r",
        linestyle="--",
        errorbar=None,
    )
    g.map_dataframe(
        lineplot_ref_sample,
        x=group,
        y="rmse",
        color="b",
        linestyle=":",
        errorbar=None,
    )
    g.set(xscale="log", ylim=(0.1, 0.6))
    g.add_legend()
    plt.tight_layout()
    rfile = EVALUATE_ALL.format(**conf.__dict__, group=group)
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
