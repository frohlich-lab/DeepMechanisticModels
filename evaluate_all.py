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
    EVALUATION_REFERENCE_REG,
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

JOBS = tuple([i for i in range(10)]) #need to change this - NO HARDCODING
dfs = []
for samples in SPLITS:
    for dataset in [
        "train",
        "test"
    ]:
        # training
        training = pd.concat(
            pd.read_csv(efile, index_col=0)
            for orth_reg_strategy, alpha, beta, gamma, delta, ldim, job, pretrain, (
                ctxt,
                features,
            ) in itt.product(
                ORTH_REG_STRATEGIES,
                ALPHAS,
                BETAS,
                GAMMAS,
                DELTAS,
                LATENT_DIMS,
                JOBS,
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
                            job=job,
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

        # linear regressor baseline
        linreg_baseline = pd.concat(
            pd.read_csv(
                EVALUATION_REFERENCE_REG.format(
                    **{
                        **conf.__dict__,
                        **dict(
                            samples=samples,
                            dataset=dataset,
                            context=ctxt,
                        ),
                    },
                    mode="linreg",
                ),
                index_col=0,
            )
            for ctxt, features in CONTEXTS_FEATURES
        )
        linreg_baseline["ref"] = "linreg"

        # Lasso regressor baseline
        lasso_baseline = pd.concat(
            pd.read_csv(
                EVALUATION_REFERENCE_REG.format(
                    **{
                        **conf.__dict__,
                        **dict(
                            samples=samples,
                            dataset=dataset,
                            context=ctxt,
                        ),
                    },
                    mode="lasso",
                ),
                index_col=0,
            )
            for ctxt, features in CONTEXTS_FEATURES
        )
        lasso_baseline["ref"] = "lasso"

        # ElasticNet regressor baseline
        elasticnet_baseline = pd.concat(
            pd.read_csv(
                EVALUATION_REFERENCE_REG.format(
                    **{
                        **conf.__dict__,
                        **dict(
                            samples=samples,
                            dataset=dataset,
                            context=ctxt,
                        ),
                    },
                    mode="elasticnet",
                ),
                index_col=0,
            )
            for ctxt, features in CONTEXTS_FEATURES
        )
        elasticnet_baseline["ref"] = "elasticnet"


        avg_ps_dfs = []
        # copy average/per sample
        for (
            orth_reg_strategy,
            alpha,
            beta,
            gamma,
            delta,
            ldim,
            job,
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
            JOBS,
            METHODS,
            PRETRAIN,
            CONTEXTS_FEATURES,
        ):
            for rdf in [ # lack context
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
                avg_ps_df["job"] = job
                avg_ps_dfs.append(avg_ps_df)

            # regression baselines already have context
            for rdf in [
                linreg_baseline,
                lasso_baseline,
                elasticnet_baseline,
            ]:
                avg_ps_df = rdf.copy()
                avg_ps_df["orth_reg_strategy"] = orth_reg_strategy
                avg_ps_df["l1reg_inflate"] = alpha
                avg_ps_df["oreg_inflate"] = beta
                avg_ps_df["l1reg_encode"] = gamma
                avg_ps_df["oreg_encode"] = delta
                avg_ps_df["layers"] = ldim
                avg_ps_df["type"] = method
                avg_ps_df["pretrain"] = pretrain
                avg_ps_df["features"] = features
                avg_ps_df["job"] = job
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


def lineplot_ref_linreg(data, *args, **kwargs):
    sns.lineplot(data[data["ref"] == "linreg"], *args, **kwargs)

def lineplot_ref_lasso(data, *args, **kwargs):
    sns.lineplot(data[data["ref"] == "lasso"], *args, **kwargs)

def lineplot_ref_elasticnet(data, *args, **kwargs):
    sns.lineplot(data[data["ref"] == "elasticnet"], *args, **kwargs)


gbs = [
    "ref",
    "dataset",
    "context",
    "latent dim",
    "job",
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
            rmse=np.sqrt(np.square(group_df["res"]).mean()), #this will produce the mean RMSE across all jobs (not the best result then)
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
    "job",
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
        palette="mako",
        markers=True,
    )

    g.map_dataframe(
        lineplot_ref_average,
        x=group,
        y="rmse",
        color="blue",
        linestyle="--",
        errorbar=None,
        markers=True,
    )

    g.map_dataframe(
        lineplot_ref_sample,
        x=group,
        y="rmse",
        color="brown",
        linestyle=":",
        errorbar=None,
        markers=True,
    )

    g.map_dataframe(
        lineplot_ref_linreg,
        x=group,
        y="rmse",
        color="red",
        linestyle="-.",
        errorbar=None,
        markers=True,
    )

    g.map_dataframe(
        lineplot_ref_lasso,
        x=group,
        y="rmse",
        color="green",
        linestyle="-.",
        errorbar=None,
        markers=True,
    )

    g.map_dataframe(
        lineplot_ref_elasticnet,
        x=group,
        y="rmse",
        color="orange",
        linestyle="-.",
        errorbar=None,
        markers=True,
    )
    if (group == "job") or (group == "orth_reg_strategy"):
        g.set(ylim=(0.1, 0.7)) # symlog for unregularised settings;
    else:
        g.set(xscale="symlog", xlim=(0, 1e10), ylim=(0.1, 0.7)) # symlog to include unregularised hyperparam combos
    g.add_legend()
    plt.tight_layout()
    rfile = EVALUATE_ALL.format(**conf.__dict__, group=group)
    plt.savefig(rfile)
    efile = rfile.replace(".pdf", ".csv")
    data.to_csv(efile)

## PERFORMANCE BARPLOT
# avg_model, regression baselines, method (DMM, average with min-max range
# across SPLITS and multistarts), per_sample
data2 = data.groupby(by = ['ref', 'dataset', 'context',
                           'latent dim', 'orth_reg_strategy',
                           'l1reg_inflate', 'oreg_inflate',
                           'l1reg_encode', 'oreg_encode',
                           'features'], as_index=False)['rmse'].mean()

fig = plt.figure()
g2 = sns.FacetGrid(
    data=data2,
    row="dataset", # top: train, bottom: test
    col="context", # columns: cytof_init, proteomics, transcriptomics
    row_order=("train", "test"),
)

g2.map_dataframe(
    sns.barplot,
    x='ref', #various regressors on x_axis
    y="rmse", #rmse on y axis
    hue="ref", #color by method/reference/baseline
    errorbar= lambda x: (x.min(), x.max()), #display performance range between various jobs using
    palette='tab10',
)

g2.set(ylim=(0.1, 0.6))
# rotate xlabels
g2.tick_params(axis='x', rotation=90)
g2.add_legend()
plt.tight_layout()
rfile = EVALUATE_ALL.format(**conf.__dict__, group="baseline_barplot")
plt.savefig(rfile)

# Log via W&B
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
