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
    evaluations_dir
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
from stat_test import statistical_significance_test


conf = fire.Fire(Conf)

outdir = fig_dir / conf.model / conf.data

# METHODS = ("pca embedding", "end-to-end")

JOBS = tuple([i for i in range(10)])  # need to change this - NO HARDCODING
dfs = []
for samples in SPLITS:
    for dataset in [
        # "train",  # TODO @GiacomoFabrini: re-enable once hyperparam grid is narrower
        "test"
    ]:
        print(f'Starting to concatenate training evaluations for {samples}, {dataset}')
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
        print(f'Finished concatenating training evaluations for {samples}, {dataset}')

        # Loss vs regularization plot
        print(f'Starting to plot loss_vs_regularization for {samples}, {dataset}')
        plot_loss_vs_regularization(training)
        plt.savefig(outdir / f"{samples}_evaluate_training_{dataset}.pdf")
        print(f'Saved loss_vs_regularization plot for {samples}, {dataset}')

        # Add necessary attributes to training DataFrame
        training["ref"] = "DMM"  # previously "meth"
        training["dataset"] = dataset
        training["samples"] = samples

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
        print(f'Processing avg_model for {samples}, {dataset}')
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
        print(f'Finished processing avg_model for {samples}, {dataset}')

        # per sample
        print(f'Processing per_sample model for {samples}, {dataset}')
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
        print(f'Finished processing per_sample model for {samples}, {dataset}')

        # Process regressors - linreg, lasso, elasticnet
        print(f'Processing regressors model for {samples}, {dataset}')
        regressor_dfs = {
            mode: pd.concat(
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
                        mode=mode,
                    ),
                    index_col=0,
                )
                for ctxt, features in CONTEXTS_FEATURES
            ).assign(ref=mode)
            for mode in ["linreg", "lasso", "elasticnet"]
        }
        print(f'Finished processing regressors for {samples}, {dataset}')

        print(f'Starting to build hyperparam/job combination copies for references models - {samples}, {dataset}')
        avg_ps_dfs = []
        for context, _ in CONTEXTS_FEATURES:
            # need to replicate info across contexts for "avg_model" and "sample"
            for rdf in [  # lack context
                # avg,
                avg_model,
                ps,
            ]:
                avg_ps_df = rdf.copy()
                # they have no hyperparams -- None
                avg_ps_df["orth_reg_strategy"] = None
                avg_ps_df["l1reg_inflate"] = None
                avg_ps_df["oreg_inflate"] = None
                avg_ps_df["l1reg_encode"] = None
                avg_ps_df["oreg_encode"] = None
                avg_ps_df["layers"] = None
                avg_ps_df["context"] = context
                # avg_ps_df["type"] = method
                # avg_ps_df["pretrain"] = pretrain
                avg_ps_df["features"] = None
                avg_ps_df["job"] = None
                avg_ps_dfs.append(avg_ps_df)
                # Once appended, this can be deleted
                del avg_ps_df

        # regression baselines already have context
        # but no hyperparameters
        for _, rdf in regressor_dfs.items():
            avg_ps_df = rdf.copy()
            avg_ps_df["orth_reg_strategy"] = None
            avg_ps_df["l1reg_inflate"] = None
            avg_ps_df["oreg_inflate"] = None
            avg_ps_df["l1reg_encode"] = None
            avg_ps_df["oreg_encode"] = None
            avg_ps_df["layers"] = None
            # avg_ps_df["type"] = method
            # avg_ps_df["pretrain"] = pretrain
            avg_ps_df["features"] = None
            avg_ps_df["job"] = None
            avg_ps_dfs.append(avg_ps_df)
            # Once appended, this can be deleted
            del avg_ps_df
        print(f"Finished processing reference models for {samples}, {dataset}")

        # dfd = pd.concat([training, pretraining])
        dfd = pd.concat([training, *avg_ps_dfs])
        # Deleting DataFrames once concatenated into dfd
        del training, avg_ps_dfs, regressor_dfs
        dfd["dataset"] = dataset
        dfd["samples"] = samples
        dfs.append(dfd)
        # Deleting dfd once appended to dfs
        del dfd
        print(f"Finished concatenating training and reference models for {samples}, {dataset}")

df = pd.concat(dfs).reset_index()
# Now that dfs have been concatenated into df, delete them
del dfs
df.rename(
    columns={
        "layers": "latent dim",
        # "type": "method", #not used at the moment?!
    },
    inplace=True,
)


gbs = [
    "dataset",
    "context",
    "features",
    "samples",
    "ref",
    "orth_reg_strategy",
    "latent dim",
    "l1reg_inflate",
    "oreg_inflate",
    "l1reg_encode",
    "oreg_encode",
    "job",
]

data_dmm = pd.DataFrame(
    [
        dict(
            zip(gbs, group),
            rmse=np.sqrt(np.square(group_df["res"]).mean()),  # mean RMSE across all jobs (not best result)
        )
        for group, group_df in df.groupby(gbs)
    ]
)

gbs_refs = [
    "dataset",
    "context",
    "samples",
    "ref",
]

df_refs = df[~df.ref.isin(["DMM"])]
data_refs = pd.DataFrame(
    [
        dict(
            zip(gbs_refs, group_ref),
            rmse=np.sqrt(np.square(group_df_ref["res"]).mean()),  # mean RMSE = RMSE (single values)
        )
        for group_ref, group_df_ref in df_refs.groupby(gbs_refs)
    ]
)

data = pd.concat([data_dmm, data_refs]).sort_values(by="ref")
# cleanup
del df, df_refs, data_dmm, data_refs

print("Overall evaluation DataFrame is now ready.")

# Create pivot table for statistical testing
cols = ['dataset', 'context', 'features', 'ref',
        'latent dim', 'orth_reg_strategy',
        'l1reg_inflate', 'oreg_inflate',
        'l1reg_encode', 'oreg_encode']
# pivot table and create one column per cross-validation split and multistart/job
pivot_data = data.pivot_table(index=cols, columns=['samples', 'job'], values='rmse')
pivot_data = pivot_data.reset_index()
# Create list of the MultiIndex RMSE columns created above
multiindex_rmse_cols = [(sample, job) for sample in SPLITS for job in JOBS]
# Create a single column 'rmse_list' listing all values from each of the MultiIndex columns (same order for all rows)
pivot_data['rmse_list'] = pivot_data.apply(lambda row: np.array([row[col] for col in multiindex_rmse_cols]), axis=1)
# Add the newly created column to the list of columns to be kept (cols)
cols += ['rmse_list']
# Subset the pivot table and reduce MultiIndex back to single-level index
data_stat_tests = pivot_data[cols]
data_stat_tests.columns = data_stat_tests.columns.droplevel(level=1)
print("DataFrame for statistical testing is now ready.")

stat_test_res_df = statistical_significance_test(data_stat_tests)
stat_test_res_df.to_csv(
    evaluations_dir
    / f"{conf.model}"
    / f"{conf.data}"
    / f"{conf.model}.{conf.data}.stat_tests_all.csv"
)


# ################################################################# #
# ####################### Performance Plots ####################### #
# ################################################################# #

def lineplot_methods(data, *args, **kwargs):
    sns.lineplot(data[data["ref"] == "DMM"], *args, **kwargs)

# Currently unused -> plt.axhline instead
# def lineplot_refs(data, *args, **kwargs):
#     # Single plotting function for all references -- this can then be used together with
#     # hue = "ref" to produce a more useful legend
#     references = ['avg_model', 'sample', 'linreg', 'lasso', 'elasticnet']
#     sns.lineplot(
#         data[data["ref"].isin(references)], #subset
#         *args, **kwargs
#     )


# Calculate the mean 'rmse' for each 'ref' value
rmse_refs = data[data['ref'].isin(
    ['avg_model', 'linreg', 'lasso', 'elasticnet', 'sample']
)].groupby(
    ['dataset', 'ref', 'context']
)['rmse'].mean()

ref_cmap = sns.color_palette("tab10")
ref_palette_dict = {
    "avg_model": ref_cmap[0],
    "linreg": ref_cmap[1],
    "lasso": ref_cmap[2],
    "elasticnet": ref_cmap[4],
    "sample": ref_cmap[5],
    "DMM": ref_cmap[3],
}
ref_linestyle_dict = {
    "avg_model": 'dotted',
    "linreg": 'dashed',
    "lasso": 'dashed',
    "elasticnet": 'dashed',
    "sample": 'dotted',
    "DMM": 'solid',
}

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
        col_order=("cytof_init", "proteomics", "transcriptomics"),
        # sharex = True,
        sharey=True,
    )

    g.map_dataframe(
        lineplot_methods,
        x=group,
        y="rmse",
        hue="features",
        errorbar="se",
        style="latent dim",
        palette="rocket",
        markers=True,
    )

    # Apply plt.axhline to each subplot
    for (dataset, ref, context), rmse in rmse_refs.items():
        g.axes_dict[dataset, context].axhline(y=rmse,
                                              color=ref_palette_dict[ref],
                                              linestyle=ref_linestyle_dict[ref],
                                              label=ref)
    # Once done, add legend to last examined dataset and context
    g.axes_dict[dataset, context].legend(frameon=False, bbox_to_anchor=[1, 1])

    if (group == "job") or (group == "orth_reg_strategy"):
        g.set(ylim=(0, 1.1))  # symlog for unregularised settings;
    else:
        # g.set(xscale="symlog", xlim=(0, 1e10), ylim=(0.1, 0.7))  # symlog to include unregularised hyperparam combos
        g.set(xscale="symlog", xlim=(0, 1e10), ylim=(0, 1.1))   # symlog to include unregularised hyperparam combos
    # g.add_legend()
    plt.tight_layout()
    rfile = EVALUATE_ALL.format(**conf.__dict__, group=group)
    plt.savefig(rfile)
    plt.savefig(rfile.replace(".pdf", ".svg"))
    efile = rfile.replace(".pdf", ".csv")
    data.to_csv(efile)

# PERFORMANCE BARPLOT
# # avg_model, regression baselines, method (DMM, average with min-max range
# # across SPLITS, i.e. "samples", and multistarts, i.e. jobs), per_sample
#
# data2 = data.groupby(by = ['dataset',
#                            'context', 'features',
#                            'ref',
#                            #'pretrain',
#                            'orth_reg_strategy', 'latent dim',
#                            'l1reg_inflate', 'oreg_inflate',
#                            'l1reg_encode', 'oreg_encode'], as_index=False)['rmse'].mean()
# useless? It already seems to aggregate over all other features when producing the barplot

fig = plt.figure()
g2 = sns.FacetGrid(
    data=data,
    row="dataset",  # top: train, bottom: test
    col="context",  # columns: cytof_init, proteomics, transcriptomics
    row_order=("train", "test"),
    col_order=("cytof_init", "proteomics", "transcriptomics"),
)

g2.map_dataframe(
    sns.barplot,
    x='ref',  # various regressors on x_axis
    y="rmse",  # rmse on y axis
    hue="ref",  # color by method/reference/baseline
    hue_order=["avg_model",
               "linreg", "lasso", "elasticnet",
               "DMM", "sample"],
    errorbar=lambda x: (x.min(), x.max()),  # display performance range between various jobs using
    palette=ref_palette_dict,
)

g2.set(ylim=(0.1, 1.1))
# rotate xlabels
g2.tick_params(axis='x', rotation=90)
g2.add_legend()
plt.tight_layout()
rfile = EVALUATE_ALL.format(**conf.__dict__, group="baseline_barplot")
plt.savefig(rfile)
plt.savefig(rfile.replace('pdf', 'svg'))


# ################################################################ #
# #################### Statistical Test Plots #################### #
# ################################################################ #
# n_hidden pairwise comparisons
# subset to where n_hidden is null (n_hidden1 and n_hidden2 will be not null)
df_plot_n_hidden = stat_test_res_df[stat_test_res_df.n_hidden.isnull()]
num_contexts = len([context for context, _ in CONTEXTS_FEATURES])
plt.subplots(num_contexts, 2, figsize=(12, num_contexts*4))
plt.subplots_adjust(wspace=0.5, hspace=0.25)
index = 1
for context, _ in CONTEXTS_FEATURES:
    plt.subplot(num_contexts, 2, index)
    ax = sns.heatmap(
        df_plot_n_hidden[df_plot_n_hidden.context == context][
            ['n_hidden1', 'n_hidden2', 'Wilcoxon_statistic', 'adj_Wilcoxon_p-value']
        ].pivot(
            index='n_hidden1', columns='n_hidden2', values='adj_Wilcoxon_p-value'
        ),
        annot=True,
        square=True,
        vmin=0, vmax=1
    )
    ax.invert_yaxis()
    ax.set_yticks([0.5, 1.5, 2.5, 3.5], labels=[2, 4, 6, 8])
    ax.set_xticks([-0.5, 0.5, 1.5, 2.5], labels=[2, 4, 6, 8])
    # ax.set_xlim([-1.0, 4])
    plt.title(f"adjusted p-value | {context}")
    plt.subplot(num_contexts, 2, index+1)
    ax2 = sns.heatmap(
        df_plot_n_hidden[df_plot_n_hidden.context == context][
            ['n_hidden1', 'n_hidden2', 'Wilcoxon_statistic', 'adj_Wilcoxon_p-value']
        ].pivot(
            index='n_hidden1', columns='n_hidden2', values='Wilcoxon_statistic'
        ),
        annot=True,
        square=True,
        vmin=1e5, vmax=1.2e7
    )
    ax2.invert_yaxis()
    ax2.set_yticks([0.5, 1.5, 2.5, 3.5], labels=[2, 4, 6, 8])
    ax2.set_xticks([-0.5, 0.5, 1.5, 2.5], labels=[2, 4, 6, 8])
    # ax2.set_xlim([-1.0, 4])
    plt.title(f"test statistic | {context}")
    index += 2  # increase subplot index
# Finally, save the whole figure combining all contexts
plt.tight_layout()
rfile = EVALUATE_ALL.format(**conf.__dict__, group="heatmaps_n_hidden_pairwise")
plt.savefig(rfile)
plt.savefig(rfile.replace('pdf', 'svg'))


# Volcano plots for significance of various hyperparameter values
def scatterplot_func(data, *args, **kwargs):
    sns.scatterplot(data, *args, **kwargs)


# subset to where n_hidden1 is null (for pairwise n_hidden comparisons above)
df_plot_hp = stat_test_res_df[stat_test_res_df.n_hidden1.isnull()]

fig = plt.figure(figsize=(30, 10))
g3 = sns.FacetGrid(
    data=df_plot_hp,
    row="context",
    col="hyperparameter",
    row_order=("cytof_init", "proteomics", "transcriptomics"),
    sharey=True,
)

g3.map_dataframe(
    scatterplot_func,
    x="log10_Wilcoxon_statistic",
    y="-log10_adj_Wilcoxon_p-value",
    hue="n_hidden",
    hue_order=[2, 4, 6, 8],
    size="log10hp_value",   # changed to log10 scale to distinguish 1e2 from 1e4 (identical in linear scale)
    palette="tab10",
    style="stat-significant",
    style_order=[True, False],
)

for (_, col) in g3.axes_dict.keys():
    # Only add legend to the first row, spread it across two columns
    g3.axes_dict['cytof_init', col].legend(frameon=False, ncols=2)

g3.set_titles("{row_name} | {col_name}")
# g3.add_legend()
g3.tick_params(direction='in', length=5)
# g3.set(ylim=(-5, 60))
# g.fig.subplots_adjust(hspace=0.1, wspace=5)
plt.tight_layout()
rfile = EVALUATE_ALL.format(**conf.__dict__, group="volcano_plot_stat_test")
plt.savefig(rfile)
plt.savefig(rfile.replace('pdf', 'svg'))


# Save dataframe to CSV
data.to_csv(
    evaluations_dir
    / f"{conf.model}"
    / f"{conf.data}"
    / f"{conf.model}.{conf.data}.evaluate_all.csv")

# Log via W&B
wandb.init(
    project=f"DeepMechanisticModels.{conf.data}.{conf.model}",
    config={
        **conf.__dict__,
    },
)
# Log "evaluate all" artifact
artifact_eval = wandb.Artifact(
    name=f"evaluate_all_{conf.model}_{conf.data}",
    description="evaluate all",
    type="evaluation",
)
artifact_eval.add(wandb.Table(dataframe=data), "evaluate_all.csv")
wandb.log_artifact(artifact_eval)

# Log "stat test all" artifact
artifact_stat = wandb.Artifact(
    name=f"stat_test_all_{conf.model}_{conf.data}",
    description="stat test all",
    type="evaluation",
)
artifact_stat.add(
    wandb.Table(dataframe=stat_test_res_df, allow_mixed_types=True),
    "stat_test_all.csv"
)
wandb.log_artifact(artifact_stat)
# Close W&B session
wandb.finish()
