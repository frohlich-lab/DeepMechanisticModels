import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from common import EVALUATE_ALL, CONTEXT_SET


# Base plotting functions for FacetGrid
def lineplot_methods(data, *args, **kwargs):
    sns.lineplot(data[data["ref"] == "DMM"], *args, **kwargs)


def scatterplot_func(data, *args, **kwargs):
    sns.scatterplot(data, *args, **kwargs)


# Set reference-specific colours and linestyle
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


def group_plots(
        dataframe: pd.DataFrame,
        conf,
):

    # Compute mean 'rmse' for each reference/baseline
    rmse_refs = dataframe[dataframe['ref'].isin(
        ['avg_model', 'linreg', 'lasso', 'elasticnet', 'sample']
    )].groupby(
        ['dataset', 'ref', 'context']
    )['rmse'].mean()

    for group in (
            "orth_reg_strategy",
            "l1reg_inflate",
            "oreg_inflate",
            "l1reg_encode",
            "oreg_encode",
            "job",
    ):
        _ = plt.figure()
        g = sns.FacetGrid(
            data=dataframe,
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
            style="n_hidden",
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
            # g.set(xscale="symlog", xlim=(0, 1e10), ylim=(0.1, 0.7))  # symlog to include unregularised settings
            g.set(xscale="symlog", xlim=(0, 1e10), ylim=(0, 1.1))  # symlog to include unregularised settings
        # g.add_legend()
        plt.tight_layout()
        rfile = EVALUATE_ALL.format(**conf.__dict__, group=group)
        plt.savefig(rfile)
        plt.savefig(rfile.replace(".pdf", ".svg"))
        efile = rfile.replace(".pdf", ".csv")
        dataframe.to_csv(efile)


def performance_barplot(
        dataframe: pd.DataFrame,
        conf
):
    # PERFORMANCE BARPLOT
    _ = plt.figure()
    g = sns.FacetGrid(
        data=dataframe,
        row="dataset",  # top: train, bottom: test
        col="context",  # columns: cytof_init, proteomics, transcriptomics
        row_order=("train", "test"),
        col_order=("cytof_init", "proteomics", "transcriptomics"),
    )

    g.map_dataframe(
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

    g.set(ylim=(0.1, 1.1))
    # rotate xlabels
    g.tick_params(axis='x', rotation=90)
    g.add_legend()
    plt.tight_layout()
    rfile = EVALUATE_ALL.format(**conf.__dict__, group="baseline_barplot")
    plt.savefig(rfile)
    plt.savefig(rfile.replace('pdf', 'svg'))
    # plt.show()


# Volcano plots for significance of various hyperparameter values
def volcano_hyperparameter_significance(
        dataframe: pd.DataFrame,
        conf,
):

    _ = plt.figure(figsize=(30, 10))
    g = sns.FacetGrid(
        data=dataframe,
        row="context",
        col="hyperparameter",
        row_order=(
            "cytof_init",
            "proteomics",
            "transcriptomics"
        ),
        sharey=True,
    )

    g.map_dataframe(
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

    for (_, col) in g.axes_dict.keys():
        # Only add legend to the first row, spread it across two columns
        g.axes_dict['cytof_init', col].legend(frameon=False, ncols=2)

    g.set_titles("{row_name} | {col_name}")
    g.tick_params(direction='in', length=5)
    plt.tight_layout()
    rfile = EVALUATE_ALL.format(**conf.__dict__, group="volcano_plot_stat_test")
    plt.savefig(rfile)
    plt.savefig(rfile.replace('pdf', 'svg'))


def n_hidden_pairwise_heatmap(
        dataframe: pd.DataFrame,
        conf
):

    num_contexts = len(CONTEXT_SET)
    plt.subplots(num_contexts, 2, figsize=(12, num_contexts * 4))
    plt.subplots_adjust(wspace=0.5, hspace=0.25)
    index = 1
    for context in CONTEXT_SET:
        plt.subplot(num_contexts, 2, index)
        ax = sns.heatmap(
            dataframe[dataframe.context == context][
                ['n_hidden1', 'n_hidden2', 'Wilcoxon_statistic', 'adj_Wilcoxon_p-value']
            ].pivot(
                index='n_hidden1', columns='n_hidden2', values='adj_Wilcoxon_p-value'
            ),
            annot=True,
            square=True,
            vmin=0, vmax=1
        )
        plt.title(f"adjusted p-value | {context}")
        plt.subplot(num_contexts, 2, index + 1)
        ax2 = sns.heatmap(
            dataframe[dataframe.context == context][
                ['n_hidden1', 'n_hidden2', 'Wilcoxon_statistic', 'adj_Wilcoxon_p-value']
            ].pivot(
                index='n_hidden1', columns='n_hidden2', values='Wilcoxon_statistic'
            ),
            annot=True,
            square=True,
            vmin=1e5, vmax=1.2e7
        )
        for axis in [ax, ax2]:
            axis.invert_yaxis()
            axis.set_yticks([0.5, 1.5, 2.5, 3.5], labels=[2, 4, 6, 8])
            axis.set_xticks([-0.5, 0.5, 1.5, 2.5], labels=[2, 4, 6, 8])
        plt.title(f"test statistic | {context}")
        index += 2  # increase subplot index
    # Finally, save the whole figure combining all contexts
    plt.tight_layout()
    rfile = EVALUATE_ALL.format(**conf.__dict__, group="heatmaps_n_hidden_pairwise")
    plt.savefig(rfile)
    plt.savefig(rfile.replace('pdf', 'svg'))
