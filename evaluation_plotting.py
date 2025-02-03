import seaborn as sns
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import EVALUATE_ALL, CONTEXT_SET, hardest_cell_lines, subtypes_tognetti
from pathlib import Path
from typing import List, Union


subtypes_pam50 = {cl: subtypes_tognetti[cl]["PAM50"] for cl in subtypes_tognetti.keys()}
subtypes_lb = {cl: subtypes_tognetti[cl]["Luminal/Basal"] for cl in subtypes_tognetti.keys()}


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
        plt.close()  # ensure figure is closed
        efile = rfile.replace(".pdf", ".csv")
        dataframe.to_csv(efile)


def performance_barplot(
        dataframe: pd.DataFrame,
        conf,
        group_name: str,
):
    # PERFORMANCE BARPLOT
    _ = plt.figure()
    g = sns.FacetGrid(
        data=dataframe,
        row="dataset",  # top: train, bottom: test
        col="context",  # columns: cytof_init, proteomics, transcriptomics
        row_order=("train", "test"),
        col_order=sorted(CONTEXT_SET),
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
    rfile = EVALUATE_ALL.format(**conf.__dict__, group=group_name)
    plt.savefig(rfile)
    plt.savefig(rfile.replace('pdf', 'svg'))
    # plt.show(
    plt.close()  # ensure figure is closed


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


def plot_latent_embeddings(
        le_df: pd.DataFrame,
        df_label: str,
        reg_param: str,
        save_path: str,
        which_cells: str
):
    plot_df = le_df.copy()

    plot_df["PAM50"] = plot_df.cell_line.map(subtypes_pam50)
    plot_df["LB"] = plot_df.cell_line.map(subtypes_lb)

    for context in plot_df.context.unique():
        sub_df = plot_df[plot_df.context == context]
        if which_cells == "val_only":
            sub_df = sub_df[sub_df.cell_line.isin(hardest_cell_lines)]
        g = sns.FacetGrid(
            sub_df,
            row="samples",
            row_order=sorted(sub_df.samples.unique()),
            col=reg_param,
            col_order=sorted(sub_df[reg_param].unique()),
        )
        g.map_dataframe(sns.scatterplot, x="L1", y="L2", hue="cell_line")
        g.add_legend()
        plt.savefig(save_path.format(context=context, df_label=df_label, which_cells=which_cells, plot_by="samples"))
        plt.close()

        for subtype_scheme in ["PAM50", "LB"]:
            g = sns.FacetGrid(
                sub_df,
                row="samples",
                row_order=sorted(sub_df.samples.unique()),
                col=reg_param,
                col_order=sorted(sub_df[reg_param].unique()),
            )
            g.map_dataframe(sns.scatterplot, x="L1", y="L2", hue=subtype_scheme)
            g.add_legend()
            plt.savefig(
                save_path.format(context=context, df_label=df_label, which_cells=which_cells, plot_by=subtype_scheme))
            plt.close()

        if which_cells == "val_only":
            g = sns.FacetGrid(
                sub_df,
                row="cell_line",
                col=reg_param,
                col_order=sorted(sub_df[reg_param].unique()),
            )
            g.map_dataframe(sns.scatterplot, x="L1", y="L2", hue="samples")
            g.add_legend()
            plt.savefig(save_path.format(context=context, df_label=df_label, which_cells=which_cells, plot_by="cell_line"))
            plt.close()

def plot_val_param_dev_spread(
        param_dev_df: pd.DataFrame,
        param_cols: List,
        top_reg_param: str,
        reg_params: List,
        figure_filepath: Union[str, Path],
):
    param_val_df = param_dev_df[param_dev_df.dataset == "test"]
    # Melt the dataframe to get a boxplot
    param_val_df = pd.melt(
        param_val_df
        .drop(columns=[col for col in param_val_df.columns if
                       col not in param_cols + reg_params + ["cell_line"]]),
        id_vars=["cell_line"] + reg_params,
        var_name='Parameter',
        value_name='value'
    )
    # Create FacetGrid with one column per parameter, one CV split per row
    g = sns.FacetGrid(param_val_df, row="cell_line", col=top_reg_param, aspect=4, sharey=True)
    g.map_dataframe(sns.boxplot, x="Parameter", y="value")
    # g.map_dataframe(sns.stripplot, x="Parameter", y="value")
    plt.ylim([-4, 4])
    for ax in g.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(90)
    plt.tight_layout()
    plt.savefig(figure_filepath)
    plt.close()


def plot_parameter_heatmaps(
        param_df: pd.DataFrame,
        param_cols: List,
        group_cols: List,
        top_reg_param: str,
        plot_label: str,
        figure_filepath: Union[str, Path],
        val_only: bool,
        add_avg_to_val: bool = False,
):
    # Adds an extra row in each heatmap corresponding to average cell-line
    if val_only and add_avg_to_val:
        # Create "average" rows for each unique group
        avg_df = (
            param_df.groupby(
                [col for col in group_cols if col not in ["cell_line", "dataset"]],
                dropna=False
            )[param_cols]
            .mean()
            .reset_index()
        ).assign(cell_line="average")  # Add label for average cell line
        param_df = pd.concat([param_df, avg_df], ignore_index=True)

    g = sns.FacetGrid(
        param_df,
        row="samples", row_order=sorted(param_df.samples.unique()),
        col=top_reg_param, col_order=sorted(param_df[top_reg_param].unique()),
        margin_titles=True, height=5, aspect=1.5
    )

    # Set colorbar range
    if plot_label == "param_dev":
        vmin, vmax = -1.5, 1.5
    elif plot_label == "param":
        vmin, vmax = -10, 10
    else:
        raise ValueError(f"Invalid plot_label: {plot_label}")

    def plot_heatmap_with_highlight(data, samples, **kwargs):
        ax = plt.gca()  # Get the current axis
        sns.heatmap(
            data.set_index("cell_line")[param_cols],
            vmin=vmin, vmax=vmax, cmap="vlag",
            xticklabels=True, yticklabels=True, ax=ax, **kwargs
        )
        # Highlight the validation cell line
        if val_only:
            # Get the correct validation cell line for the current split
            current_sample = data.samples.unique()[0]  # Extract the sample for this facet
            split_index = int(current_sample.split("of")[0])  # Parse the split index (e.g., 0of5 -> 0)
            val_cell_line = hardest_cell_lines[split_index]  # Adjust for the current split

            if val_cell_line in data.cell_line.values:
                idx = data.cell_line.tolist().index(val_cell_line)
                rect = patches.Rectangle(
                    (0, idx),  # Bottom-left corner of the rectangle
                    len(param_cols),  # Width of the rectangle (number of columns)
                    1,  # Height of the rectangle (1 row)
                    linewidth=2, edgecolor="gold", facecolor="none"
                )
                ax.add_patch(rect)

    g.map_dataframe(plot_heatmap_with_highlight, samples="samples")

    g.fig.tight_layout()
    plt.savefig(figure_filepath)
    plt.close()