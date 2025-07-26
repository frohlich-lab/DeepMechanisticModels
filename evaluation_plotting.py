from pathlib import Path
from typing import List, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common import CONTEXT_SET, EVALUATE_ALL, fig_dir, hardest_cell_lines

# subtypes_pam50 = {cl: subtypes_tognetti[cl]["PAM50"] for cl in subtypes_tognetti.keys()}
# subtypes_lb = {cl: subtypes_tognetti[cl]["Luminal/Basal"] for cl in subtypes_tognetti.keys()}
#
# colours_lb = {"Normal": "green", "Luminal": "blue", "Basal": "red", "Other": "black"}
# colours_pam50 = {"Normal": "green", "LA": "cyan", "LB": "navy", "HER2": "gold", "Basal": "red", "Other": "black"}
#
# colours_ms = {"Stable (MSS)": "green", "Instable (MSI-low)": "orange", "Instable (MSI-high)": "red", "Unknown": "gray"}
# colours_site = {
#     "In situ; Breast, epithelium": "cornflowerblue",
#     "In situ; Breast": "cyan",
#     "Metastatic; Pleural effusion": "orange",
#     "Metastatic; Pericardial effusion": "red",
#     "Metastatic; Skin": "purple",
#     "Metastatic; Ascites": "orange",
#     "Metastatic; Brain": "green",
#     "Unknown": "black"
# }


cv_samples_mapping = {f"{i}of5": hardest_cell_lines[i] for i in range(5)}
shared_category_colors = {
    "LA": "gold",
    "LB": "darkorange",
    "Basal": "cornflowerblue",
    "HER2": "firebrick",
    "Normal": "purple",
    "Other": "slategray",
    1: "navy",
    2: "darkorange",
    3: "cornflowerblue",
    4: "lightgray",
    5: "purple",
}

pam50_labels = ["LA", "LB", "HER2", "Basal", "Normal", "Other"]
lb_labels = ["Luminal", "Basal", "Normal", "Other"]
lb_to_shared = {
    "Luminal": "LA",
    "Basal": "Basal",
    "Normal": "Normal",
    "Other": "Other",
}


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
    "avg_model": "dotted",
    "linreg": "dashed",
    "lasso": "dashed",
    "elasticnet": "dashed",
    "sample": "dotted",
    "DMM": "solid",
}


def group_plots(
    dataframe: pd.DataFrame,
    conf,
):
    # Compute mean 'rmse' for each reference/baseline
    rmse_refs = (
        dataframe[
            dataframe["ref"].isin(
                ["avg_model", "linreg", "lasso", "elasticnet", "sample"]
            )
        ]
        .groupby(["dataset", "ref", "context"])["rmse"]
        .mean()
    )

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
            g.axes_dict[dataset, context].axhline(
                y=rmse,
                color=ref_palette_dict[ref],
                linestyle=ref_linestyle_dict[ref],
                label=ref,
            )
        # Once done, add legend to last examined dataset and context
        g.axes_dict[dataset, context].legend(
            frameon=False, bbox_to_anchor=[1, 1]
        )

        if (group == "job") or (group == "orth_reg_strategy"):
            g.set(ylim=(0, 1.1))  # symlog for unregularised settings;
        else:
            # g.set(xscale="symlog", xlim=(0, 1e10), ylim=(0.1, 0.7))  # symlog to include unregularised settings
            g.set(
                xscale="symlog", xlim=(0, 1e10), ylim=(0, 1.1)
            )  # symlog to include unregularised settings
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
        x="ref",  # various regressors on x_axis
        y="rmse",  # rmse on y axis
        hue="ref",  # color by method/reference/baseline
        hue_order=[
            "avg_model",
            "linreg",
            "lasso",
            "elasticnet",
            "DMM",
            "sample",
        ],
        errorbar=lambda x: (
            x.min(),
            x.max(),
        ),  # display performance range between various jobs using
        palette=ref_palette_dict,
    )

    g.set(ylim=(0.1, 1.1))
    # rotate xlabels
    g.tick_params(axis="x", rotation=90)
    g.add_legend()
    plt.tight_layout()
    rfile = EVALUATE_ALL.format(**conf.__dict__, group=group_name)
    plt.savefig(rfile)
    plt.savefig(rfile.replace("pdf", "svg"))
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
        row_order=("cytof_init", "proteomics", "transcriptomics"),
        sharey=True,
    )

    g.map_dataframe(
        scatterplot_func,
        x="log10_Wilcoxon_statistic",
        y="-log10_adj_Wilcoxon_p-value",
        hue="n_hidden",
        hue_order=[2, 4, 6, 8],
        size="log10hp_value",  # changed to log10 scale to distinguish 1e2 from 1e4 (identical in linear scale)
        palette="tab10",
        style="stat-significant",
        style_order=[True, False],
    )

    for _, col in g.axes_dict.keys():
        # Only add legend to the first row, spread it across two columns
        g.axes_dict["cytof_init", col].legend(frameon=False, ncols=2)

    g.set_titles("{row_name} | {col_name}")
    g.tick_params(direction="in", length=5)
    plt.tight_layout()
    rfile = EVALUATE_ALL.format(
        **conf.__dict__, group="volcano_plot_stat_test"
    )
    plt.savefig(rfile)
    plt.savefig(rfile.replace("pdf", "svg"))


def n_hidden_pairwise_heatmap(dataframe: pd.DataFrame, conf):
    num_contexts = len(CONTEXT_SET)
    plt.subplots(num_contexts, 2, figsize=(12, num_contexts * 4))
    plt.subplots_adjust(wspace=0.5, hspace=0.25)
    index = 1
    for context in CONTEXT_SET:
        plt.subplot(num_contexts, 2, index)
        ax = sns.heatmap(
            dataframe[dataframe.context == context][
                [
                    "n_hidden1",
                    "n_hidden2",
                    "Wilcoxon_statistic",
                    "adj_Wilcoxon_p-value",
                ]
            ].pivot(
                index="n_hidden1",
                columns="n_hidden2",
                values="adj_Wilcoxon_p-value",
            ),
            annot=True,
            square=True,
            vmin=0,
            vmax=1,
        )
        plt.title(f"adjusted p-value | {context}")
        plt.subplot(num_contexts, 2, index + 1)
        ax2 = sns.heatmap(
            dataframe[dataframe.context == context][
                [
                    "n_hidden1",
                    "n_hidden2",
                    "Wilcoxon_statistic",
                    "adj_Wilcoxon_p-value",
                ]
            ].pivot(
                index="n_hidden1",
                columns="n_hidden2",
                values="Wilcoxon_statistic",
            ),
            annot=True,
            square=True,
            vmin=1e5,
            vmax=1.2e7,
        )
        for axis in [ax, ax2]:
            axis.invert_yaxis()
            axis.set_yticks([0.5, 1.5, 2.5, 3.5], labels=[2, 4, 6, 8])
            axis.set_xticks([-0.5, 0.5, 1.5, 2.5], labels=[2, 4, 6, 8])
        plt.title(f"test statistic | {context}")
        index += 2  # increase subplot index
    # Finally, save the whole figure combining all contexts
    plt.tight_layout()
    rfile = EVALUATE_ALL.format(
        **conf.__dict__, group="heatmaps_n_hidden_pairwise"
    )
    plt.savefig(rfile)
    plt.savefig(rfile.replace("pdf", "svg"))


def plot_latent_embeddings(
    le_df: pd.DataFrame,
    df_label: str,
    reg_param: str,
    save_path: str,
    which_cells: str,
):
    plot_df = le_df.copy()

    # plot_df["PAM50"] = plot_df.cell_line.map(subtypes_pam50)
    # plot_df["LB"] = plot_df.cell_line.map(subtypes_lb)

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
        plt.savefig(
            save_path.format(
                context=context,
                df_label=df_label,
                which_cells=which_cells,
                plot_by="samples",
            )
        )
        plt.close()

        for subtype_scheme in [
            col
            for col in [
                "PAM50",
                "LB",
                "HR_Status",
                "HER2_Status",
                "Site",
                "MS_Status",
                "Disease",
            ]
            if col in sub_df.columns
        ]:
            g = sns.FacetGrid(
                sub_df,
                row="samples",
                row_order=sorted(sub_df.samples.unique()),
                col=reg_param,
                col_order=sorted(sub_df[reg_param].unique()),
            )
            g.map_dataframe(
                sns.scatterplot, x="L1", y="L2", hue=subtype_scheme
            )
            g.add_legend()
            plt.savefig(
                save_path.format(
                    context=context,
                    df_label=df_label,
                    which_cells=which_cells,
                    plot_by=subtype_scheme,
                )
            )
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
            plt.savefig(
                save_path.format(
                    context=context,
                    df_label=df_label,
                    which_cells=which_cells,
                    plot_by="cell_line",
                )
            )
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
        param_val_df.drop(
            columns=[
                col
                for col in param_val_df.columns
                if col not in param_cols + reg_params + ["cell_line"]
            ]
        ),
        id_vars=["cell_line"] + reg_params,
        var_name="Parameter",
        value_name="value",
    )
    # Create FacetGrid with one column per parameter, one CV split per row
    g = sns.FacetGrid(
        param_val_df, row="cell_line", col=top_reg_param, aspect=4, sharey=True
    )
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
    figure_filepath: Union[str, Path],
):
    # Set colorbar range
    for samples in param_df.index.get_level_values("samples").unique():
        df = param_df.loc[
            param_df.index.get_level_values("samples") == samples
        ]
        df = df.droplevel("samples", axis="index")
        vlim = df[param_cols].abs().max().max()
        vmin, vmax = -vlim, vlim

        # row_colors = pd.DataFrame(
        #     {
        #         annotation: param_df[annotation].map(
        #             {
        #                 category: color
        #                 for category, color in zip(
        #                 sorted(param_df[annotation].unique()),
        #                 sns.color_palette(palette, param_df[annotation].nunique())
        #             )
        #             }
        #         )
        #         for annotation, palette in zip(
        #         ["PAM50", "LB", "HR_Status", "HER2_Status", "Site", "Disease", "MS_Status"],
        #         ["deep", "viridis", "Blues", "Reds", "mako", "inferno", "plasma"]
        #     )
        #     },
        #     index=param_df.index
        # )

        sns.clustermap(
            data=df[
                df[param_cols].abs().max()[lambda x: x > 0.0].index.tolist()
            ],
            # row_colors=row_colors,
            col_cluster=True,
            vmin=vmin,
            vmax=vmax,
            cmap="vlag",
            xticklabels=True,
            yticklabels=True,
            figsize=(6, 12),
        )
        # Create the legend handles
        # legend_patches = []
        # for annotation in row_colors.columns:
        #     unique_categories = param_df[annotation].unique()
        #     colors = row_colors[annotation].dropna().unique()
        #
        #     # Ensure alignment between unique categories and colors
        #     category_color_map = dict(zip(unique_categories, colors))
        #
        #     # Create patches for legend
        #     for category, color in category_color_map.items():
        #         legend_patches.append(patches.Patch(color=color, label=f"{annotation}: {category}"))

        # Plot the legend
        # plt.legend(
        #     handles=legend_patches,
        #     loc="upper right",
        #     bbox_to_anchor=(5, 1),
        #     # ncol=1,
        #     frameon=False,
        #     fontsize=5
        # )
        figure_filepath.parent.mkdir(exist_ok=True, parents=True)
        plt.tight_layout()
        plt.savefig(str(figure_filepath) + f"_{samples}.pdf")
        plt.close()


def random_forest_importance_plot(
    results_dfs: dict[str, pd.DataFrame], conf, save_or_show: str = "show"
):
    plt.subplots(1, len(list(results_dfs.keys())), figsize=(14, 6))

    for ind, dataset in enumerate(["train", "val"]):
        plt.subplot(1, 2, ind + 1)
        sns.barplot(data=results_dfs[dataset], x="features", y="importances")
        plt.title(f"Feature Importances - {dataset}")
        plt.xlabel("Features")
        plt.ylabel("Importance")
        xticks = plt.gca().get_xticks()
        plt.xticks(
            xticks, rotation=90, labels=results_dfs[dataset]["features"]
        )

    plt.tight_layout()
    if save_or_show == "show":
        plt.show()
    else:
        plt.savefig(
            fig_dir
            / f"{conf.model}"
            / f"{conf.data}"
            / f"{conf.model}.{conf.data}.top_10_feature_importances.svg"
        )
    plt.close()


def plot_rmse_val_cell_lines(df, conf, reg_param):
    g = sns.FacetGrid(
        df[df["sample"].isin(hardest_cell_lines)],
        row="sample",
        row_order=hardest_cell_lines[: len(df.samples.unique())],
        col=reg_param,
        col_order=sorted(df[reg_param].unique()),
        hue="dataset",
        sharex=True,
        sharey=True,
    )
    g.map_dataframe(sns.histplot, x="rmse")
    plt.tight_layout()
    plt.legend()
    figure_filepath = (
        fig_dir
        / f"{conf.model}"
        / f"{conf.data}"
        / f"{conf.model}.{conf.data}.RMSE_top10train_by_cl.svg"
    )
    if not figure_filepath.parent.exists():
        figure_filepath.parent.mkdir(parents=True)
    plt.savefig(figure_filepath)


def plot_mse_param_dev_val_across_splits(
    diffs: pd.DataFrame, conf, context: str, reg_param: str
):
    g = sns.FacetGrid(
        diffs,
        col="cell_line",
        col_order=hardest_cell_lines[: len(diffs.cell_line.unique())],
        hue="samples",
    )
    # Disable auto legend handling and add to each subplot
    g.map_dataframe(
        sns.lineplot, x=reg_param, y="MSE", marker="o", legend=False
    )
    for ax in g.axes.flat:
        handles, labels = ax.get_legend_handles_labels()
        # Order labels in legend by CV number
        sorted_pairs = sorted(
            zip(labels, handles), key=lambda x: int(x[0].split("of")[0])
        )
        sorted_labels, sorted_handles = (
            zip(*sorted_pairs) if sorted_pairs else ([], [])
        )
        if sorted_handles:
            ax.legend(sorted_handles, sorted_labels, title="Samples")
    plt.tight_layout()
    plt.savefig(
        fig_dir
        / conf.model
        / conf.data
        / f"{context}.param_dev_mse.vs.{reg_param}.pdf"
    )
    plt.close()


def plot_param_dev_hist_min_max(
    param_dev_df: pd.DataFrame,
    reg_param: str,
    param_cols: list[str],
    conf,
    fig_name: str,
):
    for fig_label in ["single_jobs", "median"]:
        plt.subplots(param_dev_df[reg_param].nunique(), 1, sharex=True)
        for i, reg_param_value in enumerate(
            sorted(param_dev_df[reg_param].unique())
        ):
            if fig_label == "single_jobs":
                sub_df = param_dev_df[
                    param_dev_df[reg_param] == reg_param_value
                ][param_cols]
            elif fig_label == "median":
                sub_df = (
                    param_dev_df[param_dev_df[reg_param] == reg_param_value]
                    .groupby("cell_line")[param_cols]
                    .median()
                    .reset_index()[param_cols]
                )
            plt.subplot(param_dev_df.sparse_threshold_perc.nunique(), 1, i + 1)
            plt.hist(sub_df.min(), label="min", color="blue")
            plt.hist(sub_df.max(), label="max", color="orange")
            plt.title(reg_param_value)
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            str(fig_dir / conf.model / conf.data / fig_name)
            + f".vs_{reg_param}.{fig_label}.pdf"
        )


def get_annotation_palette(hue_var: str) -> dict:
    if hue_var == "subtype_pam50":
        return {label: shared_category_colors[label] for label in pam50_labels}
    elif hue_var == "subtype_lb":
        # Resolve via lb_to_shared to ensure consistent color mapping
        return {
            label: shared_category_colors[lb_to_shared[label]]
            for label in lb_labels
        }
    else:
        return {}  # fallback: let seaborn pick default palette


# Plotting function factory
def make_annotation_scatter(
    hue_var: str, is_categorical: bool, context_limits: dict
):
    def scatter(data, color, **kwargs):
        ax = plt.gca()
        context = data["context"].iloc[0]
        sample = data["samples"].iloc[0]
        L1_max = context_limits[context]["L1_max"]
        L2_max = context_limits[context]["L2_max"]

        hardest = cv_samples_mapping.get(sample, None)

        # Split data
        data_missing = data[data[hue_var].isna()]
        data_main = data[
            data[hue_var].notna() & (data["cell_line"] != hardest)
        ]
        hardest_data = data[
            (data["cell_line"] == hardest) & data[hue_var].notna()
        ]
        hardest_missing = data[
            (data["cell_line"] == hardest) & data[hue_var].isna()
        ]

        # 1. Plot missing data as gray dots
        if not data_missing.empty:
            sns.scatterplot(
                data=data_missing,
                x="L1",
                y="L2",
                color="lightgray",
                alpha=1.0,
                edgecolor="black",
                linewidth=0.3,
                s=20,
                marker="X",
                ax=ax,
                legend=False,
            )

        if not hardest_missing.empty:
            sns.scatterplot(
                data=hardest_missing,
                x="L1",
                y="L2",
                color="lightgray",
                alpha=1.0,
                edgecolor="black",
                linewidth=0.3,
                s=40,
                marker="s",
                ax=ax,
                legend=False,
            )

        # 2. Plot valid annotated points
        if is_categorical:
            palette = get_annotation_palette(hue_var)
            sns.scatterplot(
                data=data_main,
                x="L1",
                y="L2",
                hue=hue_var,
                palette=palette,
                alpha=1.0,
                edgecolor="black",
                linewidth=0.3,
                s=20,
                ax=ax,
                legend="auto",
            )
            sns.scatterplot(
                data=hardest_data,
                x="L1",
                y="L2",
                hue=hue_var,
                palette=palette,
                marker="s",
                edgecolor="black",
                linewidth=0.3,
                s=40,
                ax=ax,
                legend="auto",
            )
        else:
            vmin = data[hue_var].replace([np.inf, -np.inf], np.nan).min()
            vmax = data[hue_var].replace([np.inf, -np.inf], np.nan).max()
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            cmap = "coolwarm"
            sns.scatterplot(
                data=data_main,
                x="L1",
                y="L2",
                hue=hue_var,
                palette=cmap,
                hue_norm=norm,
                alpha=1.0,
                edgecolor="black",
                linewidth=0.3,
                s=20,
                ax=ax,
                legend=False,
            )
            sns.scatterplot(
                data=hardest_data,
                x="L1",
                y="L2",
                hue=hue_var,
                palette=cmap,
                hue_norm=norm,
                marker="s",
                edgecolor="black",
                linewidth=0.3,
                s=40,
                ax=ax,
                legend=False,
            )

        # Annotate all hardest
        for cl in cv_samples_mapping.values():
            row = data[data["cell_line"] == cl]
            if not row.empty:
                ax.text(
                    row["L1"].values[0] - 0.1,
                    row["L2"].values[0] + 0.1,
                    cl,
                    fontsize=8,
                    color="black",
                )

        # Axis style
        ax.axhline(0, color="black", lw=0.5)
        ax.axvline(0, color="black", lw=0.5)
        ax.set_xlim(-L1_max * 1.1, L1_max * 1.1)
        ax.set_ylim(-L2_max * 1.1, L2_max * 1.1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    return scatter
