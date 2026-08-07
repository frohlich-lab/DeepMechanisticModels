"""Reusable barplot / boxplot functions for paper figures.

This module provides axis-level ``render_*`` functions (for multi-panel
paper layouts) and standalone ``plot_*`` convenience wrappers (for
presentation / exploration).
"""

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Ensure project root is on sys.path so local modules (common, etc.) can be imported
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

warnings.filterwarnings("ignore")

from figure_config import (
    BARPLOT_CONTEXT_ORDER,
    BARPLOT_REF_LINE_COLORS,
    BARPLOT_REF_LINE_STYLES,
    BARPLOT_REF_LINE_WIDTH,
    CONTEXT_COLORS,
    CONTEXT_LABELS,
    CONTEXT_LABELS_2,
    CONTEXT_LABELS_5B,
    LW_THICK,
    MODALITY_COLORS,
    MODEL_COLORS,
    MODEL_GROUPS,
    MODEL_LABELS,
    REF_DISPLAY_MAP,
    configure_axis_spines,
    get_box_color,
)

from common import basedir
from training_configuration import PATHWAYS_BY_FIGURE

# ---------------------------------------------------------------------------
# Grid / layout helpers
# ---------------------------------------------------------------------------


def create_figure_grid(
    nrows,
    ncols,
    figsize=None,
    sharex=False,
    sharey=False,
    wspace=0.3,
    hspace=0.4,
    **subplot_kw,
):
    """Create a figure with a grid of axes for combining multiple panels.

    This is the entry-point for *paper* layout: create a grid, then pass
    individual axes to the ``render_*`` functions below.

    Args:
        nrows: Number of rows in the grid.
        ncols: Number of columns in the grid.
        figsize: Overall figure size (width, height).  Defaults to
            ``(6 * ncols, 5 * nrows)``.
        sharex: Share x-axes across rows.
        sharey: Share y-axes across columns.
        wspace: Horizontal spacing between subplots.
        hspace: Vertical spacing between subplots.
        **subplot_kw: Extra keyword arguments forwarded to
            ``fig.subplots``.

    Returns:
        tuple: ``(fig, axes)`` where *axes* is a 2-D numpy array of
        ``matplotlib.axes.Axes`` (even when *nrows* or *ncols* is 1).
    """
    if figsize is None:
        figsize = (6 * ncols, 5 * nrows)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        sharex=sharex,
        sharey=sharey,
        squeeze=False,
        **subplot_kw,
    )
    fig.subplots_adjust(wspace=wspace, hspace=hspace)
    return fig, axes


def _create_standalone_figure(figsize, column_ratios, n_plot_cols=1):
    """Create the 3/4-column presentation layout used by convenience wrappers.

    Returns:
        tuple: ``(fig, axes_list, ax_legend)``
            *axes_list* has length *n_plot_cols*; *ax_legend* is the
            right-hand axis reserved for the legend.
    """
    if n_plot_cols == 1:
        ratios = column_ratios if column_ratios is not None else [1, 1, 1]
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(1, 3, width_ratios=ratios)
        ax_left = fig.add_subplot(gs[0, 0])
        axes = [fig.add_subplot(gs[0, 1])]
        ax_legend = fig.add_subplot(gs[0, 2])
    else:
        ratios = (
            column_ratios
            if column_ratios is not None
            else [0.5] + [1] * n_plot_cols + [0.5]
        )
        fig = plt.figure(figsize=figsize)
        n_total = 2 + n_plot_cols  # left + plots + right
        gs = fig.add_gridspec(1, n_total, width_ratios=ratios)
        ax_left = fig.add_subplot(gs[0, 0])
        axes = [fig.add_subplot(gs[0, i + 1]) for i in range(n_plot_cols)]
        ax_legend = fig.add_subplot(gs[0, -1])

    ax_left.axis("off")
    ax_legend.axis("off")
    return fig, axes, ax_legend


def add_legend(
    ax_or_fig, handles, labels, *, loc="center left", frameon=False, **kw
):
    """Place a legend on an axis or figure.

    A thin wrapper so callers don't need to repeat boilerplate.
    """
    target = ax_or_fig
    target.legend(handles, labels, loc=loc, frameon=frameon, **kw)


# ---------------------------------------------------------------------------
# Data loading & preparation
# ---------------------------------------------------------------------------


def load_figure_data(figure: str, data: str = "dream_cytof"):
    """Load evaluation data for a specific figure.

    Args:
        figure: Figure name (e.g., 'figure1a', 'figure2', 'figure3')
        data: Dataset name

    Returns:
        DataFrame with all evaluation results for the figure
    """
    evaluations_dir = basedir / "eval"
    pathways = PATHWAYS_BY_FIGURE.get(figure, [])

    dfs = []
    for model in pathways:
        filepath = (
            evaluations_dir / model / data / f"evaluate_all_{figure}.csv"
        )
        if filepath.exists():
            dfs.append(pd.read_csv(filepath, index_col=0).assign(model=model))

    if not dfs:
        raise ValueError(f"No data found for figure '{figure}'")

    return pd.concat(dfs)


def _add_group_padding(hue_order, padding_width=0.5, palette=None):
    """Add padding entries between model groups.

    Parameters
    ----------
    palette : dict or None
        Optional mapping of hue label → colour.  When provided the colour
        for each label is looked up here first, falling back to
        ``MODEL_COLORS``.
    """
    padded_order = []
    padded_palette = []
    _pal = palette or {}

    def _resolve_group(label):
        """Return the MODEL_GROUPS index for *label* (handles composite DMM labels)."""
        for i, group in enumerate(MODEL_GROUPS):
            if label in group:
                return i
        # Composite DMM labels like "DMM · CyTOF" belong to the DMM group
        if " \u00b7 " in label:
            base = label.split(" \u00b7 ", 1)[0]
            for i, group in enumerate(MODEL_GROUPS):
                if base in group:
                    return i
        return None

    current_group_idx = -1
    prev_was_ungrouped = False
    for label in hue_order:
        label_group_idx = _resolve_group(label)

        if (
            label_group_idx is not None
            and label_group_idx != current_group_idx
        ):
            # Insert spacer when crossing from one group to another,
            # or from an ungrouped label (like "DMM") into the first group.
            if current_group_idx != -1 or prev_was_ungrouped:
                padded_order.append("")
                padded_palette.append("white")
            current_group_idx = label_group_idx
            prev_was_ungrouped = False
        elif label_group_idx is None:
            prev_was_ungrouped = True

        padded_order.append(label)
        padded_palette.append(
            _pal.get(label, MODEL_COLORS.get(label, "#333333"))
        )

    return padded_order, padded_palette


def _prepare_data(df, contexts, models, refs, ref_lines):
    """Prepare data for plotting.

    Returns:
        tuple: (plot_df, ref_df, hue_order, contexts, models)
    """
    default_context_order = BARPLOT_CONTEXT_ORDER

    if contexts is None:
        available_contexts = df[df.ref == "DMM"]["context"].unique().tolist()
        contexts = [
            c for c in default_context_order if c in available_contexts
        ]
        contexts += [c for c in available_contexts if c not in contexts]
    if models is None:
        models = df[df.ref == "DMM"]["model"].unique().tolist()

    model_labels = {m: MODEL_LABELS.get(m, m) for m in models}
    if "elasticnet" in refs:
        model_labels["elasticnet"] = MODEL_LABELS.get("elasticnet")

    plot_df = df[
        (df.ref == "DMM")
        & (df.context.isin(contexts))
        & (df.model.isin(models))
    ].copy()
    plot_df["model_label"] = plot_df["model"].map(model_labels)

    if "elasticnet" in refs:
        elasticnet_df = df[
            (df.ref == "elasticnet") & (df.context.isin(contexts))
        ].copy()
        elasticnet_df["model_label"] = MODEL_LABELS.get("elasticnet")
        plot_df = pd.concat([plot_df, elasticnet_df], ignore_index=True)

    ref_df = df[(df.ref.isin(ref_lines)) & (df.context.isin(contexts))]

    hue_order = [model_labels[m] for m in models if m in model_labels]
    if "elasticnet" in refs:
        hue_order.append(MODEL_LABELS.get("elasticnet"))

    return plot_df, ref_df, hue_order, contexts, models


def _apply_display_labels(df):
    """Replace internal ref / context codes with display labels (in-place copy)."""
    df = df.copy()
    df["ref"] = df["ref"].replace(REF_DISPLAY_MAP)
    all_context_labels = {
        **CONTEXT_LABELS,
        **CONTEXT_LABELS_2,
        **CONTEXT_LABELS_5B,
    }
    for context, label in all_context_labels.items():
        df["context"] = df["context"].replace({context: label})
    df["context"] = df["context"].replace({"cytof_init": "CyTOF"})
    return df


def _split_val_test(df, ds):
    """Return (data_df, title_suffix) for a dataset split."""
    if ds == "val":
        data_df = df[
            (df.dataset == "val") & (~df.samples.str.startswith("all"))
        ].copy()
        suffix = r" ($\\bf{Validation}$ Set)"
    elif ds == "test":
        data_df = df[
            (df.dataset == "val") & (df.samples.str.startswith("all"))
        ].copy()
        suffix = r" ($\\bf{Test}$ Set)"
    elif ds == "train":
        data_df = df[
            (df.dataset == "train") & (df.samples.str.startswith("all"))
        ].copy()
        suffix = r" ($\\bf{Training}$ Set)"
    else:
        raise ValueError(f"Unknown dataset: {ds}")
    return data_df, suffix


def _print_summary(datasets_to_plot, data_by_dataset):
    """Print summary statistics about the analysed dataset."""
    print("=" * 60)
    print("Dataset Summary Statistics")
    print("=" * 60)
    for ds in datasets_to_plot:
        data = data_by_dataset[ds]
        pdf = data["plot_df"]
        rdf = data["ref_df"]
        split_label = "Validation" if ds == "val" else "Test"
        print(f"\n--- {split_label} Set ---")
        print(f"  Total rows (plot data):     {len(pdf)}")
        for col, label in [
            ("samples", "Unique cell lines (samples)"),
            ("model", "Unique models"),
            ("context", "Unique contexts"),
            ("job", "Unique jobs/runs"),
            ("ref", "Unique ref types"),
            ("features", "Unique feature sets"),
        ]:
            val = pdf[col].nunique() if col in pdf.columns else "N/A"
            print(f"  {label + ':':<30s}{val}")
        if "context" in pdf.columns:
            print(
                f"  Contexts:                   {sorted(pdf['context'].unique().tolist())}"
            )
        if "model_label" in pdf.columns:
            print(
                f"  Model labels:               {sorted(pdf['model_label'].dropna().unique().tolist())}"
            )
        if len(rdf) > 0:
            print(
                f"  Reference lines:            {sorted(rdf['ref'].unique().tolist())}"
            )
            for ref_type in sorted(rdf["ref"].unique()):
                ref_mean = rdf[rdf.ref == ref_type]["rmse"].mean()
                ref_std = rdf[rdf.ref == ref_type]["rmse"].std()
                print(f"    {ref_type}: RMSE = {ref_mean:.4f} ± {ref_std:.4f}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Axis-level rendering functions  (the key building blocks)
# ---------------------------------------------------------------------------


def _configure_boxplot_axis(
    ax, xlabel, ylabel, ylim, title, rotate_xticks=True
):
    """Configure axis properties for boxplot subplots."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim)
    ax.set_title(title)
    if rotate_xticks:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    configure_axis_spines(ax)


def _recolor_dmm_boxes(ax, contexts, hue_order, dmm_label="DMM"):
    """Recolour DMM boxes by their context after seaborn has drawn them.

    Seaborn keeps ``"DMM"`` as a single hue level (one slot per context),
    so no ghost columns are created.  This function walks the artist list
    and paints each DMM box with the colour of the context it belongs to.

    Works with seaborn >= 0.12 which stores boxes in ``BoxPlotContainer``
    objects (one container per hue level, each containing one box per
    context).
    """
    if dmm_label not in hue_order:
        return

    dmm_idx = hue_order.index(dmm_label)

    # seaborn >= 0.12: one BoxPlotContainer per hue level, in hue_order
    containers = [c for c in ax.containers if hasattr(c, "boxes")]
    if dmm_idx < len(containers):
        dmm_container = containers[dmm_idx]
        for ctx_i, ctx in enumerate(contexts):
            if ctx_i < len(dmm_container.boxes):
                color = CONTEXT_COLORS.get(
                    ctx, CONTEXT_COLORS.get("CyTOF", "#333333")
                )
                dmm_container.boxes[ctx_i].set_facecolor(color)


def _plot_boxplot(ax, plot_df, contexts, hue_order, group_padding=True):
    """Plot a grouped boxplot on the given axis.

    ``"DMM"`` is kept as a single hue level so seaborn only reserves one
    slot per context.  After drawing, the DMM boxes are recoloured by
    context so that each modality gets its own colour.

    Non-DMM models are coloured via their ``MODEL_COLORS`` entry.
    """
    palette = {h: MODEL_COLORS.get(h, "#333333") for h in hue_order}
    # Use a neutral placeholder for DMM; it will be recoloured afterwards
    if "DMM" in palette:
        palette["DMM"] = CONTEXT_COLORS.get("CyTOF", "#333333")

    if group_padding:
        padded_hue_order, padded_palette = _add_group_padding(
            hue_order,
            palette=palette,
        )

        plot_df = plot_df.copy()
        for _ in [h for h in padded_hue_order if h == ""]:
            for ctx in contexts:
                dummy_row = pd.DataFrame(
                    {"context": [ctx], "rmse": [np.nan], "model_label": [""]}
                )
                plot_df = pd.concat([plot_df, dummy_row], ignore_index=True)
    else:
        padded_hue_order = hue_order
        padded_palette = palette

    sns.boxplot(
        data=plot_df,
        x="context",
        y="rmse",
        hue="model_label",
        hue_order=padded_hue_order,
        order=contexts,
        palette=padded_palette,
        ax=ax,
    )

    _recolor_dmm_boxes(ax, contexts, padded_hue_order)


def _add_reference_lines(ax, ref_df, ref_lines, with_uncertainty=False):
    """Add horizontal reference lines to the axis."""
    for ref_type in ref_lines:
        ref_subset = ref_df[ref_df.ref == ref_type]
        if len(ref_subset) > 0:
            mean_rmse = ref_subset["rmse"].mean()
            std_rmse = ref_subset["rmse"].std()
            color = BARPLOT_REF_LINE_COLORS.get(ref_type, "gray")
            linestyle = BARPLOT_REF_LINE_STYLES.get(ref_type, "--")
            if with_uncertainty:
                ax.axhspan(
                    mean_rmse - std_rmse,
                    mean_rmse + std_rmse,
                    color=color,
                    alpha=0.15,
                    linewidth=0,
                )
            ax.axhline(
                y=mean_rmse,
                color=color,
                linestyle=linestyle,
                linewidth=BARPLOT_REF_LINE_WIDTH,
                label=ref_type,
            )


def _run_model_wilcoxon_tests(
    plot_df, hue_order, baseline_label, contexts, alpha=0.05
):
    """Run paired Wilcoxon tests comparing each non-baseline model to the baseline.

    Tests are run per context, paired by job.  P-values are FDR-corrected
    (Benjamini–Hochberg) across all (context × model) tests.

    Returns:
        tuple: ``(stats_df, sig_lookup)`` where *sig_lookup* maps
        ``(context, model_label)`` → ``(adjusted_p, direction)``.
    """
    from scipy.stats import false_discovery_control, wilcoxon

    if baseline_label not in hue_order:
        return None, {}

    non_baseline = [h for h in hue_order if h != baseline_label and h != ""]

    stat_results = []
    for ctx in contexts:
        context_df = plot_df[
            (plot_df.context == ctx)
            & (plot_df.model_label.isin([baseline_label, *non_baseline]))
        ].copy()
        if context_df.empty or "job" not in context_df.columns:
            continue

        pivot = context_df.pivot_table(
            index="job",
            columns="model_label",
            values="rmse",
            aggfunc="first",
        )
        if baseline_label not in pivot.columns:
            continue

        for model_label in non_baseline:
            if model_label not in pivot.columns:
                continue

            paired = pivot[[baseline_label, model_label]].dropna()
            if paired.empty:
                continue

            baseline_vals = paired[baseline_label].to_numpy()
            model_vals = paired[model_label].to_numpy()
            if np.array_equal(model_vals, baseline_vals):
                continue

            try:
                _, p = wilcoxon(baseline_vals, model_vals)
            except ValueError:
                p = 1.0

            direction = (
                "lower"
                if np.mean(model_vals) < np.mean(baseline_vals)
                else "higher"
            )
            stat_results.append((ctx, model_label, p, direction))

    if not stat_results:
        return None, {}

    raw_pvals = np.array([r[2] for r in stat_results])
    adjusted_pvals = false_discovery_control(raw_pvals, method="bh")

    sig_lookup = {}
    for (ctx, model_label, _, direction), adj_p in zip(
        stat_results, adjusted_pvals, strict=True
    ):
        sig_lookup[(ctx, model_label)] = (adj_p, direction)

    stats_df = pd.DataFrame(
        stat_results,
        columns=["context", "model_label", "p_raw", "direction"],
    )
    stats_df["p_adjusted"] = adjusted_pvals
    stats_df["significant"] = stats_df["p_adjusted"] < alpha
    stats_df = stats_df.sort_values("p_adjusted")
    return stats_df, sig_lookup


def _annotate_model_significance(
    ax, plot_df, contexts, padded_hue_order, sig_lookup, alpha=0.05
):
    """Add significance stars above boxes for model-comparison plots."""

    def _pval_to_stars(p):
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < alpha:
            return "*"
        return ""

    n_hues = len(padded_hue_order)
    hue_width = 0.8
    offsets = {
        hue: (i - (n_hues - 1) / 2) * (hue_width / n_hues)
        for i, hue in enumerate(padded_hue_order)
    }

    for i_ctx, ctx in enumerate(contexts):
        for model_label in padded_hue_order:
            if model_label == "":
                continue
            key = (ctx, model_label)
            if key not in sig_lookup:
                continue
            adj_p, direction = sig_lookup[key]
            stars = _pval_to_stars(adj_p)
            if not stars:
                continue
            x_pos = i_ctx + offsets.get(model_label, 0)
            aug_data = plot_df[
                (plot_df.context == ctx) & (plot_df.model_label == model_label)
            ]["rmse"]
            if aug_data.empty:
                continue
            q3 = aug_data.quantile(0.75)
            iqr = q3 - aug_data.quantile(0.25)
            whisker_top = min(aug_data.max(), q3 + 1.5 * iqr)
            y_pos = whisker_top + 0.005
            color = "green" if direction == "lower" else "red"
            ax.text(
                x_pos,
                y_pos,
                stars,
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color=color,
            )


def render_model_comparison(
    ax,
    df,
    *,
    ylim=(0.15, 1.0),
    refs=("DMM", "elasticnet"),
    ref_lines=("negative control", "positive control"),
    contexts=None,
    models=None,
    dataset="test",
    with_reference_uncertainty=False,
    title=None,
    xlabel="Input Features (Baseline)",
    ylabel="RMSE",
    show_legend=False,
    group_padding=True,
    show_stats=False,
    baseline_model=None,
    alpha=0.05,
):
    """Render a model-comparison boxplot onto an existing axis.

    This is the *axis-level* function: it draws on *ax* and does **not**
    create its own figure.  Use it in grid / paper layouts.

    Args:
        ax: Matplotlib axis to draw on.
        df: Raw DataFrame as returned by ``load_figure_data``.
        ylim, refs, ref_lines, contexts, models, dataset,
        with_reference_uncertainty: Same semantics as ``plot_model_comparison``.
        title: Axis title.`.
        xlabel: X-axis label.
        ylabel: Y-axis label (set to ``""`` to hide on non-first columns).
        show_legend: Whether to show the legend on this axis.
        group_padding: Insert blank spacers between model groups (default True).
            Set to False for panels with a single model group.
        show_stats: Run paired Wilcoxon tests comparing each model to
            *baseline_model* and annotate significant differences with stars.
        baseline_model: Model name (internal key) used as the reference for
            statistical comparisons.  Required when *show_stats* is True.
        alpha: Significance threshold (after FDR correction).

    Returns:
        tuple: ``(handles, labels)`` when *show_stats* is False (default).
        ``(handles, labels, stats_df)`` when *show_stats* is True, where
        *stats_df* is a DataFrame of paired Wilcoxon test results.
    """
    df = _apply_display_labels(df)

    data_df, _ = _split_val_test(df, dataset)
    data_df.features.fillna("None", inplace=True)
    plot_df, ref_df, hue_order, contexts, models = _prepare_data(
        data_df,
        contexts,
        models,
        refs,
        ref_lines,
    )

    padded_hue_order, _ = (
        _add_group_padding(hue_order) if group_padding else (hue_order, None)
    )

    _plot_boxplot(
        ax, plot_df, contexts, hue_order, group_padding=group_padding
    )
    _add_reference_lines(
        ax, ref_df, ref_lines, with_uncertainty=with_reference_uncertainty
    )

    stats_df = None
    if show_stats and baseline_model is not None:
        baseline_label = MODEL_LABELS.get(baseline_model, baseline_model)
        stats_df, sig_lookup = _run_model_wilcoxon_tests(
            plot_df,
            hue_order,
            baseline_label,
            contexts,
            alpha=alpha,
        )
        if sig_lookup:
            _annotate_model_significance(
                ax,
                plot_df,
                contexts,
                padded_hue_order,
                sig_lookup,
                alpha,
            )

    _title = title if title is not None else ""
    _configure_boxplot_axis(ax, xlabel, ylabel, ylim, _title)

    handles, labels = ax.get_legend_handles_labels()
    if not show_legend:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    if show_stats:
        return handles, labels, stats_df
    return handles, labels


def render_delta_model_comparison(
    ax,
    df,
    *,
    refs=("DMM",),
    contexts=None,
    models=None,
    baseline_model=None,
    dataset="test",
    ylim=None,
    title=None,
    xlabel="Input Features (Baseline)",
    ylabel="\u0394RMSE",
    show_legend=False,
    show_stats=True,
    alpha=0.05,
):
    """Render a delta-RMSE model-comparison boxplot onto *ax*.

    Subtracts the *baseline_model* RMSE from every other model per job,
    then shows the distribution of delta-RMSE values.

    Returns:
        ``(handles, labels, stats_df)``
    """
    from scipy.stats import false_discovery_control, wilcoxon

    df = _apply_display_labels(df)
    data_df, _ = _split_val_test(df, dataset)
    data_df.features.fillna("None", inplace=True)
    plot_df, ref_df, hue_order, contexts, resolved_models = _prepare_data(
        data_df,
        contexts,
        models,
        refs,
        ("negative control", "positive control"),
    )

    if baseline_model is None:
        raise ValueError("baseline_model is required for delta-RMSE plots")
    baseline_label = MODEL_LABELS.get(baseline_model, baseline_model)
    non_baseline = [h for h in hue_order if h != baseline_label and h != ""]

    # Compute delta-RMSE per (context, job)
    delta_rows = []
    for ctx in contexts:
        ctx_df = plot_df[plot_df.context == ctx]
        pivot = ctx_df.pivot_table(
            index="job", columns="model_label", values="rmse", aggfunc="first"
        )
        if baseline_label not in pivot.columns:
            continue
        for ml in non_baseline:
            if ml not in pivot.columns:
                continue
            paired = pivot[[baseline_label, ml]].dropna()
            for job, row in paired.iterrows():
                delta_rows.append(
                    {
                        "context": ctx,
                        "model_label": ml,
                        "job": job,
                        "delta_rmse": row[ml] - row[baseline_label],
                    }
                )

    delta_df = pd.DataFrame(delta_rows)
    if delta_df.empty:
        return [], [], None

    palette = {ml: MODEL_COLORS.get(ml, "#333333") for ml in non_baseline}

    sns.boxplot(
        data=delta_df,
        x="context",
        y="delta_rmse",
        hue="model_label",
        hue_order=non_baseline,
        order=contexts,
        palette=palette,
        ax=ax,
    )
    ax.axhline(
        0,
        color="black",
        linestyle="-",
        linewidth=LW_THICK,
        alpha=1.0,
        zorder=0,
    )
    ax.grid(False, axis="y")

    # One-sided Wilcoxon tests (H1: delta < 0)
    stats_df = None
    if show_stats:
        stat_results = []
        for ctx in contexts:
            ctx_delta = delta_df[delta_df.context == ctx]
            for ml in non_baseline:
                vals = ctx_delta[ctx_delta.model_label == ml][
                    "delta_rmse"
                ].values
                if len(vals) == 0:
                    continue
                try:
                    _, p = wilcoxon(vals, alternative="less")
                except ValueError:
                    p = 1.0
                stat_results.append((ctx, ml, p))

        if stat_results:
            raw_pvals = np.array([r[2] for r in stat_results])
            adjusted_pvals = false_discovery_control(raw_pvals, method="bh")
            sig_lookup = {}
            for (ctx, ml, _), adj_p in zip(
                stat_results, adjusted_pvals, strict=True
            ):
                sig_lookup[(ctx, ml)] = adj_p

            stats_df = pd.DataFrame(
                stat_results, columns=["context", "model_label", "p_raw"]
            )
            stats_df["p_adjusted"] = adjusted_pvals
            stats_df["significant"] = stats_df["p_adjusted"] < alpha

            delta_summary = (
                delta_df.groupby(["context", "model_label"])["delta_rmse"]
                .agg(
                    delta_rmse_mean="mean",
                    delta_rmse_std="std",
                    delta_rmse_median="median",
                )
                .reset_index()
            )
            stats_df = stats_df.merge(
                delta_summary, on=["context", "model_label"], how="left"
            )
            stats_df = stats_df.sort_values("p_adjusted")

            # Annotate significance
            n_hues = len(non_baseline)
            hue_width = 0.8
            offsets = {
                h: (i - (n_hues - 1) / 2) * (hue_width / n_hues)
                for i, h in enumerate(non_baseline)
            }
            for i_ctx, ctx in enumerate(contexts):
                for ml in non_baseline:
                    adj_p = sig_lookup.get((ctx, ml))
                    if adj_p is None:
                        continue
                    if adj_p < 0.001:
                        stars = "***"
                    elif adj_p < 0.01:
                        stars = "**"
                    elif adj_p < alpha:
                        stars = "*"
                    else:
                        continue
                    x_pos = i_ctx + offsets.get(ml, 0)
                    aug_data = delta_df[
                        (delta_df.context == ctx)
                        & (delta_df.model_label == ml)
                    ]["delta_rmse"]
                    q3 = aug_data.quantile(0.75)
                    iqr = q3 - aug_data.quantile(0.25)
                    whisker_top = min(aug_data.max(), q3 + 1.5 * iqr)
                    ax.text(
                        x_pos,
                        whisker_top + 0.005,
                        stars,
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        fontweight="bold",
                        color="green",
                    )

    _configure_boxplot_axis(ax, xlabel, ylabel, ylim, title or "")
    _yabs = max(abs(v) for v in ax.get_ylim())
    ax.set_ylim(-_yabs, _yabs)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", which="both", length=0)
    handles, labels = ax.get_legend_handles_labels()
    if not show_legend:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    return handles, labels, stats_df


def plot_delta_model_comparison(
    df,
    *,
    refs=("DMM",),
    contexts=None,
    models=None,
    baseline_model=None,
    dataset="test",
    ylim=None,
    figsize=(10, 6),
    column_ratios=None,
    save_path=None,
    show_stats=True,
    alpha=0.05,
):
    """Standalone delta-RMSE model comparison (presentation layout)."""
    fig, axes, ax_legend = _create_standalone_figure(
        figsize,
        column_ratios or [0.1, 1, 0.25],
        n_plot_cols=1,
    )
    handles, labels, stats_df = render_delta_model_comparison(
        axes[0],
        df,
        refs=refs,
        contexts=contexts,
        models=models,
        baseline_model=baseline_model,
        dataset=dataset,
        ylim=ylim,
        show_stats=show_stats,
        alpha=alpha,
    )
    add_legend(ax_legend, handles, labels)
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, axes[0], stats_df


def render_figure6_features(
    ax,
    *,
    ylim=(0.25, 0.75),
    ref_lines=("negative control", "positive control"),
    refs=("DMM + EGFR transcript levels", "linear regression"),
    title=None,
    xlabel="Number of Cell Lines Used for Training",
    ylabel="RMSE",
    show_legend=False,
):
    """Render figure 6 (data-ablation line plot) onto an existing axis.

    Returns:
        tuple: ``(handles, labels)``
    """
    df = load_figure_data("figure6")
    df["ref"] = df["ref"].replace(
        {
            "avg_model": "negative control",
            "sample": "positive control",
            "elasticnet": "linear regression",
            "DMM": "DMM + EGFR transcript levels",
        }
    )

    data_df = df[
        (df.dataset == "val")
        & df.ref.isin(refs)
        & df.samples.str.contains("pct")
    ].copy()
    data_df[["pct_missing", "seed"]] = data_df["samples"].str.extract(
        r"_(\d+)pct_(\d+)"
    )
    data_df["pct_missing"] = data_df["pct_missing"].astype(int)
    data_df["seed"] = data_df["seed"].astype(int)
    data_df["n_cell_lines"] = (
        (data_df["pct_missing"] / 100 * 61).round().astype(int)
    )
    data_df = data_df.groupby(["n_cell_lines", "seed", "ref"], as_index=False)[
        "rmse"
    ].mean()

    ref_df = df[(df.dataset == "val") & df.ref.isin(ref_lines)]

    hue_order = [r for r in refs if r in data_df["ref"].unique()]
    palette = [get_box_color(label) for label in hue_order]

    sns.lineplot(
        data=data_df,
        x="n_cell_lines",
        y="rmse",
        hue="ref",
        hue_order=hue_order,
        marker="o",
        errorbar="sd",
        palette=palette,
        ax=ax,
    )
    _add_reference_lines(ax, ref_df, ref_lines)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim)
    configure_axis_spines(ax)

    handles, labels = ax.get_legend_handles_labels()
    if not show_legend:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    return handles, labels


def render_figure3b_nhidden(
    ax,
    *,
    ylim=(0.25, 0.75),
    ref_lines=("negative control", "positive control"),
    models=None,
    title=None,
    xlabel="Number of Hidden Units",
    ylabel="RMSE",
    show_legend=False,
):
    """Render figure 3b (hidden-unit scan line plot) onto an existing axis.

    Returns:
        tuple: ``(handles, labels)``
    """
    df = load_figure_data("figure3b")
    df["ref"] = df["ref"].replace(
        {
            "avg_model": "negative control",
            "sample": "positive control",
            "elasticnet": "linear regression",
        }
    )
    df["model_label"] = df["model"].map(MODEL_LABELS).fillna(df["model"])

    if models is None:
        models = df[df.ref == "DMM"]["model"].dropna().unique().tolist()
    model_label_list = [MODEL_LABELS.get(m, m) for m in models]

    data_df = df[
        (df.dataset == "val")
        & (df.ref == "DMM")
        & df.model.isin(models)
        & df.n_hidden.notna()
    ].copy()
    data_df["n_hidden"] = data_df["n_hidden"].astype(int)

    ref_df = df[(df.dataset == "val") & df.ref.isin(ref_lines)]

    hue_order = [
        ml for ml in model_label_list if ml in data_df["model_label"].unique()
    ]
    palette = [get_box_color(label) for label in hue_order]

    sns.lineplot(
        data=data_df,
        x="n_hidden",
        y="rmse",
        hue="model_label",
        hue_order=hue_order,
        marker="o",
        errorbar="sd",
        palette=palette,
        ax=ax,
    )
    _add_reference_lines(ax, ref_df, ref_lines)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim)
    ax.set_xticks(sorted(data_df["n_hidden"].unique()))
    ax.set_title(title or "")
    configure_axis_spines(ax)

    handles, labels = ax.get_legend_handles_labels()
    if not show_legend:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    return handles, labels


def render_nhidden_trainval(
    ax,
    *,
    models=None,
    highlights=None,
    annotate_nh=False,
    max_nhidden=None,
    title=None,
    xlabel="Train RMSE",
    ylabel="Val RMSE",
    min_s=30,
    max_s=180,
):
    """Render n_hidden scan as train-vs-val XY scatter (like the data-ablation plot).

    Each point is one n_hidden value (averaged over jobs); marker size encodes
    the number of hidden units.  A connecting line shows the trajectory as
    n_hidden increases.

    Args:
        highlights: Optional dict mapping model name → one or more n_hidden
            values to highlight with a black ring.  Each value may be a single
            int or a tuple/list of ints, e.g.
            ``{"EGFR_MAPK__logobs": 4,
               "EGFR_MAPK__logobs_tegfr_aggavg": (3, 5)}``.
        max_nhidden: If given, only include n_hidden values up to this maximum.

    Returns:
        tuple: ``(color_handles, color_labels, size_handles, size_labels)``
    """
    df = load_figure_data("figure3b")
    df["ref"] = df["ref"].replace(
        {
            "avg_model": "negative control",
            "sample": "positive control",
            "elasticnet": "linear regression",
        }
    )
    df["model_label"] = df["model"].map(MODEL_LABELS).fillna(df["model"])

    if models is None:
        models = df[df.ref == "DMM"]["model"].dropna().unique().tolist()
    model_label_list = [MODEL_LABELS.get(m, m) for m in models]

    dmm = df[
        (df.ref == "DMM") & df.model.isin(models) & df.n_hidden.notna()
    ].copy()
    dmm["n_hidden"] = dmm["n_hidden"].astype(int)
    if max_nhidden is not None:
        dmm = dmm[dmm["n_hidden"] <= max_nhidden]

    train_df = dmm[dmm.dataset == "train"]
    val_df = dmm[dmm.dataset == "val"]

    # Average RMSE per (model_label, n_hidden, job) then merge
    train_agg = train_df.groupby(
        ["model_label", "n_hidden", "job"], as_index=False
    )["rmse"].mean()
    val_agg = val_df.groupby(
        ["model_label", "n_hidden", "job"], as_index=False
    )["rmse"].mean()
    merged = train_agg.merge(
        val_agg,
        on=["model_label", "n_hidden", "job"],
        suffixes=("_train", "_val"),
    )

    # Mean trajectory per (model_label, n_hidden) – averaged over jobs
    mean_traj = (
        merged.groupby(["model_label", "n_hidden"], as_index=False)[
            ["rmse_train", "rmse_val"]
        ]
        .mean()
        .sort_values("n_hidden")
    )

    # Marker sizes: encode n_hidden
    nh_vals = mean_traj["n_hidden"]
    if nh_vals.max() > nh_vals.min():
        scaled = (
            min_s
            + (nh_vals - nh_vals.min())
            / (nh_vals.max() - nh_vals.min())
            * (max_s - min_s)
        ) / 2
    else:
        scaled = pd.Series(min_s, index=nh_vals.index)

    # Resolve highlights to model_label → list of n_hidden values
    hl_by_label = {}
    if highlights:
        for model_key, nh_val in highlights.items():
            ml = MODEL_LABELS.get(model_key, model_key)
            hl_by_label[ml] = (
                list(nh_val) if hasattr(nh_val, "__iter__") else [nh_val]
            )

    # Plot per model
    handles, labels = [], []
    for ml in model_label_list:
        color = get_box_color(ml)
        mask = mean_traj["model_label"] == ml
        traj = mean_traj[mask]
        sizes = scaled[mask]
        if traj.empty:
            continue

        ax.plot(
            traj["rmse_train"],
            traj["rmse_val"],
            color=color,
            lw=1.2,
            alpha=0.5,
            zorder=1,
        )
        ax.scatter(
            traj["rmse_train"],
            traj["rmse_val"],
            s=sizes,
            alpha=0.7,
            color=color,
            edgecolors="white",
            linewidths=0.5,
            zorder=2,
        )

        # Highlight specific n_hidden values with a black ring
        if ml in hl_by_label:
            for nh_hl in hl_by_label[ml]:
                hl_row = traj[traj["n_hidden"] == nh_hl]
                if not hl_row.empty:
                    hl_size = scaled[hl_row.index]
                    ax.scatter(
                        hl_row["rmse_train"],
                        hl_row["rmse_val"],
                        s=hl_size * 1.8,
                        facecolors="none",
                        edgecolors="black",
                        linewidths=2.0,
                        zorder=3,
                    )

        # Annotate each point with its n_hidden value
        if annotate_nh:
            for _, row in traj.iterrows():
                ax.annotate(
                    str(int(row["n_hidden"])),
                    (row["rmse_train"], row["rmse_val"]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=7,
                    color=color,
                    ha="left",
                    va="bottom",
                )

        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markersize=8,
            )
        )
        labels.append(ml)

    # Diagonal reference
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
    hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], ls="--", color="grey", lw=0.8, zorder=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or "n_hidden Scan")
    configure_axis_spines(ax)

    # Size legend entries
    unique_nh = sorted(mean_traj["n_hidden"].unique())
    if len(unique_nh) > 3:
        legend_nh = [
            unique_nh[0],
            unique_nh[len(unique_nh) // 2],
            unique_nh[-1],
        ]
    else:
        legend_nh = unique_nh
    nh_min, nh_max = nh_vals.min(), nh_vals.max()
    size_handles = [
        ax.scatter(
            [],
            [],
            s=min_s + (n - nh_min) / max(nh_max - nh_min, 1) * (max_s - min_s),
            color="grey",
            edgecolors="white",
            linewidths=0.5,
        )
        for n in legend_nh
    ]
    size_labels = [f"$n_h$ = {n}" for n in legend_nh]

    return handles, labels, size_handles, size_labels


def _prepare_figure5b_data():
    """Load and prepare delta-RMSE data for figure 5b augmentation plots.

    Returns delta-RMSE values (augmentation − baseline) per job.
    """
    from training_configuration import (
        EXTRA_MARKERS_5B_PROT,
        EXTRA_MARKERS_5B_TX,
    )

    df = load_figure_data("figure5b")
    df["ref"] = df["ref"].replace(REF_DISPLAY_MAP)

    data_df = df[
        (df.dataset == "val") & df.samples.str.startswith("all")
    ].copy()
    ref_df = data_df[
        data_df.ref.isin(("negative control", "positive control"))
    ]
    dmm_df = data_df[data_df.ref == "DMM"].copy()

    def _parse_context(ctx):
        if ctx == "cytof_init":
            return "CyTOF (baseline)", "baseline"
        for marker in EXTRA_MARKERS_5B_TX:
            if ctx == f"cytof_init_plus_t{marker}":
                return marker, "transcript"
        for marker in EXTRA_MARKERS_5B_PROT:
            if ctx == f"cytof_init_plus_p{marker}":
                return marker, "protein"
        return ctx, "other"

    parsed = dmm_df["context"].apply(_parse_context)
    dmm_df["marker"] = parsed.apply(lambda x: x[0])
    dmm_df["augmentation_type"] = parsed.apply(lambda x: x[1])

    _baseline_vals = (
        dmm_df[dmm_df.augmentation_type == "baseline"]
        .sort_values("job")["rmse"]
        .values
    )
    _noop_keys = set()
    for _m, _at in [
        *[(_m, "transcript") for _m in EXTRA_MARKERS_5B_TX],
        *[(_m, "protein") for _m in EXTRA_MARKERS_5B_PROT],
    ]:
        _av = (
            dmm_df[(dmm_df.marker == _m) & (dmm_df.augmentation_type == _at)]
            .sort_values("job")["rmse"]
            .values
        )
        if len(_av) == len(_baseline_vals) and np.array_equal(
            _av, _baseline_vals
        ):
            _noop_keys.add((_m, _at))
    if _noop_keys:
        dmm_df = dmm_df[
            ~dmm_df.apply(
                lambda r: (r["marker"], r["augmentation_type"]) in _noop_keys,
                axis=1,
            )
        ]

    # Compute delta-RMSE: augmentation − baseline per job
    baseline_by_job = (
        dmm_df[dmm_df.augmentation_type == "baseline"]
        .set_index("job")["rmse"]
        .rename("baseline_rmse")
    )
    dmm_df = dmm_df[dmm_df.augmentation_type != "baseline"].copy()
    dmm_df = dmm_df.merge(baseline_by_job, on="job", how="left")
    dmm_df["delta_rmse"] = dmm_df["rmse"] - dmm_df["baseline_rmse"]

    return dmm_df, ref_df


def _run_wilcoxon_tests(dmm_df, markers_tx, markers_prot, alpha=0.05):
    """Run one-sample Wilcoxon signed-rank tests on delta-RMSE with FDR correction."""
    from scipy.stats import false_discovery_control, wilcoxon

    stat_results = []
    for marker, aug_type in [
        *[(m, "transcript") for m in sorted(markers_tx)],
        *[(m, "protein") for m in sorted(markers_prot)],
    ]:
        delta_vals = (
            dmm_df[
                (dmm_df.marker == marker)
                & (dmm_df.augmentation_type == aug_type)
            ]
            .sort_values("job")["delta_rmse"]
            .values
        )
        if len(delta_vals) > 0:
            try:
                _, p = wilcoxon(delta_vals, alternative="less")
            except ValueError:
                p = 1.0
            stat_results.append((marker, aug_type, p))

    if not stat_results:
        return None, {}

    raw_pvals = np.array([r[2] for r in stat_results])
    adjusted_pvals = false_discovery_control(raw_pvals, method="bh")

    sig_lookup = {}
    for (marker, aug_type, _), adj_p in zip(
        stat_results, adjusted_pvals, strict=True
    ):
        sig_lookup[(marker, aug_type)] = adj_p

    stats_df = pd.DataFrame(
        stat_results, columns=["marker", "augmentation_type", "p_raw"]
    )
    stats_df["p_adjusted"] = adjusted_pvals
    stats_df["significant"] = stats_df["p_adjusted"] < alpha

    delta_summary = (
        dmm_df.groupby(["marker", "augmentation_type"])["delta_rmse"]
        .agg(
            delta_rmse_mean="mean",
            delta_rmse_std="std",
            delta_rmse_median="median",
        )
        .reset_index()
    )
    stats_df = stats_df.merge(
        delta_summary, on=["marker", "augmentation_type"], how="left"
    )
    stats_df = stats_df.sort_values("p_adjusted")

    return stats_df, sig_lookup


def _annotate_significance(
    ax, dmm_df, marker_order, hue_order_full, sig_lookup, alpha=0.05
):
    """Add significance stars to a boxplot axis."""

    def _pval_to_stars(p):
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < alpha:
            return "*"
        return ""

    n_hues = len(hue_order_full)
    hue_width = 0.8
    offsets = {
        hue: (i - (n_hues - 1) / 2) * (hue_width / n_hues)
        for i, hue in enumerate(hue_order_full)
    }

    for i_marker, marker in enumerate(marker_order):
        for aug_type in hue_order_full:
            key = (marker, aug_type)
            if key in sig_lookup:
                adj_p = sig_lookup[key]
                stars = _pval_to_stars(adj_p)
                if stars:
                    x_pos = i_marker + offsets[aug_type]
                    aug_data = dmm_df[
                        (dmm_df.marker == marker)
                        & (dmm_df.augmentation_type == aug_type)
                    ]["delta_rmse"]
                    q3 = aug_data.quantile(0.75)
                    iqr = q3 - aug_data.quantile(0.25)
                    whisker_top = min(aug_data.max(), q3 + 1.5 * iqr)
                    y_pos = whisker_top + 0.005
                    ax.text(
                        x_pos,
                        y_pos,
                        stars,
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        fontweight="bold",
                        color="green",
                    )


def render_figure5b_augmentations(
    ax,
    *,
    ylim=None,
    show_stats=True,
    alpha=0.05,
    title=None,
    xlabel="Marker Augmentation",
    ylabel="\u0394RMSE",
    show_legend=False,
):
    """Render figure 5b (feature augmentations delta-RMSE boxplot) onto an existing axis.

    Returns:
        tuple: ``(handles, labels, stats_df)``
    """
    from training_configuration import (
        EXTRA_MARKERS_5B_PROT,
        EXTRA_MARKERS_5B_TX,
    )

    dmm_df, ref_df = _prepare_figure5b_data()

    marker_order = sorted(dmm_df["marker"].unique())

    hue_order_full = ["transcript", "protein"]
    palette_full = {k: MODALITY_COLORS[k] for k in hue_order_full}

    sns.boxplot(
        data=dmm_df,
        x="marker",
        y="delta_rmse",
        hue="augmentation_type",
        hue_order=hue_order_full,
        order=marker_order,
        palette=palette_full,
        ax=ax,
    )
    ax.axhline(
        0,
        color="black",
        linestyle="-",
        linewidth=LW_THICK,
        alpha=1.0,
        zorder=0,
    )
    ax.grid(False, axis="y")

    stats_df = None
    if show_stats:
        stats_df, sig_lookup = _run_wilcoxon_tests(
            dmm_df,
            EXTRA_MARKERS_5B_TX,
            EXTRA_MARKERS_5B_PROT,
            alpha=alpha,
        )
        if sig_lookup:
            _annotate_significance(
                ax, dmm_df, marker_order, hue_order_full, sig_lookup, alpha
            )

    _configure_boxplot_axis(ax, xlabel, ylabel, ylim, title or "")
    _yabs = max(abs(v) for v in ax.get_ylim())
    ax.set_ylim(-_yabs, _yabs)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", which="both", length=0)

    handles, labels = ax.get_legend_handles_labels()
    nice = {"transcript": "transcript", "protein": "protein"}
    labels = [nice.get(l, l) for l in labels]
    if not show_legend:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    return handles, labels, stats_df


def render_figure5b_group(
    ax,
    group_name,
    group_markers,
    dmm_df,
    ref_df,
    *,
    ylim=None,
    show_stats=True,
    alpha=0.05,
    title=None,
    xlabel="Marker Augmentation",
    ylabel="\u0394RMSE",
    show_legend=False,
):
    """Render a single figure-5b gene-group panel onto *ax*.

    Args:
        group_name: Display name for this group.
        group_markers: Dict with keys ``"tx"`` and ``"prot"`` listing marker names.
        dmm_df, ref_df: As returned by ``_prepare_figure5b_data()``.

    Returns:
        tuple: ``(handles, labels, stats_df)``
    """
    markers_tx = group_markers.get("tx", ())
    markers_prot = group_markers.get("prot", ())
    group_marker_set = set(markers_tx) | set(markers_prot)

    group_df = dmm_df[dmm_df.marker.isin(group_marker_set)].copy()

    marker_order = sorted(group_df["marker"].unique())

    hue_order_full = ["transcript", "protein"]
    palette_full = {k: MODALITY_COLORS[k] for k in hue_order_full}

    sns.boxplot(
        data=group_df,
        x="marker",
        y="delta_rmse",
        hue="augmentation_type",
        hue_order=hue_order_full,
        order=marker_order,
        palette=palette_full,
        ax=ax,
    )
    ax.axhline(
        0,
        color="black",
        linestyle="-",
        linewidth=LW_THICK,
        alpha=1.0,
        zorder=0,
    )
    ax.grid(False, axis="y")

    stats_df = None
    if show_stats:
        stats_df, sig_lookup = _run_wilcoxon_tests(
            group_df,
            markers_tx,
            markers_prot,
            alpha=alpha,
        )
        if sig_lookup:
            _annotate_significance(
                ax, group_df, marker_order, hue_order_full, sig_lookup, alpha
            )

    _configure_boxplot_axis(
        ax,
        xlabel,
        ylabel,
        ylim,
        title or f"Feature Augmentations: {group_name}",
    )
    _yabs = max(abs(v) for v in ax.get_ylim())
    ax.set_ylim(-_yabs, _yabs)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", which="both", length=0)

    handles, labels = ax.get_legend_handles_labels()
    nice = {"transcript": "transcript", "protein": "protein"}
    labels = [nice.get(l, l) for l in labels]
    if not show_legend:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    return handles, labels, stats_df


# ---------------------------------------------------------------------------
# Convenience wrappers (standalone / presentation figures)
#
# Each wrapper creates its own figure with the padded 3-column layout
# (left-pad | plot | legend), calls the corresponding render_* function,
# places the legend in the right-hand column, and optionally saves.
# ---------------------------------------------------------------------------


def plot_model_comparison(
    df,
    figsize=(20, 6),
    ylim=(0.15, 1.0),
    refs=("DMM", "elasticnet"),
    ref_lines=("negative control", "positive control"),
    contexts=None,
    models=None,
    dataset="val",
    column_ratios=None,
    save_path=None,
    with_reference_uncertainty=False,
    show_stats=False,
    baseline_model=None,
    alpha=0.05,
):
    """Standalone model-comparison figure (presentation layout).

    For paper grids, use ``render_model_comparison`` directly.
    """
    df = _apply_display_labels(df)

    # Normalize dataset to a tuple
    if isinstance(dataset, str):
        datasets_to_plot = (dataset,)
    else:
        datasets_to_plot = tuple(dataset)

    # Prepare data for each dataset split
    data_by_dataset = {}
    for ds in datasets_to_plot:
        data_df, suffix = _split_val_test(df, ds)
        data_df.features.fillna("None", inplace=True)
        (
            plot_df,
            ref_df,
            hue_order,
            resolved_contexts,
            resolved_models,
        ) = _prepare_data(
            data_df,
            contexts,
            models,
            refs,
            ref_lines,
        )
        data_by_dataset[ds] = {"plot_df": plot_df, "ref_df": ref_df}
        if contexts is None:
            contexts = resolved_contexts
        if models is None:
            models = resolved_models

    model_labels = {m: MODEL_LABELS.get(m, m) for m in models}
    hue_order = [model_labels[m] for m in models if m in model_labels]
    if "elasticnet" in refs:
        hue_order.append(MODEL_LABELS.get("elasticnet"))

    # Create standalone layout
    fig, axes, ax_legend = _create_standalone_figure(
        figsize,
        column_ratios,
        n_plot_cols=len(datasets_to_plot),
    )

    # Print summary
    _print_summary(datasets_to_plot, data_by_dataset)

    padded_hue_order, _ = _add_group_padding(hue_order)

    # Render each split
    all_stats = []
    for i, ds in enumerate(datasets_to_plot):
        ax = axes[i]
        data = data_by_dataset[ds]
        _plot_boxplot(ax, data["plot_df"], contexts, hue_order)
        _add_reference_lines(
            ax,
            data["ref_df"],
            ref_lines,
            with_uncertainty=with_reference_uncertainty,
        )

        stats_df = None
        if show_stats and baseline_model is not None:
            baseline_label = MODEL_LABELS.get(baseline_model, baseline_model)
            stats_df, sig_lookup = _run_model_wilcoxon_tests(
                data["plot_df"],
                hue_order,
                baseline_label,
                contexts,
                alpha=alpha,
            )
            if sig_lookup:
                _annotate_model_significance(
                    ax,
                    data["plot_df"],
                    contexts,
                    padded_hue_order,
                    sig_lookup,
                    alpha,
                )
        all_stats.append(stats_df)

        _dataset_label = (
            " + ".join(dataset)
            if isinstance(dataset, (list, tuple))
            else dataset
        )

        ylabel = f"RMSE ({_dataset_label} set)" if i == 0 else ""
        _configure_boxplot_axis(
            ax, "Input Features (Baseline)", ylabel, ylim, ""
        )
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
        if i > 0:
            ax.set_yticklabels([])

    handles, labels = axes[0].get_legend_handles_labels()
    add_legend(ax_legend, handles, labels)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    # Combine stats from all splits
    combined_stats = (
        pd.concat([s for s in all_stats if s is not None], ignore_index=True)
        if any(s is not None for s in all_stats)
        else None
    )

    if show_stats:
        if len(axes) == 1:
            return fig, axes[0], combined_stats
        return fig, tuple(axes), combined_stats
    if len(axes) == 1:
        return fig, axes[0]
    return fig, tuple(axes)


def plot_figure6_features(
    figsize=(20, 6),
    ylim=(0.25, 0.75),
    ref_lines=("negative control", "positive control"),
    refs=("DMM + EGFR transcript levels", "linear regression"),
    column_ratios=None,
    save_path=None,
):
    """Standalone figure 6 (presentation layout).

    For paper grids, use ``render_figure6_features`` directly.
    """
    fig, axes, ax_legend = _create_standalone_figure(
        figsize,
        column_ratios or [0.5, 1, 0.5],
        n_plot_cols=1,
    )
    handles, labels = render_figure6_features(
        axes[0],
        ylim=ylim,
        ref_lines=ref_lines,
        refs=refs,
    )
    add_legend(ax_legend, handles, labels)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, axes[0]


def plot_figure3b_nhidden(
    figsize=(20, 6),
    ylim=(0.25, 0.75),
    ref_lines=("negative control", "positive control"),
    models=None,
    column_ratios=None,
    save_path=None,
):
    """Standalone figure 3b (presentation layout).

    For paper grids, use ``render_figure3b_nhidden`` directly.
    """
    fig, axes, ax_legend = _create_standalone_figure(
        figsize,
        column_ratios or [0.5, 1, 0.5],
        n_plot_cols=1,
    )
    handles, labels = render_figure3b_nhidden(
        axes[0],
        ylim=ylim,
        ref_lines=ref_lines,
        models=models,
    )
    add_legend(ax_legend, handles, labels)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, axes[0]


def plot_figure5b_augmentations(
    figsize=(20, 6),
    ylim=None,
    column_ratios=None,
    save_path=None,
    show_stats=True,
    alpha=0.05,
):
    """Standalone figure 5b – all markers (presentation layout).

    For paper grids, use ``render_figure5b_augmentations`` directly.
    """
    fig, axes, ax_legend = _create_standalone_figure(
        figsize,
        column_ratios or [0.1, 1, 0.25],
        n_plot_cols=1,
    )
    handles, labels, stats_df = render_figure5b_augmentations(
        axes[0],
        ylim=ylim,
        show_stats=show_stats,
        alpha=alpha,
    )
    add_legend(ax_legend, handles, labels)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, axes[0], stats_df


def plot_figure5b_by_group(
    figsize_per_group=(12, 6),
    ylim=None,
    save_path_template=None,
    show_stats=True,
    alpha=0.05,
):
    """Standalone figure 5b – one figure per gene group (presentation layout).

    For paper grids, use ``render_figure5b_group`` directly.
    """
    from training_configuration import EXTRA_MARKERS_5B_GROUPS

    dmm_df, ref_df = _prepare_figure5b_data()
    results = {}

    for group_name, group_markers in EXTRA_MARKERS_5B_GROUPS.items():
        markers_tx = group_markers.get("tx", ())
        markers_prot = group_markers.get("prot", ())
        group_marker_set = set(markers_tx) | set(markers_prot)
        _markers_with_data = sorted(
            {
                r["marker"]
                for _, r in dmm_df[
                    dmm_df.marker.isin(group_marker_set)
                ].iterrows()
            }
        )
        if not _markers_with_data:
            continue

        fig, axes, ax_legend = _create_standalone_figure(
            figsize_per_group,
            [0.1, 1, 0.25],
            n_plot_cols=1,
        )
        handles, labels, stats_df = render_figure5b_group(
            axes[0],
            group_name,
            group_markers,
            dmm_df,
            ref_df,
            ylim=ylim,
            show_stats=show_stats,
            alpha=alpha,
        )
        add_legend(ax_legend, handles, labels)

        if save_path_template is not None:
            safe_name = group_name.replace(" ", "_")
            save_path = save_path_template.format(group=safe_name)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        results[group_name] = (fig, axes[0], stats_df)

    return results
