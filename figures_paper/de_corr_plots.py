"""Shared volcano-plot and correlation-barplot helpers for paper figures.

Two public functions:

* ``plot_panel_volcano`` – volcano plot of DE / fold-change results
* ``draw_corr_barplot``  – horizontal Pearson / Spearman correlation barplot

Both are axis-level helpers designed for embedding into larger multi-panel
figure layouts.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from figure_config import (
    FONTSIZE,
    FONTSIZE_HEADING,
    LW_THIN,
    configure_axis_spines,
)

_adjust_text = None
try:
    from adjustText import adjust_text as _adjust_text

    _HAS_ADJUST_TEXT = True
except ImportError:
    _HAS_ADJUST_TEXT = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CORE_MAPK_GENES = {
    "EGFR",
    "ERBB2",
    "ERBB3",
    "ERBB4",
    "GRB2",
    "SOS1",
    "SOS2",
    "SHC1",
    "GAB1",
    "SPRY2",
    "SPRY4",
    "HRAS",
    "KRAS",
    "NRAS",
    "BRAF",
    "RAF1",
    "ARAF",
    "MAP2K1",
    "MAP2K2",
    "MAPK1",
    "MAPK3",
    "RPS6KA1",
    "RPS6KA2",
    "RPS6KA3",
    "NF1",
    "RASA1",
    "DUSP4",
    "DUSP6",
}

# ---------------------------------------------------------------------------
# Volcano plot
# ---------------------------------------------------------------------------


def plot_panel_volcano(
    ax,
    de_df,
    *,
    fc_col="logFC",
    q_col="q_value",
    gene_col="gene",
    alpha=0.05,
    title="",
    show_legend=False,
    show_annotations=True,
    color="#377eb8",
    gene_groups=None,
    adjust_text_kwargs=None,
):
    """Volcano plot of DE results (fold change vs -log10 q/p-value).

    Parameters
    ----------
    ax : matplotlib Axes
    de_df : DataFrame containing at least *fc_col*, *q_col*, *gene_col*
    fc_col : column name for log₂ fold change (default ``"logFC"``)
    q_col : column name for adjusted p-value / q-value (default ``"q_value"``)
    gene_col : column name for gene / feature identifier (default ``"gene"``)
    alpha : significance threshold (default 0.05)
    title : str
    show_legend : bool
    show_annotations : bool – label significant MAPK genes
    color : str – colour for significant points
    gene_groups : dict or None
        ``{process_name: ([gene1, gene2, ...], color_str)}``.  When provided,
        only genes in these groups that are also significant are labelled, each
        coloured by their process.  First match wins for genes in multiple
        groups.  Overrides the default MAPK-centric label selection.
    adjust_text_kwargs : dict or None
        Extra keyword arguments forwarded to ``adjustText.adjust_text``.
        Merged on top of the defaults; useful for tuning ``expand``,
        ``force_text``, etc.
    """
    df = de_df.copy()
    df["nlog10q"] = -np.log10(df[q_col].clip(lower=1e-300))
    sig_mask = df[q_col] < alpha

    # Non-significant points (grey)
    _ns = df[~sig_mask]
    ax.scatter(
        _ns[fc_col],
        _ns["nlog10q"],
        s=4,
        alpha=0.3,
        color="#cccccc",
        edgecolors="none",
        rasterized=True,
    )

    # Significant points — colour by process group if provided
    _sig = df[sig_mask]
    if gene_groups is not None:
        _gene_to_proc_color = {}
        for _, (_genes, _pcol) in gene_groups.items():
            for _g in _genes:
                _gene_to_proc_color.setdefault(_g, _pcol)
        _sig_group = _sig[_sig[gene_col].isin(_gene_to_proc_color)]
        _sig_other = _sig[~_sig[gene_col].isin(_gene_to_proc_color)]
        ax.scatter(
            _sig_other[fc_col],
            _sig_other["nlog10q"],
            s=8,
            alpha=0.5,
            color=color,
            edgecolors="none",
            rasterized=True,
        )
        for _g, _grp in _sig_group.groupby(
            _sig_group[gene_col].map(_gene_to_proc_color)
        ):
            ax.scatter(
                _grp[fc_col],
                _grp["nlog10q"],
                s=12,
                alpha=0.9,
                color=_g,
                edgecolors="none",
                rasterized=True,
                zorder=3,
            )
    else:
        ax.scatter(
            _sig[fc_col],
            _sig["nlog10q"],
            s=8,
            alpha=0.7,
            color=color,
            edgecolors="none",
            rasterized=True,
        )

    # Threshold line
    _thresh_y = -np.log10(alpha)
    ax.axhline(_thresh_y, color="grey", ls="--", lw=LW_THIN, zorder=0)

    # Annotate significant genes
    if show_annotations and len(_sig) > 0:
        _mapk_set = set(_CORE_MAPK_GENES)
        _sig_sorted = _sig.sort_values(q_col)

        if gene_groups is not None:
            _gene_to_proc_color = {}
            for _, (_genes, _pcol) in gene_groups.items():
                for _g in _genes:
                    _gene_to_proc_color.setdefault(_g, _pcol)
            _to_label = _sig_sorted[
                _sig_sorted[gene_col].isin(_gene_to_proc_color)
            ]
        else:
            _mapk_hits = _sig_sorted[_sig_sorted[gene_col].isin(_mapk_set)]
            _other_hits = _sig_sorted[~_sig_sorted[gene_col].isin(_mapk_set)]
            _to_label = pd.concat(
                [
                    _mapk_hits.head(5),
                    _other_hits.head(max(0, 8 - len(_mapk_hits.head(5)))),
                ]
            )

        _texts = []
        for _, row in _to_label.iterrows():
            if gene_groups is not None:
                _gene_color = _gene_to_proc_color.get(row[gene_col], color)
            elif row[gene_col] in _CORE_MAPK_GENES:
                _gene_color = "red"
            elif row[gene_col] in _mapk_set:
                _gene_color = "darkgreen"
            else:
                _gene_color = color
            _texts.append(
                ax.text(
                    row[fc_col],
                    row["nlog10q"],
                    row[gene_col],
                    fontsize=FONTSIZE,
                    color=_gene_color,
                    fontweight="bold",
                )
            )
        if _texts and _HAS_ADJUST_TEXT:
            _at_kw = {
                "arrowprops": {"arrowstyle": "-", "color": "grey", "lw": 0.4},
                "expand": (2.5, 2.5),
            }
            if adjust_text_kwargs:
                _at_kw.update(adjust_text_kwargs)
            try:
                _adjust_text(_texts, ax=ax, **_at_kw)
            except TypeError:
                # older adjustText (<1.0) uses expand_text / expand_points
                _at_kw_legacy = {
                    k: v for k, v in _at_kw.items() if k != "expand"
                }
                _at_kw_legacy["expand_text"] = _at_kw.get("expand", (2.5, 2.5))
                _at_kw_legacy["expand_points"] = _at_kw.get(
                    "expand", (2.5, 2.5)
                )
                _adjust_text(_texts, ax=ax, **_at_kw_legacy)

    ax.set_xlabel("log$_2$ FC", fontsize=FONTSIZE)
    ax.set_ylabel("$-\\log_{10}(q)$", fontsize=FONTSIZE)
    if title:
        ax.set_title(title, fontsize=FONTSIZE_HEADING)
    configure_axis_spines(ax)

    if show_legend:
        _leg = [
            Line2D(
                [],
                [],
                marker="o",
                ls="",
                markersize=3,
                color=color,
                label=f"FDR < {alpha}",
            ),
            Line2D(
                [],
                [],
                marker="o",
                ls="",
                markersize=3,
                color="#cccccc",
                label="n.s.",
            ),
        ]
        ax.legend(
            handles=_leg,
            fontsize=FONTSIZE - 1,
            frameon=False,
            loc="upper right",
        )


# ---------------------------------------------------------------------------
# Correlation barplot
# ---------------------------------------------------------------------------


def draw_corr_barplot(
    ax,
    df,
    *,
    filter_by,
    effect_col="pearson_r",
    rank_col=None,
    sig_col="pearson_q_target",
    feature_col="feature",
    n_top=5,
    alpha=0.05,
    color_pos="#377eb8",
    color_neg="#aec7e8",
    show_yticklabels=True,
    xlabel=None,
    ylabel=None,
    reference_genes=None,
):
    """Horizontal correlation barplot for one data slice.

    Bar labels (feature name + r value) are drawn inside each bar using
    data coordinates.  Bars are coloured by significance and direction:
    firebrick / steelblue (significant), lightcoral / lightsteelblue (n.s.).

    Parameters
    ----------
    ax : matplotlib Axes
    df : DataFrame
    filter_by : dict ``{col: val, ...}`` — rows to select from *df*
    effect_col : column for the effect size (Pearson r or Spearman ρ);
        default ``"pearson_r"``
    rank_col : column for sorting by absolute effect; defaults to
        ``"abs_" + effect_col``
    sig_col : column with (adjusted) p-values used for coloring;
        default ``"pearson_q_target"``
    feature_col : column with feature / gene names for bar labels;
        default ``"feature"``
    n_top : number of top features to show (by absolute effect)
    alpha : significance threshold for coloring
    color_pos : accent colour for significant positive-effect labels
    color_neg : kept for API compatibility (not used for label colouring)
    show_yticklabels : kept for API compatibility (labels are drawn inside bars)
    xlabel : optional x-axis label
    ylabel : optional y-axis label
    reference_genes : list of str or None
        Gene / feature names for which a labeled vertical dashed reference line
        is drawn at their correlation value.  Genes that are absent from the
        (filtered) data are silently skipped.  Lines are drawn even when the
        gene is not in the top-*n_top* bars.
    """
    sub = df.copy()
    for col, val in filter_by.items():
        sub = sub[sub[col] == val]
    if sub.empty:
        ax.set_visible(False)
        return

    _rank_col = rank_col if rank_col is not None else "abs_" + effect_col
    top = sub.nlargest(n_top, _rank_col).sort_values(
        effect_col, ascending=True
    )

    bar_colors = []
    sig_flags = []
    for _, row in top.iterrows():
        is_sig = row[sig_col] < alpha
        sig_flags.append(is_sig)
        r = row[effect_col]
        if is_sig:
            bar_colors.append("firebrick" if r > 0 else "steelblue")
        else:
            bar_colors.append("lightcoral" if r > 0 else "lightsteelblue")

    y_pos = np.arange(len(top))
    ax.barh(y_pos, top[effect_col], color=bar_colors, edgecolor="none")
    ax.set_yticks([])
    ax.set_yticklabels([])

    for i, (feat, r, is_sig) in enumerate(
        zip(top[feature_col], top[effect_col], sig_flags, strict=False)
    ):
        label_color = color_pos if is_sig else "#555555"
        if r < 0:
            ax.text(
                0.02,
                i,
                feat,
                ha="left",
                va="center",
                fontsize=FONTSIZE,
                color=label_color,
            )
            ax.text(
                -0.02,
                i,
                "-" + (f"{r:.2f}"[2:]),
                ha="right",
                va="center",
                fontsize=FONTSIZE,
                color="white",
            )
        else:
            ax.text(
                -0.02,
                i,
                feat,
                ha="right",
                va="center",
                fontsize=FONTSIZE,
                color=label_color,
            )
            ax.text(
                0.02,
                i,
                f"{r:.2f}"[1:],
                ha="left",
                va="center",
                fontsize=FONTSIZE,
                color="white",
            )

    ax.set_xlim(-1, 1)
    ax.axvline(0, color="black", linewidth=0.5)

    # Reference-gene vertical lines
    if reference_genes:
        _xform = ax.get_xaxis_transform()  # x: data coords, y: axes 0–1
        for _gene in reference_genes:
            _hit = sub[sub[feature_col] == _gene]
            if _hit.empty:
                continue
            _corr_val = float(_hit.iloc[0][effect_col])
            ax.axvline(
                _corr_val,
                ls="--",
                lw=LW_THIN,
                color=color_pos,
                alpha=0.8,
                zorder=3,
            )
            ax.text(
                _corr_val,
                0.99,
                _gene,
                transform=_xform,
                rotation=90,
                ha="center",
                va="top",
                fontsize=FONTSIZE,
                color=color_pos,
            )

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FONTSIZE)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FONTSIZE)
    sns.despine(ax=ax, left=True)
    ax.grid(False, axis="both")
    ax.tick_params(axis="x", labelsize=FONTSIZE)
