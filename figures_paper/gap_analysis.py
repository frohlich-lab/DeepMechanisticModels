"""
Generalisation-gap analysis utilities.

Extracted from ``generalisation_gap.ipynb`` to reduce notebook bulk and enable
reuse.  All heavy plotting helpers, outlier-detection logic, gap / RMSE
computation, and annotation-palette machinery live here.

Usage (from a notebook whose cwd is *inside* the project)::

    from figures_paper.gap_analysis import (
        # data wrangling
        compute_gap_per_cell_line,
        compute_rmse_per_cell_line,
        detect_heatmap_outliers,
        # palettes & annotations
        build_annotation_specs,
        build_row_colors,
        PALETTES,
        # visualisation
        plot_heatmap,
        scatter_generalisation_gap,
        scatter_gap_vs_rmse,
        plot_gap_on_pca,
        plot_gap_on_pca_v2,
        correlate_gap_vs_markers,
    )
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests

# Ensure project root is on sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from cell_line_annotations import (
    get_subtype_annotations,
    load_marcotte_subtypes,
)
from cellosaurus_utils import get_cell_line_cellosaurus_annotations

# Lazy import: figures_paper.embeddings drags in heavy dependencies that need
# sys.path set up by the caller.  We import inside the functions that need them.
_embeddings_cache: dict = {}


def _get_embeddings_funcs():
    """Lazy-import ``get_highly_variable_genes`` and ``load_all_marker_data``."""
    if not _embeddings_cache:
        from figures_paper.embeddings import (
            get_highly_variable_genes,
            load_all_marker_data,
        )

        _embeddings_cache[
            "get_highly_variable_genes"
        ] = get_highly_variable_genes
        _embeddings_cache["load_all_marker_data"] = load_all_marker_data
    return _embeddings_cache


def load_all_marker_data(dtype: str) -> pd.DataFrame:
    """Convenience re-export of ``figures_paper.embeddings.load_all_marker_data``."""
    return _get_embeddings_funcs()["load_all_marker_data"](dtype)


# ═══════════════════════════════════════════════════════════════════════
#  Colour palettes
# ═══════════════════════════════════════════════════════════════════════

PALETTES = {
    "PAM50": {
        "Basal": "#e41a1c",
        "HER2": "#984ea3",
        "LA": "#377eb8",
        "LB": "#4daf4a",
        "Normal": "#ff7f00",
        "Other": "#999999",
    },
    "Lum/Bas": {"Basal": "#e41a1c", "Luminal": "#377eb8", "Normal": "#ff7f00"},
    "Receptor": {"ER": "#377eb8", "ERBB2": "#984ea3", "TNBC": "#e41a1c"},
    "Intrinsic": {
        "Basal": "#e41a1c",
        "CL": "#ff7f00",
        "HER2": "#984ea3",
        "LuminalA": "#377eb8",
        "LuminalB": "#4daf4a",
        "Normal": "#a6d854",
    },
    "Claudin": {"CL": "#ff7f00", "other": "#cccccc"},
    "Neve": {
        "basala": "#e41a1c",
        "basalb": "#fc8d62",
        "her2": "#984ea3",
        "luminal": "#377eb8",
    },
    "Site": {
        "Primary": "#1b9e77",
        "Metastatic": "#d95f02",
        "Unknown": "#999999",
    },
    "Mutation": {"mutated": "#e41a1c", "WT": "#cccccc"},
    "pERBB2": {"imputed": "#d95f02", "measured": "#1b9e77"},
}

# Aliases used in several notebooks
PAM50_PALETTE = PALETTES["PAM50"]
LB_PALETTE = PALETTES["Lum/Bas"]
RECEPTOR_PALETTE = PALETTES["Receptor"]
INTRINSIC_PALETTE = PALETTES["Intrinsic"]
CLAUDIN_PALETTE = PALETTES["Claudin"]
NEVE_PALETTE = PALETTES["Neve"]
SITE_PALETTE = PALETTES["Site"]
MUT_PALETTE = PALETTES["Mutation"]
ERBB2_PALETTE = PALETTES["pERBB2"]


# ═══════════════════════════════════════════════════════════════════════
#  Annotation machinery
# ═══════════════════════════════════════════════════════════════════════

# Cell lines whose pERBB2 measurement is imputed rather than measured
_MISSING_ERBB2_CELL_LINES = {
    "cBT483",
    "cHCC1187",
    "cHCC1419",
    "cHCC1569",
    "cHCC1806",
    "cHCC1937",
    "cHCC70",
    "cMCF12A",
    "cMDAkb2",
    "cMDAMB231",
    "cMDAMB361",
    "cMDAMB453",
    "cMDAMB468",
    "cT47D",
    "cUACC893",
}


def build_annotation_specs(
    cell_lines: list[str],
    *,
    file_dir: Path | str | None = None,
) -> dict[str, tuple[dict, dict]]:
    """Build ``{name: (lookup_dict, palette)}`` for all annotation tracks.

    Parameters
    ----------
    cell_lines : list[str]
        Cell-line identifiers (``cXXX`` format).
    file_dir : Path, optional
        Directory that contains cached cellosaurus annotation files.

    Returns
    -------
    dict
        ``ANNOTATION_SPECS``-style mapping consumed by :func:`build_row_colors`.
        Also returns ``cello_annot`` and ``cello_modality`` as attributes on the
        dict so callers can reuse them without re-fetching.
    """
    # -- PAM50 / luminal-basal subtypes --
    subtypes_pam50, subtypes_lb = get_subtype_annotations(cell_lines)

    # -- Marcotte subtypes --
    marcotte = load_marcotte_subtypes(cell_lines)
    subtypes_receptor = marcotte["subtype_three_receptor"].to_dict()
    subtypes_intrinsic = marcotte["subtype_intrinsic"].to_dict()
    subtypes_claudin = marcotte["subtype_claudin_low"].to_dict()
    subtypes_neve = marcotte["subtype_neve"].to_dict()

    # -- Cellosaurus annotations --
    cello_annot, cello_modality = get_cell_line_cellosaurus_annotations(
        file_dir=file_dir
    )

    # Site: Primary / Metastatic
    _site_raw = cello_annot["Site"].str.split(";").str[0].str.strip()
    subtypes_site = {
        cl: (
            "Primary"
            if s == "In situ"
            else "Metastatic"
            if s == "Metastatic"
            else "Unknown"
        )
        for cl, s in _site_raw.items()
    }

    # Key mutations (TP53, PIK3CA, CDH1, PTEN)
    _MUT_GENES = ["TP53", "PIK3CA", "CDH1", "PTEN"]
    _mut_lookups: dict[str, dict] = {}
    for gene in _MUT_GENES:
        col = f"{gene}_M"
        if col in cello_modality.columns:
            _mut_lookups[gene] = {
                cl: ("mutated" if v > 0 else "WT")
                for cl, v in cello_modality[col].items()
            }

    # pERBB2 imputation status
    subtypes_erbb2 = {
        cl: ("imputed" if cl in _MISSING_ERBB2_CELL_LINES else "measured")
        for cl in cell_lines
    }

    # -- Assemble --
    specs: dict[str, tuple[dict, dict]] = {
        "PAM50": (subtypes_pam50, PAM50_PALETTE),
        "Lum/Bas": (subtypes_lb, LB_PALETTE),
        "Receptor": (subtypes_receptor, RECEPTOR_PALETTE),
        "Intrinsic": (subtypes_intrinsic, INTRINSIC_PALETTE),
        "Claudin": (subtypes_claudin, CLAUDIN_PALETTE),
        "Neve": (subtypes_neve, NEVE_PALETTE),
        "Site": (subtypes_site, SITE_PALETTE),
        "pERBB2": (subtypes_erbb2, ERBB2_PALETTE),
    }
    for gene in _MUT_GENES:
        if gene in _mut_lookups:
            specs[gene] = (_mut_lookups[gene], MUT_PALETTE)

    # Stash cellosaurus objects as attributes for downstream callers
    specs["_cello_annot"] = cello_annot  # type: ignore[assignment]
    specs["_cello_modality"] = cello_modality  # type: ignore[assignment]
    return specs


def build_row_colors(
    cell_lines: list[str] | pd.Index,
    annotation_specs: dict[str, tuple[dict, dict]],
) -> pd.DataFrame:
    """Return a DataFrame of hex colours for ``sns.clustermap(row_colors=…)``."""
    rc = pd.DataFrame(index=cell_lines)
    for col_name, val in annotation_specs.items():
        if col_name.startswith("_"):
            continue  # skip internal stashed objects
        lookup, palette = val
        rc[col_name] = [
            palette.get(lookup.get(c, ""), "#dddddd") for c in cell_lines
        ]
    return rc


# ═══════════════════════════════════════════════════════════════════════
#  Gap / RMSE computation
# ═══════════════════════════════════════════════════════════════════════


def compute_gap_per_cell_line(
    by_df: pd.DataFrame,
    ref_filter: pd.Series,
    label: str,
) -> pd.DataFrame:
    """Per-cell-line generalisation gap for a given method.

    Returns DataFrame with columns: cell_line, model_label, context, gap,
    method, rmse_train, rmse_val.
    """
    sub = by_df[ref_filter].copy()
    agg = sub.groupby(
        ["cell_line", "model_label", "context", "dataset"], as_index=False
    ).agg(mean_rmse=("rmse", "mean"))
    pivot = agg.pivot_table(
        index=["cell_line", "model_label", "context"],
        columns="dataset",
        values="mean_rmse",
    ).reset_index()
    pivot = pivot.rename(columns={"train": "rmse_train", "val": "rmse_val"})
    pivot = pivot.dropna(subset=["rmse_train", "rmse_val"])
    pivot["gap"] = pivot["rmse_val"] - pivot["rmse_train"]
    pivot["method"] = label
    return pivot[
        [
            "cell_line",
            "model_label",
            "context",
            "gap",
            "method",
            "rmse_train",
            "rmse_val",
        ]
    ]


def compute_rmse_per_cell_line(
    all_df: pd.DataFrame,
    ref_filter: pd.Series,
    label: str,
) -> pd.DataFrame:
    """Per-cell-line mean RMSE from the all-split (all cell lines in training).

    Uses the ``'train'`` dataset since all cell lines are in the training set.
    """
    sub = all_df[ref_filter & (all_df["dataset"] == "train")].copy()
    agg = sub.groupby(
        ["cell_line", "model_label", "context"], as_index=False
    ).agg(mean_rmse=("rmse", "mean"))
    agg["method"] = label
    return agg


# ═══════════════════════════════════════════════════════════════════════
#  Outlier detection
# ═══════════════════════════════════════════════════════════════════════


def outliers_from_dendrogram(
    cg: sns.matrix.ClusterGrid,
    n_clusters: int | None = None,
) -> set[str]:
    """Extract outlier cell lines by cutting the row dendrogram.

    Cuts the row linkage tree into flat clusters and returns the cell lines
    belonging to the **smallest** cluster(s) — i.e. everything except the
    largest cluster, which is assumed to be the "normal" group.

    Parameters
    ----------
    cg : seaborn.matrix.ClusterGrid
        The object returned by :func:`plot_heatmap` (a ``sns.clustermap``).
    n_clusters : int or None
        Number of flat clusters to form.  When *None* (default), the cut is
        chosen automatically by finding the largest gap in the dendrogram
        merge distances, which typically separates a tight outlier branch
        from the main body of cell lines.

    Returns
    -------
    set[str]
        Cell line names in the outlier cluster(s).
    """
    from scipy.cluster.hierarchy import fcluster

    linkage = cg.dendrogram_row.linkage
    # fcluster returns assignments in the *original data* order
    original_labels = list(cg.data.index)

    if n_clusters is None:
        # Auto-detect: find the biggest gap between consecutive merge
        # distances in the linkage matrix (column 2 = merge distance).
        dists = linkage[:, 2]
        gaps = np.diff(dists)
        # Cut just above the biggest gap → keep merges below it
        cut_idx = int(np.argmax(gaps))
        # Distance threshold: midpoint between the two merges around the gap
        threshold = (dists[cut_idx] + dists[cut_idx + 1]) / 2.0
        labels = fcluster(linkage, t=threshold, criterion="distance")
    else:
        labels = fcluster(linkage, t=n_clusters, criterion="maxclust")

    # Find the largest cluster and call everything else "outlier"
    cluster_ids, counts = np.unique(labels, return_counts=True)
    main_cluster = cluster_ids[counts.argmax()]
    outlier_set = {
        cl for cl, cid in zip(original_labels, labels) if cid != main_cluster
    }
    return outlier_set


def detect_heatmap_outliers(
    df_in: pd.DataFrame,
    model_label: str,
    context: str,
    mode: str = "gap",
    ref: str = "DMM",
    iqr_factor: float = 1.5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """IQR-based outlier detection per (condition, observable) column.

    Returns
    -------
    outlier_flags : DataFrame[bool]
        ``(cell_line × column)``; ``True`` = outlier.
    outlier_summary : DataFrame
        Rows for every cell line flagged in ≥ 1 column, sorted by severity.
    mat : DataFrame
        The pivoted data matrix used for detection.
    """
    df = df_in[
        (df_in["model_label"] == model_label)
        & (df_in["context"] == context)
        & (df_in["ref"] == ref)
    ].copy()
    if mode == "rmse":
        df = df[df["dataset"] == "train"]
    df = df[np.isfinite(df["rmse"])]

    if mode == "gap":
        dfp = df.pivot_table(
            index="cell_line",
            columns=["condition", "observable", "dataset"],
            values="rmse",
        )
        df_val = dfp.xs("val", level="dataset", axis=1)
        df_train = dfp.xs("train", level="dataset", axis=1)
        df_train = df_train.reindex(columns=df_val.columns)
        mat = df_val - df_train
    else:
        mat = df.pivot_table(
            index="cell_line",
            columns=["condition", "observable"],
            values="rmse",
        )

    mat = mat.dropna(axis=0, how="all").dropna(axis=1, how="all").fillna(0.0)

    # IQR-based detection per column
    Q1 = mat.quantile(0.25)
    Q3 = mat.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - iqr_factor * IQR
    upper = Q3 + iqr_factor * IQR
    outlier_flags = (mat < lower) | (mat > upper)

    n_cols = mat.shape[1]
    records: list[dict] = []
    for cl in mat.index:
        ol_cols = outlier_flags.columns[outlier_flags.loc[cl]]
        if len(ol_cols) == 0:
            continue
        high = (mat.loc[cl] > upper).sum()
        low = (mat.loc[cl] < lower).sum()
        direction = (
            "mixed"
            if (high > 0 and low > 0)
            else ("high" if high > 0 else "low")
        )
        col_labels = [
            f"{c[0]}_{c[1]}" if isinstance(c, tuple) else str(c)
            for c in ol_cols
        ]
        records.append(
            {
                "cell_line": cl,
                "n_outlier_cols": len(ol_cols),
                "frac": len(ol_cols) / n_cols,
                "direction": direction,
                "outlier_columns": ", ".join(col_labels),
            }
        )

    if records:
        outlier_summary = (
            pd.DataFrame(records)
            .sort_values("n_outlier_cols", ascending=False)
            .reset_index(drop=True)
        )
    else:
        outlier_summary = pd.DataFrame(
            columns=[
                "cell_line",
                "n_outlier_cols",
                "frac",
                "direction",
                "outlier_columns",
            ]
        )
    return outlier_flags, outlier_summary, mat


# ═══════════════════════════════════════════════════════════════════════
#  Marker-gap correlations
# ═══════════════════════════════════════════════════════════════════════


def correlate_gap_vs_markers(
    gap: pd.Series,
    marker_data: pd.DataFrame,
    label: str,
    *,
    n_hvg: int | None = None,
    alpha: float = 0.05,
    groupby: dict | pd.Series | None = None,
) -> pd.DataFrame:
    """Spearman-correlate a per-cell-line gap vector with every marker column.

    Parameters
    ----------
    gap : Series indexed by cell_line
    marker_data : DataFrame indexed by cell_line, columns = markers
    label : data-type label (string)
    n_hvg : if given, restrict markers to top HVGs first
    alpha : FDR threshold
    groupby : optional marker → group mapping for group-wise BH correction

    Returns
    -------
    DataFrame sorted by |ρ| with *q*-values and significance flags.
    """
    common = sorted(set(gap.index) & set(marker_data.index))
    g = gap.loc[common]
    md = marker_data.loc[common].copy()

    if n_hvg is not None:
        get_highly_variable_genes = _get_embeddings_funcs()[
            "get_highly_variable_genes"
        ]
        hvg = get_highly_variable_genes(md, n_top=n_hvg)
        md = md[hvg]

    md = md.dropna(axis=1, how="any")
    markers = list(md.columns)

    rhos, pvals = [], []
    for col in markers:
        rho, p = spearmanr(g, md[col])
        rhos.append(rho)
        pvals.append(p)

    out = pd.DataFrame(
        {"marker": markers, "rho": rhos, "pval": pvals, "data_type": label}
    )
    out["abs_rho"] = out["rho"].abs()
    out["qval"] = np.nan

    valid = out["pval"].notna()
    if groupby is None:
        if valid.sum() > 0:
            _, qvals, _, _ = multipletests(
                out.loc[valid, "pval"], alpha=alpha, method="fdr_bh"
            )
            out.loc[valid, "qval"] = qvals
    else:
        gb = (
            groupby.to_dict()
            if isinstance(groupby, pd.Series)
            else dict(groupby)
        )
        out["_group"] = out["marker"].map(lambda m: gb.get(m, None))
        for _grp, grp_df in out.groupby("_group"):
            idx = grp_df.index[grp_df["pval"].notna()]
            if len(idx) > 0:
                _, qvals, _, _ = multipletests(
                    out.loc[idx, "pval"], alpha=alpha, method="fdr_bh"
                )
                out.loc[idx, "qval"] = qvals
        out = out.drop(columns=["_group"])

    out["significant"] = out["qval"] < alpha
    out = out.sort_values("abs_rho", ascending=False).reset_index(drop=True)
    return out


# ═══════════════════════════════════════════════════════════════════════
#  Plotting helpers
# ═══════════════════════════════════════════════════════════════════════


def _add_subtype_legends(
    cg,
    annotation_specs: dict[str, tuple[dict, dict]],
) -> None:
    """Add colour legends for all annotation tracks to a seaborn clustermap.

    Mutation tracks that share :data:`MUT_PALETTE` are merged into one legend.
    """
    mut_names = [
        n
        for n, val in annotation_specs.items()
        if not n.startswith("_") and val[1] is MUT_PALETTE
    ]
    other_names = [
        n
        for n in annotation_specs
        if not n.startswith("_") and n not in mut_names
    ]

    n_blocks = len(other_names) + (1 if mut_names else 0)
    spacing = min(0.16, 0.9 / max(n_blocks, 1))
    y_offset = 1.0

    legends = []
    for name in other_names:
        _, palette = annotation_specs[name]
        handles = [Patch(facecolor=c, label=s) for s, c in palette.items()]
        leg = cg.ax_heatmap.legend(
            handles=handles,
            title=name,
            loc="upper left",
            bbox_to_anchor=(1.12, y_offset),
            fontsize=5,
            title_fontsize=6,
            frameon=True,
            handlelength=0.8,
            handleheight=0.6,
            borderpad=0.3,
            labelspacing=0.2,
        )
        legends.append(leg)
        cg.ax_heatmap.add_artist(leg)
        y_offset -= spacing

    if mut_names:
        handles = [
            Patch(facecolor=MUT_PALETTE["mutated"], label="mutated"),
            Patch(facecolor=MUT_PALETTE["WT"], label="WT"),
        ]
        title = "Mutations\n(" + ", ".join(mut_names) + ")"
        leg = cg.ax_heatmap.legend(
            handles=handles,
            title=title,
            loc="upper left",
            bbox_to_anchor=(1.12, y_offset),
            fontsize=5,
            title_fontsize=6,
            frameon=True,
            handlelength=0.8,
            handleheight=0.6,
            borderpad=0.3,
            labelspacing=0.2,
        )
        legends.append(leg)
        cg.ax_heatmap.add_artist(leg)


def plot_heatmap(
    df_in: pd.DataFrame,
    model_label: str,
    context: str,
    annotation_specs: dict[str, tuple[dict, dict]],
    *,
    mode: str = "gap",
    ref: str = "DMM",
    gap_threshold: float = 0.0,
    metric: str = "euclidean",
    method: str = "average",
    figsize: tuple[int, int] = (14, 16),
):
    """Unified clustermap for per-cell-line × (condition, observable) data.

    ``mode='gap'``  → generalisation gap (val − train RMSE), diverging.
    ``mode='rmse'`` → absolute RMSE from one split, sequential.
    """
    df = df_in[
        (df_in["model_label"] == model_label)
        & (df_in["context"] == context)
        & (df_in["ref"] == ref)
    ].copy()
    if mode == "rmse":
        df = df[df["dataset"] == "train"]
    df = df[np.isfinite(df["rmse"])]
    if df.empty:
        raise ValueError(
            f"No rows for model_label={model_label!r}, context={context!r}"
            + (f", ref={ref!r}" if mode == "rmse" else "")
        )

    # Pivot
    if mode == "gap":
        dfp = df.pivot_table(
            index="cell_line",
            columns=["condition", "observable", "dataset"],
            values="rmse",
        )
        df_val = dfp.xs("val", level="dataset", axis=1)
        df_train = dfp.xs("train", level="dataset", axis=1)
        df_train = df_train.reindex(columns=df_val.columns)
        mat = df_val - df_train
    else:
        mat = df.pivot_table(
            index="cell_line",
            columns=["condition", "observable"],
            values="rmse",
        )

    if mode == "gap" and gap_threshold > 0.0:
        mat = mat.loc[mat.abs().max(axis=1) > gap_threshold]
    mat = mat.dropna(axis=0, how="all").dropna(axis=1, how="all").fillna(0.0)
    if mat.empty:
        raise ValueError("No data left after filtering")

    # Colour scale
    if mode == "gap":
        max_abs = np.nanmax(np.abs(mat.values))
        if not np.isfinite(max_abs) or max_abs == 0:
            max_abs = 1.0
        cmap_kw = {
            "cmap": "coolwarm",
            "center": 0.0,
            "vmin": -max_abs,
            "vmax": max_abs,
        }
    else:
        vmax = np.nanmax(mat.values)
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1.0
        cmap_kw = {"cmap": "YlOrRd", "vmin": 0, "vmax": vmax}

    rc = build_row_colors(mat.index, annotation_specs)

    cg = sns.clustermap(
        mat,
        metric=metric,
        method=method,
        **cmap_kw,
        row_colors=rc,
        row_cluster=True,
        col_cluster=True,
        figsize=figsize,
        yticklabels=True,
        xticklabels=1,
    )

    if mode == "gap":
        title = f"{model_label} – {context}\nGeneralisation gap (val RMSE − train RMSE)"
    else:
        title = f"{model_label} – {context} ({ref})\nRMSE per cell line (all split)"
    cg.ax_heatmap.set_title(title)
    cg.ax_heatmap.set_xlabel("condition / observable")
    cg.ax_heatmap.set_ylabel("cell line")

    _add_subtype_legends(cg, annotation_specs)
    plt.tight_layout()
    return cg


def scatter_generalisation_gap(
    merged: pd.DataFrame,
    context: str | None = None,
    figsize: tuple[int, int] = (7, 7),
):
    """Scatter: DMM gap (x) vs Regression gap (y) per cell line."""
    plot_df = merged.copy()
    title_suffix = ""
    if context is not None:
        plot_df = plot_df[plot_df["context"] == context]
        title_suffix = f" ({context})"

    fig, ax = plt.subplots(figsize=figsize)

    if "context" in plot_df.columns and plot_df["context"].nunique() > 1:
        for ctx, grp in plot_df.groupby("context"):
            ax.scatter(
                grp["gap_dmm"],
                grp["gap_regression"],
                label=ctx,
                s=50,
                alpha=0.8,
            )
        ax.legend(title="Context")
    else:
        ax.scatter(
            plot_df["gap_dmm"], plot_df["gap_regression"], s=50, alpha=0.8
        )

    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, "k--", alpha=0.4, linewidth=1, zorder=0)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    tick_min = np.floor(lims[0] * 10) / 10
    tick_max = np.ceil(lims[1] * 10) / 10
    ticks = np.arange(tick_min, tick_max + 0.05, 0.1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("DMM generalisation gap (RMSE)")
    ax.set_ylabel("Regression generalisation gap (RMSE)")
    ax.set_title(f"Generalisation gap per cell line{title_suffix}")

    for _, row in plot_df.iterrows():
        ax.annotate(
            row["cell_line"].replace("c", "", 1),
            (row["gap_dmm"], row["gap_regression"]),
            fontsize=6,
            alpha=0.7,
            textcoords="offset points",
            xytext=(4, 4),
        )

    plt.tight_layout()
    return fig, ax


def scatter_gap_vs_rmse(
    gap_vs_rmse: pd.DataFrame,
    method: str = "dmm",
    context: str | None = None,
    figsize: tuple[int, int] = (7, 7),
):
    """Scatter: RMSE (all split) vs generalisation gap (LOOCV)."""
    if method == "dmm":
        x_col, y_col = "rmse_all_dmm", "gap_dmm"
        x_label = "DMM RMSE (all split)"
        y_label = "DMM generalisation gap (LOOCV)"
        title_method = "DMM"
    else:
        x_col, y_col = "rmse_all_reg", "gap_regression"
        x_label = "Regression RMSE (all split)"
        y_label = "Regression generalisation gap (LOOCV)"
        title_method = "Regression"

    plot_df = gap_vs_rmse.copy()
    title_suffix = ""
    if context is not None:
        plot_df = plot_df[plot_df["context"] == context]
        title_suffix = f" ({context})"

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(plot_df[x_col], plot_df[y_col], s=50, alpha=0.8)
    ax.axhline(0, color="k", linestyle="--", alpha=0.3, linewidth=1)

    for _, row in plot_df.iterrows():
        ax.annotate(
            row["cell_line"].replace("c", "", 1),
            (row[x_col], row[y_col]),
            fontsize=6,
            alpha=0.7,
            textcoords="offset points",
            xytext=(4, 4),
        )

    r, p = pearsonr(plot_df[x_col], plot_df[y_col])
    rho, p_s = spearmanr(plot_df[x_col], plot_df[y_col])
    ax.text(
        0.05,
        0.95,
        f"Pearson r = {r:.3f} (p = {p:.2e})\nSpearman ρ = {rho:.3f} (p = {p_s:.2e})",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "wheat", "alpha": 0.5},
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{title_method}: gap vs RMSE{title_suffix}")
    plt.tight_layout()
    return fig, ax


def plot_gap_on_pca(
    pca_df: pd.DataFrame,
    gap_series: pd.Series,
    outlier_set: set[str],
    title: str,
    ax,
    *,
    ve1: float | None = None,
    ve2: float | None = None,
    cmap: str = "RdBu_r",
    center: bool = True,
    label_outliers: bool = True,
):
    """Scatter cell lines in PCA space, coloured by gap value.

    Parameters
    ----------
    ve1, ve2 : float, optional
        Variance explained for PC1/PC2 used in axis labels.  When ``None``
        the labels show only "PC1" / "PC2".
    """
    from adjustText import adjust_text as _adjust

    common = sorted(set(pca_df.index) & set(gap_series.dropna().index))
    df = pca_df.loc[common].copy()
    df["gap"] = gap_series.loc[common]
    df["is_outlier"] = df.index.isin(outlier_set)
    df["cl_short"] = [c.replace("c", "", 1) for c in df.index]

    if center:
        vmax = max(abs(df["gap"].min()), abs(df["gap"].max()))
        vmin = -vmax
    else:
        vmin, vmax = df["gap"].min(), df["gap"].max()

    inl = df[~df["is_outlier"]]
    sc = ax.scatter(
        inl["PC1"],
        inl["PC2"],
        c=inl["gap"],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        s=50,
        edgecolor="grey",
        linewidth=0.5,
        alpha=0.8,
        zorder=2,
    )
    outl = df[df["is_outlier"]]
    ax.scatter(
        outl["PC1"],
        outl["PC2"],
        c=outl["gap"],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        s=90,
        marker="D",
        edgecolor="black",
        linewidth=1.0,
        alpha=0.95,
        zorder=3,
    )

    if label_outliers and len(outl) > 0:
        texts = []
        for _, r in outl.iterrows():
            texts.append(
                ax.text(
                    r["PC1"],
                    r["PC2"],
                    r["cl_short"],
                    fontsize=6,
                    alpha=0.85,
                    weight="bold",
                )
            )
        _adjust(
            texts,
            ax=ax,
            arrowprops={
                "arrowstyle": "-",
                "color": "grey",
                "alpha": 0.4,
                "lw": 0.5,
            },
            expand=(1.4, 1.6),
            force_text=(0.4, 0.6),
        )

    ax.axhline(0, color="grey", alpha=0.3, linewidth=0.5, zorder=0)
    ax.axvline(0, color="grey", alpha=0.3, linewidth=0.5, zorder=0)

    pc1_lbl = f"PC1 ({ve1:.0%})" if ve1 is not None else "PC1"
    pc2_lbl = f"PC2 ({ve2:.0%})" if ve2 is not None else "PC2"
    ax.set_xlabel(pc1_lbl)
    ax.set_ylabel(pc2_lbl)
    ax.set_title(title, fontsize=10)
    return sc


# Alias kept for backward compatibility (v2 signature just adds explicit VE args)
plot_gap_on_pca_v2 = plot_gap_on_pca


# ═══════════════════════════════════════════════════════════════════════
#  Per-observable marker–gap correlation heatmap
# ═══════════════════════════════════════════════════════════════════════


def marker_obs_correlation_heatmap(
    pivot_obs: pd.DataFrame,
    data_type: str,
    *,
    n_hvg: int | None = None,
    n_show: int = 10,
    show_all_markers: bool = False,
    sort_by_abs_rho: bool = False,
    vmin: float = -0.6,
    vmax: float = 0.6,
    save_path: str | Path | None = None,
    figsize_width: float = 7,
    row_height: float = 0.35,
    min_fig_height: float = 6.0,
    title: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, "plt.Figure"]:
    """Compute per-observable Spearman correlations and plot a ρ heatmap.

    This consolidates the repeated proteomics / transcriptomics / CyTOF
    heatmap blocks that were duplicated in the notebook.

    Parameters
    ----------
    pivot_obs : DataFrame
        Must have columns ``observable``, ``cell_line``, ``gap``.
    data_type : ``"proteomics"`` | ``"transcriptomics"`` | ``"cytof"``
    n_hvg : int, optional
        Passed to :func:`correlate_gap_vs_markers`.
    n_show : int
        Number of top markers to keep *per observable* (for union).
        Ignored when *show_all_markers* is True.
    show_all_markers : bool
        If True, show every marker (useful for small panels like CyTOF).
    sort_by_abs_rho : bool
        If True, sort rows by max |ρ| across observables.
    vmin, vmax : float
        Colour-scale limits.
    save_path : str or Path, optional
        If given, save the figure to this path.
    figsize_width : float
        Width of the figure in inches.
    row_height : float
        Height per marker row (used to compute total fig height).
    min_fig_height : float
        Minimum figure height in inches.
    title : str, optional
        Custom title.  Defaults to an auto-generated one.

    Returns
    -------
    rho_matrix : DataFrame  (markers × observables, display-labelled columns)
    qval_matrix : DataFrame (same shape)
    fig : matplotlib.figure.Figure
    """
    observables = sorted(pivot_obs["observable"].unique())
    obs_labels = {
        o: o.replace("_obs", "").replace("_", " ") for o in observables
    }
    marker_data = load_all_marker_data(data_type)

    # --- correlate per observable ---
    per_obs: dict[str, pd.DataFrame] = {}
    for obs in observables:
        gap_obs = pivot_obs[pivot_obs["observable"] == obs].set_index(
            "cell_line"
        )["gap"]
        per_obs[obs] = correlate_gap_vs_markers(
            gap_obs,
            marker_data,
            data_type,
            n_hvg=n_hvg,
        ).set_index("marker")

    # --- select markers to display ---
    if show_all_markers:
        selected = sorted(marker_data.columns)
    else:
        selected: list[str] = []
        for obs in observables:
            selected.extend(per_obs[obs].head(n_show).index.tolist())
        selected = list(dict.fromkeys(selected))  # deduplicate, preserve order

    # --- build ρ / q matrices ---
    rho_mat = pd.DataFrame(index=selected, columns=observables, dtype=float)
    qval_mat = pd.DataFrame(index=selected, columns=observables, dtype=float)
    for obs in observables:
        for m in selected:
            if m in per_obs[obs].index:
                rho_mat.loc[m, obs] = per_obs[obs].loc[m, "rho"]
                qval_mat.loc[m, obs] = per_obs[obs].loc[m, "qval"]

    # Optional: sort rows by max |ρ|
    if sort_by_abs_rho:
        rho_mat = rho_mat.astype(float)
        order = rho_mat.abs().max(axis=1).sort_values(ascending=False).index
        rho_mat = rho_mat.loc[order]
        qval_mat = qval_mat.loc[order]

    rho_display = rho_mat.rename(columns=obs_labels).astype(float)
    qval_display = qval_mat.rename(columns=obs_labels).astype(float)

    # --- significance stars ---
    annot = rho_display.round(2).astype(str)
    for i in range(annot.shape[0]):
        for j in range(annot.shape[1]):
            q = qval_display.iloc[i, j]
            if q < 0.001:
                annot.iloc[i, j] += "***"
            elif q < 0.01:
                annot.iloc[i, j] += "**"
            elif q < 0.05:
                annot.iloc[i, j] += "*"

    # --- plot ---
    fig_h = max(min_fig_height, len(selected) * row_height)
    fig, ax = plt.subplots(figsize=(figsize_width, fig_h))
    sns.heatmap(
        rho_display,
        annot=annot,
        fmt="",
        cmap="RdBu_r",
        center=0,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Spearman ρ", "shrink": 0.7},
        ax=ax,
    )
    if title is None:
        hvg_note = f", {n_hvg} HVGs" if n_hvg else ""
        mode_note = (
            "all markers"
            if show_all_markers
            else f"top {n_show} per observable"
        )
        title = f"Per-observable gap vs {data_type} markers\n({mode_note}{hvg_note})"
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("")
    ax.set_xlabel("")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(str(save_path), bbox_inches="tight", dpi=150)

    return rho_display, qval_display, fig
