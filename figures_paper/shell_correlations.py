"""
Shell-restricted Spearman correlation utilities.

Provides a single function, `shell_restricted_correlations`, that correlates
a per-cell-line metric (or dict of per-job metrics) with markers restricted to
the gene neighbourhood shells produced by `gene_shells`.

Used by:
  - generalisation_gap.ipynb  (per-job dict → median aggregation)
  - embeddings_parameters.ipynb (single parameter vector)
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

from figures_paper.embeddings import get_highly_variable_genes

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _hvg_filter(
    marker_data: pd.DataFrame, seeds_set: set, n_hvg: int | None
) -> pd.DataFrame:
    """Pre-filter *marker_data* to top-*n_hvg* HVGs, always keeping seeds."""
    if n_hvg is None or len(marker_data.columns) <= n_hvg:
        return marker_data
    hvg = set(get_highly_variable_genes(marker_data, n_top=n_hvg))
    keep = hvg | seeds_set
    return marker_data[[c for c in marker_data.columns if c in keep]]


def _drop_high_missing(
    marker_data: pd.DataFrame, max_missing_frac: float | None
) -> pd.DataFrame:
    """Drop columns whose missing fraction exceeds *max_missing_frac*."""
    if max_missing_frac is None:
        return marker_data
    frac = marker_data.isna().mean()
    keep = frac[frac <= max_missing_frac].index
    return marker_data[keep]


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------


def shell_restricted_correlations(
    metric,  # Series OR dict[job_id → Series] indexed by cell_line
    marker_data,  # DataFrame indexed by cell_line, columns = markers
    shell_genes,  # set of gene names to test (e.g. seeds ∪ shell1)
    seeds_set,  # set of seed gene names
    data_type_label: str,
    n_hvg: int | None = None,
    max_missing_frac: float | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Shell-restricted Spearman correlations.

    Parameters
    ----------
    metric : pd.Series or dict[str, pd.Series]
        If a *Series*, correlate directly with each marker column.
        If a *dict* of job_id → Series, correlate per job and report median
        ρ / p across jobs (with rho_std, pval_std, n_jobs columns).
    marker_data : pd.DataFrame
        Rows = cell lines, columns = markers.
    shell_genes : set[str]
        Marker columns to test (typically ``seeds ∪ shell1``).
    seeds_set : set[str]
        Seed genes, used for HVG keep-list and for the ``shell`` label.
    data_type_label : str
        Value written into the ``data_type`` column of the result.
    n_hvg : int or None
        If set, pre-filter ``marker_data`` to the top-*n_hvg* HVGs
        (seeds are always kept).
    max_missing_frac : float or None
        If set, drop columns whose missing fraction exceeds this threshold
        *before* any other filtering.
    alpha : float
        FDR significance threshold (BH method).

    Returns
    -------
    pd.DataFrame
        One row per tested marker, sorted by |ρ|.  Columns always include
        ``marker, rho, pval, data_type, n_tested, abs_rho, qval,
        significant, shell``.  When *metric* is a dict, additional columns
        ``rho_std, pval_std, n_jobs`` are present.
    """
    per_job = isinstance(metric, dict)

    # Pre-filter: missingness, HVG, restrict to shell
    md = _drop_high_missing(marker_data, max_missing_frac)
    md = _hvg_filter(md, seeds_set, n_hvg)
    valid_cols = [c for c in md.columns if c in shell_genes]
    if not valid_cols:
        return pd.DataFrame()
    md = md[valid_cols].copy()

    # ----- per-job branch ------------------------------------------------
    if per_job:
        job_rhos = {col: [] for col in md.columns}
        job_pvals = {col: [] for col in md.columns}

        for _job_id, metric_vec in metric.items():
            common = sorted(set(metric_vec.index) & set(md.index))
            if len(common) < 5:
                continue
            g = metric_vec.loc[common]
            md_job = md.loc[common].dropna(axis=1, how="any")
            for col in md_job.columns:
                rho, p = spearmanr(g, md_job[col])
                job_rhos[col].append(rho)
                job_pvals[col].append(p)

        rows = []
        for col in md.columns:
            if not job_rhos[col]:
                continue
            rows.append(
                {
                    "marker": col,
                    "rho": np.median(job_rhos[col]),
                    "rho_std": (
                        np.std(job_rhos[col], ddof=1)
                        if len(job_rhos[col]) > 1
                        else 0.0
                    ),
                    "pval": np.median(job_pvals[col]),
                    "pval_std": (
                        np.std(job_pvals[col], ddof=1)
                        if len(job_pvals[col]) > 1
                        else 0.0
                    ),
                    "data_type": data_type_label,
                    "n_tested": len(md.columns),
                    "n_jobs": len(job_rhos[col]),
                }
            )

    # ----- single-vector branch ------------------------------------------
    else:
        common = sorted(set(metric.index) & set(md.index))
        if len(common) < 5:
            return pd.DataFrame()
        g = metric.loc[common]
        md_single = md.loc[common].dropna(axis=1, how="any")

        rows = []
        for col in md_single.columns:
            rho, p = spearmanr(g, md_single[col])
            rows.append(
                {
                    "marker": col,
                    "rho": rho,
                    "pval": p,
                    "data_type": data_type_label,
                    "n_tested": len(md_single.columns),
                }
            )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out["abs_rho"] = out["rho"].abs()

    # BH FDR
    valid = out["pval"].notna()
    if valid.sum() > 0:
        _, qvals, _, _ = multipletests(
            out.loc[valid, "pval"],
            alpha=alpha,
            method="fdr_bh",
        )
        out.loc[valid, "qval"] = qvals
    else:
        out["qval"] = np.nan

    out["significant"] = out["qval"] < alpha
    out = out.sort_values("abs_rho", ascending=False).reset_index(drop=True)

    # Shell membership label
    out["shell"] = out["marker"].apply(
        lambda m: "seed" if m in seeds_set else "S1"
    )
    return out
