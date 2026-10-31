"""HVG prefilter of select_features.py: candidate sets before and after a fix.

``prepare_inputs_for_context`` mean-centers the omics inputs before
``get_hvg`` runs, so the "drop the 20 % of features with the lowest mean" step
compares floating-point residuals of zero. This script rebuilds the HVG
candidate set of every split and omics context three ways:

- ``current``: ``get_hvg`` on the centered inputs, as the pipeline runs it;
- ``mean_uncentered`` (fix A): the mean step uses the uncentered, imputed
  inputs; the variance step is unchanged;
- ``no_mean_step`` (fix B): the mean step is dropped; the 501 most variable
  features are kept.

It reuses ``rf_inputs``, ``get_hvg`` and ``hvg_diagnostics`` from
``figures_paper/receptor_evidence.py`` (branch
``claude/revision-receptor-evidence-b04f76``).

Usage: python revision_notes/hvg_prefilter_impact.py [OUTDIR]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

_root = Path(__file__).resolve().parent.parent
for _p in (_root, _root / "figures_paper"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from receptor_evidence import (
    DATA,
    MODALITIES,
    MODEL,
    RECEPTORS,
    SPLITS,
    get_hvg,
    hvg_diagnostics,
    load_petab_tables,
    rf_inputs,
)

from common import Wildcards, features_dir, training_samples
from dmm.feature_selection import load_data

TOP_N = 500
VARIANTS = ("current", "mean_uncentered", "no_mean_step")

# Curated genes of select_features.py worth tracking (receptors, PAM50, IHC)
PAM50 = (
    "FOXC1 MIA NDC80 CEP55 ANLN MELK GPR160 TMEM45B ESR1 FOXA1 ERBB2 GRB7 "
    "FGFR4 BLVRA BAG1 CDC20 CCNE1 ACTR3B MYC SFRP1 KRT14 KRT17 KRT5 MLPH "
    "CCNB1 CDC6 TYMS UBE2T RRM2 MMP11 CXXC5 ORC6 MDM2 KIF2C PGR MKI67 BCL2 "
    "EGFR PHGDH CDH3 NAT1 SLC39A6 MAPT UBE2C PTTG1 EXO1 CENPF NUF2 MYBL2 BIRC5"
).split()
IHC = (
    "ERBB2 EGFR KRT5 KRT6A KRT6B PGR ESR1 MKI67 TP53 GATA3 SOX10 SCGB2A2 "
    "SCGB2A1 KRT7 PIP CDX2 KRT20 NKX2-1"
).split()
NOTABLE = sorted(set(RECEPTORS) | set(PAM50) | set(IHC))


def imputed_inputs(tables: dict, split: str, context: str) -> pd.DataFrame:
    """``rf_inputs`` inputs before centering (missingness filter + KNN)."""
    samples = training_samples(Wildcards(DATA, split))
    inputs, *_ = load_data(
        contextualization=context, samples=samples, features=None, **tables
    )
    return pd.DataFrame(
        KNNImputer().fit_transform(inputs),
        index=inputs.index,
        columns=inputs.columns,
    )


def hvg(centered: pd.DataFrame, means, top_n: int = TOP_N) -> pd.Index:
    """``get_hvg`` with the mean step on ``means`` (None: no mean step).

    The variance step runs on the centered inputs, as in the pipeline.
    """
    if means is not None:
        means = means.reindex(centered.columns)
        centered = centered.loc[:, means >= np.percentile(means, 20)]
    var = np.nanvar(centered, axis=0)
    return centered.columns[var >= sorted(var, reverse=True)[top_n]]


def candidate_sets(tables: dict, split: str, context: str):
    centered, _ = rf_inputs(tables, split, context)
    uncentered = imputed_inputs(tables, split, context)
    # rf_inputs centers the same imputed matrix; check we rebuilt it exactly
    assert np.array_equal(
        (uncentered - uncentered.mean()).to_numpy(), centered.to_numpy()
    )
    sets = {
        "current": get_hvg(centered, TOP_N).columns,
        "mean_uncentered": hvg(centered, uncentered.mean()),
        "no_mean_step": hvg(centered, None),
    }
    assert sets["current"].equals(hvg(centered, centered.mean()))
    return centered, uncentered, sets


def var_rank(centered: pd.DataFrame) -> pd.Series:
    return (
        pd.Series(np.nanvar(centered, axis=0), index=centered.columns)
        .rank(ascending=False, method="min")
        .astype(int)
    )


def saved_selections(split: str, context: str) -> dict:
    """Selected features of the pipeline's saved HVGRFE / multimodal files."""
    out = {}
    train_dir = features_dir / MODEL / DATA / "train"
    prefix = {"proteomics": "p", "transcriptomics": "t"}[context]
    for path in sorted(train_dir.glob(f"{context}__{split}__HVGRFE_*.csv")):
        out[path.name] = list(pd.read_csv(path, index_col=0, nrows=0).columns)
    for path in sorted(train_dir.glob(f"multimodal__{split}__RFE_*.csv")):
        cols = pd.read_csv(path, index_col=0, nrows=0).columns
        out[path.name] = [
            c[1:] for c in cols if c.startswith(prefix) and not c[1:2] == "."
        ]
    return out


def compare(tables: dict | None = None):
    tables = tables or load_petab_tables()
    summary, notable, receptors, saved, members = [], [], [], [], []
    for split in SPLITS:
        for context in MODALITIES:
            print(split, context, flush=True)
            centered, uncentered, sets = candidate_sets(tables, split, context)
            rank = var_rank(centered)
            top = set(rank.index[rank <= TOP_N + 1])
            cur_means = centered.mean()
            removed_now = set(
                centered.columns[cur_means < np.percentile(cur_means, 20)]
            )
            unc_means = uncentered.mean()
            removed_fix = set(
                centered.columns[unc_means < np.percentile(unc_means, 20)]
            )
            cur, fa, fb = (set(sets[v]) for v in VARIANTS)
            summary.append(
                {
                    "split": split,
                    "context": context,
                    "n_candidates": centered.shape[1],
                    "n_current": len(cur),
                    "n_mean_uncentered": len(fa),
                    "n_no_mean_step": len(fb),
                    "current_and_mean_uncentered": len(cur & fa),
                    "current_and_no_mean_step": len(cur & fb),
                    "mean_uncentered_and_no_mean_step": len(fa & fb),
                    "top501_removed_by_current_mean_step": len(
                        top & removed_now
                    ),
                    "top501_removed_by_uncentered_mean_step": len(
                        top & removed_fix
                    ),
                    "max_abs_centered_mean": float(cur_means.abs().max()),
                    "uncentered_mean_threshold": float(
                        np.percentile(unc_means, 20)
                    ),
                }
            )
            for gene in NOTABLE:
                if gene not in centered:
                    continue
                flags = {v: gene in sets[v] for v in VARIANTS}
                if len(set(flags.values())) > 1 or gene in RECEPTORS:
                    notable.append(
                        {
                            "split": split,
                            "context": context,
                            "gene": gene,
                            "variance_rank": rank[gene],
                            "uncentered_mean": unc_means[gene],
                            **flags,
                        }
                    )
            receptors.append(
                hvg_diagnostics(centered, TOP_N)
                .reset_index()
                .assign(split=split, context=context)
            )
            union = sorted(cur | fa | fb)
            members.append(
                pd.DataFrame(
                    {
                        "split": split,
                        "context": context,
                        "feature": union,
                        "variance_rank": rank[union].to_numpy(),
                        "uncentered_mean": unc_means[union].to_numpy(),
                        "centered_mean": cur_means[union].to_numpy(),
                        **{v: np.isin(union, sets[v]) for v in VARIANTS},
                    }
                )
            )
            for name, selected in saved_selections(split, context).items():
                saved.append(
                    {
                        "file": name,
                        "context": context,
                        "n_selected": len(selected),
                        **{
                            f"outside_{v}": sorted(
                                set(selected) - set(sets[v])
                            )
                            for v in VARIANTS
                        },
                    }
                )
    return (
        pd.DataFrame(summary),
        pd.DataFrame(notable),
        pd.concat(receptors, ignore_index=True),
        pd.DataFrame(saved),
        pd.concat(members, ignore_index=True),
    )


if __name__ == "__main__":
    outdir = Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else _root / "eval" / MODEL / DATA / "hvg_prefilter"
    )
    outdir.mkdir(parents=True, exist_ok=True)
    tables = load_petab_tables()
    summary, notable, receptors, saved, members = compare(tables)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(summary.to_string())
    print(notable.to_string())
    print(receptors.to_string())
    print(saved.to_string())
    summary.to_csv(outdir / "summary.csv", index=False)
    notable.to_csv(outdir / "notable_genes.csv", index=False)
    receptors.to_csv(outdir / "receptors.csv", index=False)
    saved.to_csv(outdir / "saved_selections.csv", index=False)
    members.to_csv(outdir / "membership.csv", index=False)
