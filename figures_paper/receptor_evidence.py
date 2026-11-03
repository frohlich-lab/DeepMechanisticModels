"""Model-independent evidence on receptor levels (Reviewer 2, major point 2).

Two analyses that do not involve the DMM:

1. Correlation of receptor transcript and protein levels (EGFR, ERBB2, ERBB3)
   with CyTOF phospho-ERBB2 (``p.HER2``, Y1248) across cell lines. Baseline
   p-ERBB2 is the EGF-condition t = 0 value, built exactly like the encoder's
   ``cytof_init`` features (replicate mean, KNN-imputed where missing). The
   CyTOF panel has no phospho-EGFR or phospho-ERBB3 channel, so EGF-induced
   responses of the four model observables (maximum over the EGF time course
   minus baseline) are used as context instead.
2. Random-forest importance of receptor levels for the CyTOF dynamics, with
   the estimator, targets and preprocessing of ``select_features.py`` (its
   ``RFE_*``/``HVGRFE_*`` branch). ``select_features.py`` runs ``fire.Fire`` at
   import, so the relevant logic is copied here rather than imported.
3. How far receptor levels survive the pipeline's full RFE, read from the
   feature files of the Fig. 1C feature-count scan.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from common import (
    Wildcards,
    evaluations_dir,
    features_dir,
    test_samples,
    training_samples,
)
from cytof import get_samples
from dmm.config_options import Conf
from dmm.feature_selection import load_data
from training_configuration import (
    BEST_FEATURE_SETS,
    CONTEXTS_FEATURES_1C,
    SPLITS_1C,
)
from util import load_petab_base_files

MODEL = "EGFR_MAPK__logobs"
DATA = "dream_cytof"
RECEPTORS = ("EGFR", "ERBB2", "ERBB3")
MODALITIES = ("transcriptomics", "proteomics")
MODALITY_LABELS = {"transcriptomics": "transcript", "proteomics": "protein"}
PERBB2 = "p.HER2"  # CyTOF channel for ERBB2 Y1248, model observable pERBB2

# CV splits of Figure 1 plus the "all" split used for Figures 2-4
SPLITS = ("BT20", "HCC1500", "HCC2185", "MCF7", "UACC3199", "all")
# seed 42 is the one select_features.py uses; the others check stability
SEEDS = (42, 43, 44, 45, 46)
N_JOBS = 3  # shared laptop: never more than 3 workers

RESPONSE_LABELS = {
    "baseline pERBB2": "baseline\npERBB2",
    "pERBB2_Y1248_obs": "ΔpERBB2",
    "pMEK_S222_obs": "ΔpMEK",
    "pERK_Y204_obs": "ΔpERK",
    "pRPS6KA1_S380_obs": "Δp90RSK",
}

CACHE_DIR = evaluations_dir / MODEL / DATA / "receptor_evidence"


def receptor_label(modality: str, gene: str) -> str:
    return f"{gene} {MODALITY_LABELS[modality]}"


def load_petab_tables(model: str = MODEL, data: str = DATA) -> dict:
    """Measurement and observable tables, as ``select_features.py`` loads them."""
    tables = load_petab_base_files(Conf(model=model, data=data))
    del tables["condition_table"]
    return tables


# ---------------------------------------------------------------------------
# 1. Correlations across cell lines
# ---------------------------------------------------------------------------


def receptor_levels(tables: dict, samples: list[str]) -> pd.DataFrame:
    """Receptor levels per cell line, replicate mean, no imputation.

    Columns are ``(modality, gene)``; values are on the scale of the input
    tables (the same values mechanistic integration rescales by).
    """
    mt = tables["measurement_table"]
    mt = mt[
        mt["measurementType"].isin(MODALITIES)
        & mt["FEATURE_ID"].isin(RECEPTORS)
        & (mt["time"] == 0)
    ]
    levels = mt.pivot_table(
        index="preequilibrationConditionId",
        columns=["measurementType", "FEATURE_ID"],
        values="measurement",
        aggfunc="mean",
    )
    return levels.reindex(
        index=samples,
        columns=pd.MultiIndex.from_product([MODALITIES, RECEPTORS]),
    )


def baseline_perbb2(tables: dict, samples: list[str]) -> pd.DataFrame:
    """Baseline p-ERBB2 as in the encoder's ``cytof_init`` features.

    Returns the KNN-imputed value used by the encoder and whether it was
    measured (``True``) or imputed (``False``) for each cell line.
    """
    imputed, *_ = load_data(
        contextualization="cytof_init",
        samples=samples,
        features=None,
        **tables,
    )
    raw, *_ = load_data(
        contextualization="cytof_init",
        samples=samples,
        features=None,
        impute=False,
        **tables,
    )
    imputed = imputed[PERBB2].reindex(samples)
    raw = raw[PERBB2].reindex(samples)
    measured = raw.notna()
    assert np.allclose(imputed[measured], raw[measured])
    return pd.DataFrame({"baseline pERBB2": imputed, "measured": measured})


def egf_responses(tables: dict, samples: list[str]) -> pd.DataFrame:
    """EGF-induced response per model observable: max over t > 0 minus t = 0.

    Uses the harmonised ``cytof_dynamic`` table (the DMM training targets)
    without imputation, so cell lines lacking a baseline value are NaN.
    """
    dyn, *_ = load_data(
        contextualization="cytof_dynamic",
        samples=samples,
        features=None,
        **tables,
    )
    egf = dyn.xs("EGF", axis=1, level=1)
    return pd.DataFrame(
        {
            obs: egf[obs].drop(columns=0.0).max(axis=1) - egf[obs][0.0]
            for obs in egf.columns.get_level_values(0).unique()
        }
    )


def correlate(x: pd.Series, y: pd.Series) -> dict:
    """Pearson and Spearman correlation on complete pairs."""
    xy = pd.concat([x, y], axis=1).dropna()
    r, p = stats.pearsonr(xy.iloc[:, 0], xy.iloc[:, 1])
    rho, p_rho = stats.spearmanr(xy.iloc[:, 0], xy.iloc[:, 1])
    return {
        "n": len(xy),
        "pearson_r": r,
        "pearson_p": p,
        "spearman_rho": rho,
        "spearman_p": p_rho,
    }


def partial_correlate(x: pd.Series, y: pd.Series, z: pd.Series) -> dict:
    """Pearson and Spearman partial correlation of x and y given z."""

    def _partial(a, b, c):
        design = np.column_stack([np.ones_like(c), c])
        res_a = a - design @ np.linalg.lstsq(design, a, rcond=None)[0]
        res_b = b - design @ np.linalg.lstsq(design, b, rcond=None)[0]
        r = np.corrcoef(res_a, res_b)[0, 1]
        dof = len(a) - 3
        t = r * np.sqrt(dof / (1 - r**2))
        return r, 2 * stats.t.sf(abs(t), dof)

    xyz = pd.concat([x, y, z], axis=1).dropna()
    r, p = _partial(*(xyz.iloc[:, i].to_numpy() for i in range(3)))
    rho, p_rho = _partial(
        *(xyz.iloc[:, i].rank().to_numpy() for i in range(3))
    )
    return {
        "n": len(xyz),
        "pearson_r": r,
        "pearson_p": p,
        "spearman_rho": rho,
        "spearman_p": p_rho,
    }


def correlation_table(levels: pd.DataFrame, readouts: pd.DataFrame):
    """Long table of correlations, receptor level x readout.

    Adds Benjamini-Hochberg q-values over all pairs in the table.
    """
    rows = [
        {
            "modality": modality,
            "gene": gene,
            "receptor": receptor_label(modality, gene),
            "readout": readout,
            **correlate(levels[(modality, gene)], readouts[readout]),
        }
        for modality, gene in levels.columns
        for readout in readouts.columns
    ]
    table = pd.DataFrame(rows)
    for method in ("pearson", "spearman"):
        p = table[f"{method}_p"]
        table[f"{method}_q"] = multipletests(p, method="fdr_bh")[1]
    return table


def correlation_summary(tables: dict | None = None) -> dict:
    """All correlation results used by the figure and the revision note."""
    tables = tables or load_petab_tables()
    samples = get_samples(DATA)
    levels = receptor_levels(tables, samples)
    baseline = baseline_perbb2(tables, samples)
    responses = egf_responses(tables, samples)
    readouts = pd.concat([baseline[["baseline pERBB2"]], responses], axis=1)

    baseline_rows = []
    for subset, mask in (
        ("all cell lines, encoder baseline", pd.Series(True, samples)),
        ("measured baseline only", baseline["measured"]),
        (
            "training cell lines (split 'all')",
            pd.Series([s not in test_samples for s in samples], samples),
        ),
    ):
        for modality, gene in levels.columns:
            baseline_rows.append(
                {
                    "subset": subset,
                    "receptor": receptor_label(modality, gene),
                    **correlate(
                        levels.loc[mask, (modality, gene)],
                        baseline.loc[mask, "baseline pERBB2"],
                    ),
                }
            )

    transcript_vs_protein = pd.DataFrame(
        [
            {
                "gene": gene,
                **correlate(
                    levels[("transcriptomics", gene)],
                    levels[("proteomics", gene)],
                ),
            }
            for gene in RECEPTORS
        ]
    )
    egfr_tx = levels[("transcriptomics", "EGFR")]
    partial_given_egfr_tx = pd.DataFrame(
        [
            {
                "receptor": receptor_label(modality, gene),
                "readout": readout,
                **partial_correlate(
                    levels[(modality, gene)], readouts[readout], egfr_tx
                ),
            }
            for modality, gene in levels.columns
            if (modality, gene) != ("transcriptomics", "EGFR")
            for readout in readouts.columns
        ]
    )
    return {
        "samples": samples,
        "levels": levels,
        "baseline": baseline,
        "readouts": readouts,
        "baseline_correlations": pd.DataFrame(baseline_rows),
        "readout_correlations": correlation_table(levels, readouts),
        "transcript_vs_protein": transcript_vs_protein,
        "partial_given_egfr_transcript": partial_given_egfr_tx,
    }


# ---------------------------------------------------------------------------
# 2. Random-forest importance, as in select_features.py
# ---------------------------------------------------------------------------


def get_hvg(input_df: pd.DataFrame, top_n: int = 500) -> pd.DataFrame:
    """Copy of ``select_features.get_hvg`` for an integer ``top_n``.

    As fixed in commit 50bd9e1 (no low-mean step, which compared
    floating-point residuals of the mean-centered inputs): keeps the
    ``top_n + 1`` most variable features.
    """
    var_threshold = sorted(np.nanvar(input_df, axis=0), reverse=True)[top_n]
    return input_df.loc[:, np.nanvar(input_df, axis=0) >= var_threshold]


def hvg_diagnostics(input_df: pd.DataFrame, top_n: int = 500) -> pd.DataFrame:
    """Why each receptor did or did not pass ``get_hvg``."""
    var_rank = (
        pd.Series(np.nanvar(input_df, axis=0), index=input_df.columns)
        .rank(ascending=False, method="min")
        .astype(int)
    )
    kept = get_hvg(input_df, top_n).columns
    return pd.DataFrame(
        {
            "variance_rank": var_rank.reindex(RECEPTORS),
            "in_hvg_set": [g in kept for g in RECEPTORS],
        },
        index=pd.Index(RECEPTORS, name="gene"),
    )


def rf_inputs(tables: dict, split: str, context: str):
    """Inputs and targets of the first RFE round for one split.

    Mirrors ``prepare_inputs_for_context`` and ``get_selected_features``:
    candidate features after the missingness filter of ``load_data``,
    KNN-imputed and mean-centered on the training cell lines; targets are
    the KNN-imputed ``cytof_dynamic`` table, centered.
    """
    samples = training_samples(Wildcards(DATA, split))
    output, *_ = load_data(
        contextualization="cytof_dynamic",
        samples=samples,
        features=None,
        **tables,
    )
    output = pd.DataFrame(
        KNNImputer().fit_transform(output),
        index=output.index,
        columns=output.columns,
    )
    inputs, *_ = load_data(
        contextualization=context, samples=samples, features=None, **tables
    )
    inputs = pd.DataFrame(
        KNNImputer().fit_transform(inputs),
        index=inputs.index,
        columns=inputs.columns,
    )
    inputs -= inputs.mean()
    output = output.loc[inputs.index, :]
    inputs = inputs.dropna(axis=1, how="any")
    output -= output.mean(axis=0)
    return inputs, output


def rf_importance(inputs, output, seed: int = 42) -> pd.Series:
    """First-round importance, same estimator and rule as the pipeline.

    Impurity importance above 500 features, permutation importance otherwise.
    """
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "regressor",
                RandomForestRegressor(
                    random_state=seed, max_features=0.8, n_jobs=N_JOBS
                ),
            ),
        ]
    ).fit(inputs, output)
    if inputs.shape[1] > 500:
        importances = pipeline.named_steps["regressor"].feature_importances_
    else:
        importances = permutation_importance(
            pipeline, inputs, output, n_repeats=10, random_state=42
        ).importances_mean
    return pd.Series(importances, index=inputs.columns)


def compute_rf_importances(
    tables: dict | None = None,
    splits=SPLITS,
    candidate_sets=("all", "hvg"),
):
    """Importances for all candidate features, per split/context/set/seed.

    ``candidate_set`` is ``"all"`` (every feature passing the missingness
    filter) or ``"hvg"`` (the pipeline's HVG prefilter, i.e. the first round
    of ``HVGRFE_*``). Both use ``SEEDS``; seed 42 is the pipeline's forest.
    """
    tables = tables or load_petab_tables()
    importances, diagnostics = [], []
    for split in splits:
        for context in MODALITIES:
            inputs, output = rf_inputs(tables, split, context)
            diagnostics.append(
                hvg_diagnostics(inputs)
                .reset_index()
                .assign(
                    passes_missingness_filter=[g in inputs for g in RECEPTORS],
                    split=split,
                    context=context,
                    n_candidates=inputs.shape[1],
                )
            )
            for candidate_set in candidate_sets:
                x = get_hvg(inputs) if candidate_set == "hvg" else inputs
                for seed in SEEDS:
                    print(f"{split} {context} {candidate_set} seed {seed}")
                    importances.append(
                        rf_importance(x, output, seed)
                        .rename("importance")
                        .rename_axis("feature")
                        .reset_index()
                        .assign(
                            split=split,
                            context=context,
                            candidate_set=candidate_set,
                            seed=seed,
                        )
                    )
    return pd.concat(importances, ignore_index=True), pd.concat(
        diagnostics, ignore_index=True
    )


def load_rf_importances(recompute: bool = False, tables: dict | None = None):
    """Cached ``compute_rf_importances`` (parquet under ``eval/``).

    Parquet rather than CSV: ``run_notebooks.py`` redirects ``to_csv``.
    """
    imp_file = CACHE_DIR / "rf_importances.parquet"
    diag_file = CACHE_DIR / "rf_receptor_filters.parquet"
    if recompute or not (imp_file.exists() and diag_file.exists()):
        importances, diagnostics = compute_rf_importances(tables)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        importances.to_parquet(imp_file)
        diagnostics.to_parquet(diag_file)
    return pd.read_parquet(imp_file), pd.read_parquet(diag_file)


def receptor_ranks(importances: pd.DataFrame) -> pd.DataFrame:
    """Rank and percentile of each receptor within its modality.

    Importances are averaged over seeds first. ``rank`` is 1 + the number of
    features with strictly higher importance; ``percentile`` is the share of
    features with strictly lower importance (100 = most important, 0 = tied
    with the least important, e.g. never used for a split).
    """
    rows = []
    for (split, context, candidate_set), grp in importances.groupby(
        ["split", "context", "candidate_set"]
    ):
        imp = grp.groupby("feature")["importance"].mean()
        for gene in RECEPTORS:
            row = {
                "split": split,
                "context": context,
                "candidate_set": candidate_set,
                "gene": gene,
                "receptor": receptor_label(context, gene),
                "n_features": len(imp),
                "n_nonzero": int((imp > 0).sum()),
                "n_seeds": grp["seed"].nunique(),
            }
            if gene in imp:
                value = imp[gene]
                row |= {
                    "importance": value,
                    "rank": int((imp > value).sum() + 1),
                    "percentile": 100 * (imp < value).mean(),
                }
            rows.append(row)
    return pd.DataFrame(rows)


def zero_importance_rates(importances: pd.DataFrame) -> pd.DataFrame:
    """Share of single forests (one split, one seed) giving zero importance.

    Compares each receptor with the average feature of its modality, on the
    full candidate set.
    """
    full = importances[importances["candidate_set"] == "all"]
    rows = []
    for context, grp in full.groupby("context"):
        zero = grp["importance"] == 0
        rows.append(
            {
                "context": context,
                "feature": "all features",
                "zero_rate": zero.mean(),
            }
        )
        for gene in RECEPTORS:
            is_gene = grp["feature"] == gene
            if is_gene.any():
                rows.append(
                    {
                        "context": context,
                        "feature": receptor_label(context, gene),
                        "zero_rate": zero[is_gene].mean(),
                    }
                )
    return pd.DataFrame(rows)


def first_round_status(
    ranks: pd.DataFrame, filters: pd.DataFrame
) -> pd.DataFrame:
    """What happens to each receptor in the pipeline's first RFE round.

    One row per receptor and split: excluded by the missingness filter, by
    the HVG prefilter (not among the 501 most variable candidates), or
    ranked among the HVG candidates (rank out of ``n_features``).
    """
    hvg = ranks[ranks["candidate_set"] == "hvg"].set_index(
        ["split", "context", "gene"]
    )
    rows = []
    for _, f in filters.iterrows():
        key = (f["split"], f["context"], f["gene"])
        if not f["passes_missingness_filter"]:
            status = "excluded: >30% missing"
        elif f["in_hvg_set"]:
            status = "ranked"
        else:
            status = "excluded: not among 501 most variable"
        rows.append(
            {
                "split": f["split"],
                "receptor": receptor_label(f["context"], f["gene"]),
                "status": status,
                "variance_rank": f["variance_rank"],
                "n_candidates": f["n_candidates"],
                "rank": hvg.loc[key, "rank"],
                "n_features": hvg.loc[key, "n_features"],
            }
        )
    return pd.DataFrame(rows)


def check_hvg_replication(importances: pd.DataFrame) -> pd.DataFrame:
    """Do the pipeline's saved selections lie inside our replicated HVG sets?

    Reads ``features/<model>/<data>/train/<context>__<split>__HVGRFE_*`` where
    present (untracked pipeline output) and reports features outside the set.
    Only feature files produced after the HVG fix (50bd9e1) should match.
    """
    rows = []
    hvg = importances[importances["candidate_set"] == "hvg"]
    for (split, context), grp in hvg.groupby(["split", "context"]):
        for path in sorted(
            (features_dir / MODEL / DATA / "train").glob(
                f"{context}__{split}__HVGRFE_*_permute.csv"
            )
        ):
            selected = pd.read_csv(path, index_col=0).columns
            outside = sorted(set(selected) - set(grp["feature"]))
            rows.append(
                {
                    "file": path.name,
                    "n_selected": len(selected),
                    "n_outside_replicated_hvg": len(outside),
                    "receptors_selected": [
                        g for g in RECEPTORS if g in selected
                    ],
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Survival through the full RFE (Fig. 1C feature-count scan)
# ---------------------------------------------------------------------------


def rfe_survival(splits: tuple[str, ...] = tuple(sorted(SPLITS_1C))):
    """Smallest selection of the Fig. 1C scan that still contains a receptor.

    Reads the pipeline's ``<context>__<split>__HVGRFE_<N>_permute`` feature
    files (untracked output of the ``figure1c`` workflow) for every size N
    of the scan. Only files produced with the HVG fix (50bd9e1) are
    meaningful. ``check_hvg_replication`` is necessary but not sufficient to
    confirm that: older transcript selections often lie inside the fixed HVG
    set too.

    One row per receptor and split. ``smallest_selection`` is the number of
    features in the smallest scanned selection containing the receptor (NaN
    if none does); ``nested`` checks that every larger selection contains it
    too; ``in_final_size_selection`` refers to the selection with the size of
    ``BEST_FEATURE_SETS`` (the feature sets of the final models). Splits
    without feature files are skipped.
    """
    rows = []
    train_dir = features_dir / MODEL / DATA / "train"
    for context in MODALITIES:
        sizes = sorted(
            int(features.split("_")[1])
            for ctx, features in CONTEXTS_FEATURES_1C
            if ctx == context and features.startswith("HVGRFE_")
        )
        final_size = int(BEST_FEATURE_SETS[context].split("_")[1])
        for split in splits:
            selections = {
                size: set(pd.read_csv(path, index_col=0, nrows=0).columns)
                for size in sizes
                if (
                    path := train_dir
                    / f"{context}__{split}__HVGRFE_{size}_permute.csv"
                ).exists()
            }
            if not selections:
                continue
            for gene in RECEPTORS:
                kept = [
                    size for size, sel in selections.items() if gene in sel
                ]
                rows.append(
                    {
                        "split": split,
                        "context": context,
                        "gene": gene,
                        "receptor": receptor_label(context, gene),
                        "n_sizes_found": len(selections),
                        "smallest_selection": min(
                            (len(selections[size]) for size in kept),
                            default=np.nan,
                        ),
                        "nested": all(
                            gene in sel
                            for size, sel in selections.items()
                            if kept and size >= min(kept)
                        ),
                        "final_size": final_size,
                        "in_final_size_selection": gene
                        in selections.get(final_size, set()),
                    }
                )
    return pd.DataFrame(rows)


def rfe_path(
    inputs: pd.DataFrame, output: pd.DataFrame, sizes
) -> dict[int, list[str]]:
    """Selections of the pipeline's ``HVGRFE_<N>_permute`` RFE for each N.

    Copies the ``RFE_*``/``HVGRFE_*`` loop of ``select_features.py``
    (reduce factor 0.8, impurity importance above 500 features, permutation
    importance otherwise, columns reordered by importance after each step;
    final selection: top N by permutation importance). Runs with different
    N follow the same elimination path, since the forest seed is fixed, so
    one pass covers all sizes. ``inputs`` are the HVG candidates.

    Outside the pipeline's container (scikit-learn 1.7.1, x86_64) the
    forests, and with them the selections, differ: with scikit-learn 1.8.0
    on arm64, 7 of 130 Fig. 1C CV-split selections came out identical to the
    pipeline's (median overlap 75 %).
    """
    reduce_factor = 0.8
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "regressor",
                RandomForestRegressor(
                    random_state=42, max_features=reduce_factor
                ),
            ),
        ]
    )
    x = inputs
    remaining = sorted(sizes, reverse=True)
    selections = {}
    while remaining:
        pipeline.fit(x, output)
        # sizes whose elimination loop stops at the current stage
        stop_here = [n for n in remaining if x.shape[1] * reduce_factor <= n]
        perm = None
        if stop_here or x.shape[1] <= 500:
            perm = permutation_importance(
                pipeline, x, output, n_repeats=10, random_state=42
            ).importances_mean
        for n in stop_here:
            selections[n] = list(x.columns[np.argsort(perm)[::-1][:n]])
            remaining.remove(n)
        if not remaining:
            break
        importances = (
            pipeline.named_steps["regressor"].feature_importances_
            if x.shape[1] > 500
            else perm
        )
        n_target = int(np.ceil(len(importances) * reduce_factor))
        if n_target == x.shape[1]:
            n_target -= 1  # reduce by at least one feature
        x = x.iloc[:, np.argsort(importances)[::-1][:n_target]]
    return selections
