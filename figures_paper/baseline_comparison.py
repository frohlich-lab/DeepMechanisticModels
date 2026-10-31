"""DMM vs. reference models on held-out cell lines (Fig. 1C).

- per-(cell line x condition x observable) RMSE of the DMM, the mechanistic
  controls (``avg_model``, ``sample``) and the regression baselines
- paired Wilcoxon signed-rank tests of the DMM against each of them
- the measurement noise floor, estimated from replicate measurements

Run as a script to print the noise floor and, where the aggregated
evaluation outputs exist, the paired tests:

    python -m figures_paper.baseline_comparison
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import petab.v1 as petab
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from common import (  # noqa: E402
    MEASUREMENTS_FILE,
    OBSERVABLES_FILE,
    REGRESSION_MODES,
    Wildcards,
    evaluations_dir,
    test_samples,
    training_samples,
)

COMBO = ["cell_line", "condition", "observable"]
MECHANISTIC_REFS = ["DMM", "avg_model", "sample"]
BASELINES = ["avg_model", *REGRESSION_MODES, "sample"]

# Which held-out predictions count as "test". The models of split "all" are
# trained on all 58 training lines and predict the five test lines; each CV
# split holds out one training line, and those lines also informed the
# choice of hyperparameters.
HELDOUT_SPLITS = {
    "test": ["all"],
    "cv+test": None,  # every split's held-out line (Fig. 1C before revision)
}


# ── Measurement noise floor ───────────────────────────────────────────


def load_scored_measurements(
    model: str = "EGFR_MAPK__logobs",
    data: str = "dream_cytof",
    cell_lines=None,
) -> pd.DataFrame:
    """CyTOF measurements the mechanistic models are scored on.

    One row per replicate measurement of a model observable, on the PEtab
    measurement scale (cell-averaged arcsinh-transformed intensities, as in
    ``data/cytof.csv``). Full-medium conditions are not scored and dropped.
    """
    observables = pd.read_csv(
        OBSERVABLES_FILE.format(model=model, data=data), sep="\t", index_col=0
    )
    meas = pd.read_csv(
        MEASUREMENTS_FILE.format(model=model, data=data),
        sep="\t",
        index_col=0,
        dtype={"date": str, "time_course": str},
    )
    meas = meas[
        (meas["measurementType"] == "cytof")
        & meas[petab.OBSERVABLE_ID].isin(observables.index)
        & meas[petab.MEASUREMENT].notna()
        & ~meas[petab.SIMULATION_CONDITION_ID].str.endswith("__full")
    ]
    meas = meas.assign(
        cell_line=meas[petab.PREEQUILIBRATION_CONDITION_ID],
        condition=meas[petab.SIMULATION_CONDITION_ID].str.split("__").str[1],
        observable=meas[petab.OBSERVABLE_ID],
    )
    if cell_lines is not None:
        meas = meas[meas["cell_line"].isin(cell_lines)]
    return meas[
        [*COMBO, petab.TIME, "date", "time_course", petab.MEASUREMENT]
    ].reset_index(drop=True)


def replicate_pairs(meas: pd.DataFrame) -> pd.DataFrame:
    """Pairs of measurements of one cell line, condition, observable and time.

    Each cell line was measured on a single date, as two interleaved time
    courses (A and B) that share the 0-min point under every condition and
    the 9-min point under EGF; each shared point gives one A-B pair. A few
    lines were measured as two complete sets (same date and time-course
    label, separate files); these give one pair per time point within a time
    course. Where both occur at one point (A, A, B, B), which A belongs with
    which B is unknown, so only the within-time-course pairs are used there.
    """
    point = COMBO + [petab.TIME]

    n_per_course = meas.groupby(point + ["time_course"])[
        petab.MEASUREMENT
    ].transform("size")
    if n_per_course.max() > 2:
        raise ValueError("expected at most two sets per time course")
    sets = (
        meas[n_per_course == 2]
        .groupby(point + ["time_course"])[petab.MEASUREMENT]
        .agg(["first", "last"])
    )
    set_pairs = (
        (sets["first"] - sets["last"])
        .rename("difference")
        .reset_index()
        .assign(pairing="replicate set")
    )

    courses = meas.groupby(point)["time_course"].agg(
        lambda s: "".join(sorted(s))
    )
    ab = (
        meas.set_index(point)
        .loc[courses[courses == "AB"].index]
        .pivot_table(
            index=point, columns="time_course", values=petab.MEASUREMENT
        )
    )
    ab_pairs = (
        (ab["A"] - ab["B"])
        .rename("difference")
        .reset_index()
        .assign(time_course="A-B", pairing="time course A vs B")
    )
    return pd.concat([ab_pairs, set_pairs], ignore_index=True)


def noise_floor(pairs: pd.DataFrame, by=None) -> pd.DataFrame:
    """Noise SD of a single measurement, pooled over replicate pairs.

    For a pair, E[(x1 - x2)^2] = 2 sigma^2, so sigma = sqrt(mean(d^2) / 2):
    the RMSE that predicting every expected value exactly would still incur
    against individual measurements, i.e. the floor for the per-combo RMSEs.
    The RMSE between the two replicates themselves is sqrt(2) sigma.
    """
    groups = pairs.groupby(by) if by else [("all", pairs)]
    rows = []
    for key, grp in groups:
        msd = np.mean(np.square(grp["difference"]))
        rows.append(
            {
                **dict(zip(by or ["pairs"], np.atleast_1d(key))),
                "n_pairs": len(grp),
                "n_cell_lines": grp["cell_line"].nunique(),
                "sigma": np.sqrt(msd / 2),
                "replicate_rmse": np.sqrt(msd),
            }
        )
    return pd.DataFrame(rows)


# ── Per-combo RMSE of every method ────────────────────────────────────


def _select(df, context, features, splits):
    df = df[df["context"] == context]
    # avg_model and sample carry features "None" and are copied per context
    df = df[
        df["ref"].isin(["avg_model", "sample"]) | (df["features"] == features)
    ]
    if splits is not None:
        df = df[df["samples"].isin(splits)]
    return df


def combo_rmse(
    by_cl: pd.DataFrame, context: str, features: str, splits=None
) -> pd.DataFrame:
    """Per-combo RMSE as ``aggregate_and_log`` scored it.

    DMM values are medians over random seeds, training-set values also over
    the CV splits that trained on the line. Regressors are scored against
    replicate-averaged, time-harmonised targets here (see
    ``matched_combo_rmse``).
    """
    df = _select(
        by_cl.rename(columns={"sample": "cell_line"}),
        context,
        features,
        splits,
    )
    return (
        df.groupby(["ref", "dataset", *COMBO])["rmse"].median().reset_index()
    )


def matched_combo_rmse(
    traj: pd.DataFrame, context: str, features: str, splits=None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-combo RMSE of every method on the same individual measurements.

    ``evaluate_regressors.py`` scores the regressors against their own
    training targets: replicate means on a gap-filled time grid. At the
    shared 0- and 9-min points they therefore see one averaged value where
    the mechanistic models see two replicates. Here the regressors'
    predictions are scored against the measurement rows the mechanistic
    models are scored on, matched on time point.

    Returns the per-combo RMSE (medians over seeds and splits, as in
    ``combo_rmse``) and, per regressor and dataset, the fraction of
    measurement rows that had a regressor prediction to score.
    """
    df = _select(
        traj.rename(columns={"sample": "cell_line"}), context, features, splits
    )
    keys = ["samples", "dataset", *COMBO, petab.TIME]
    # avg_model has exactly one row per measurement and split
    rows = df.loc[df["ref"] == "avg_model", keys + ["obs"]]

    mech = df.loc[
        df["ref"].isin(MECHANISTIC_REFS), keys + ["ref", "job", "obs", "sim"]
    ]
    reg = df.loc[df["ref"].isin(REGRESSION_MODES), keys + ["ref", "sim"]]
    if reg.duplicated(keys + ["ref"]).any():
        raise ValueError("expected one regressor prediction per time point")
    reg = rows.merge(reg, on=keys, how="left")
    coverage = (
        reg.assign(scored=reg["sim"].notna())
        .groupby(["ref", "dataset"])["scored"]
        .mean()
        .reset_index()
    )

    scored = pd.concat([mech, reg.dropna(subset=["sim"])], ignore_index=True)
    scored["sq"] = np.square(scored["obs"] - scored["sim"])
    # as in aggregate_and_log, failed simulations do not count
    scored = scored[np.isfinite(scored["sq"])]
    per_run = (
        scored.groupby(
            ["ref", "dataset", "samples", "job", *COMBO], dropna=False
        )["sq"]
        .mean()
        .pow(0.5)
        .rename("rmse")
        .reset_index()
    )
    return (
        per_run.groupby(["ref", "dataset", *COMBO])["rmse"]
        .median()
        .reset_index(),
        coverage,
    )


# ── Paired tests ──────────────────────────────────────────────────────


def _sign_flip_p(deltas: np.ndarray) -> float:
    """Exact two-sided sign-flip test of mean(deltas) = 0."""
    observed = abs(deltas.mean())
    flips = np.array(list(itertools.product([1, -1], repeat=len(deltas))))
    return float(
        np.mean(np.abs((flips * deltas).mean(axis=1)) >= observed - 1e-12)
    )


def paired_tests(
    combo: pd.DataFrame,
    dataset: str = "val",
    reference: str = "DMM",
    baselines=BASELINES,
) -> pd.DataFrame:
    """Wilcoxon signed-rank test of the DMM against each baseline.

    Pairs are (cell line x condition x observable) combos with an RMSE for
    both methods. ``delta`` is DMM minus baseline, so negative values favour
    the DMM; ``pct_lower`` compares the median RMSEs (what the boxes in
    Fig. 1C show). Combos of one cell line are not independent, so
    ``p_cell_lines`` repeats the comparison with one mean delta per cell
    line (exact sign-flip test; with five lines its smallest value is
    0.0625).
    """
    wide = combo[combo["dataset"] == dataset].pivot_table(
        index=COMBO, columns="ref", values="rmse"
    )
    rows = []
    for baseline in baselines:
        if baseline not in wide:
            continue
        pair = wide[[reference, baseline]].dropna()
        delta = pair[reference] - pair[baseline]
        test = wilcoxon(pair[reference], pair[baseline])
        per_line = delta.groupby(level="cell_line").mean()
        rows.append(
            {
                "baseline": baseline,
                "n_combos": len(pair),
                "n_cell_lines": len(per_line),
                "median_rmse_dmm": pair[reference].median(),
                "median_rmse_baseline": pair[baseline].median(),
                "median_delta": delta.median(),
                "pct_lower": 100
                * (1 - pair[reference].median() / pair[baseline].median()),
                "median_pct_lower": 100 * (-delta / pair[baseline]).median(),
                "dmm_lower": int((delta < 0).sum()),
                "W": test.statistic,
                "p": test.pvalue,
                "cell_lines_dmm_lower": int((per_line < 0).sum()),
                "p_cell_lines": _sign_flip_p(per_line.to_numpy()),
            }
        )
    out = pd.DataFrame(rows)
    out["p_holm"] = multipletests(out["p"], method="holm")[1]
    return out


# ── Plotting ──────────────────────────────────────────────────────────

# (ref, dataset) -> tick label; "val" holds the held-out predictions
PLOT_LABELS = {
    ("DMM", "train"): "DMM train",
    ("DMM", "val"): "DMM test",
    ("avg_model", "val"): "Neg. control",
    ("linreg", "val"): "Linear reg.",
    ("lasso", "val"): "Lasso",
    ("elasticnet", "val"): "Elastic net",
    ("sample", "val"): "Pos. control",
}


def plot_rmse_by_method(ax, combo: pd.DataFrame, noise_sigma: float = None):
    """Boxplot of per-combo RMSE: DMM train/test and the test-set baselines."""
    import seaborn as sns

    from figures_paper.figure_config import (
        BARPLOT_REF_LINE_COLORS,
        FONTSIZE,
        LW_THIN,
        REF_PALETTE,
        configure_axis_spines,
    )

    colors = {
        ("DMM", "train"): "#1f77b4",
        ("DMM", "val"): "#d62728",
        ("avg_model", "val"): BARPLOT_REF_LINE_COLORS["negative control"],
        ("sample", "val"): REF_PALETTE["sample"],
        **{(m, "val"): "#bdbdbd" for m in REGRESSION_MODES},
    }
    keys = list(zip(combo["ref"], combo["dataset"]))
    df = combo.assign(method=[PLOT_LABELS.get(k) for k in keys])
    df = df[df["method"].notna()]
    order = [k for k in PLOT_LABELS if k in set(keys)]
    names = [PLOT_LABELS[k] for k in order]

    sns.boxplot(
        data=df,
        x="method",
        y="rmse",
        order=names,
        hue="method",
        palette={PLOT_LABELS[k]: colors[k] for k in order},
        legend=False,
        width=0.6,
        linewidth=LW_THIN,
        fliersize=0,
        ax=ax,
    )
    sns.stripplot(
        data=df,
        x="method",
        y="rmse",
        order=names,
        color="#333333",
        size=1,
        alpha=0.5,
        jitter=0.2,
        ax=ax,
    )
    if noise_sigma is not None:
        ax.axhline(
            noise_sigma, color="#555555", linestyle="--", linewidth=LW_THIN
        )
        ax.text(
            1.01,
            noise_sigma,
            "noise",
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=FONTSIZE,
            color="#555555",
        )
    ax.set_xlabel("")
    ax.set_ylabel("RMSE")
    ax.tick_params(axis="x", labelsize=FONTSIZE, rotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
        label.set_rotation_mode("anchor")
    configure_axis_spines(ax, top=False, right=False, left=True, bottom=True)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.grid(False)
    return ax


def main(
    model: str = "EGFR_MAPK__logobs",
    data: str = "dream_cytof",
    context: str = "cytof_init",
    features: str = "RFE_8_permute",
    figure: str = "figure1a",
):
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    train_lines = training_samples(Wildcards(data, "all"))
    meas = load_scored_measurements(model, data, train_lines + test_samples)
    pairs = replicate_pairs(meas)
    pairs["cell_lines"] = np.where(
        pairs["cell_line"].isin(test_samples), "test", "train"
    )
    print(
        f"{meas['cell_line'].nunique()} cell lines, {len(meas)} measurements\n"
    )
    for by in (
        None,
        ["pairing"],
        ["cell_lines"],
        ["pairing", "cell_lines"],
        ["pairing", "observable"],
        ["pairing", "condition", "time"],
    ):
        print(noise_floor(pairs, by).round(3).to_string(index=False), "\n")

    outdir = evaluations_dir / model / data
    traj_file = outdir / f"trajectories_{figure}.csv"
    if not traj_file.exists():
        print(f"{traj_file} not found, skipping the paired tests")
        return
    traj = pd.read_csv(traj_file, index_col=0)
    by_cl = pd.read_csv(outdir / f"by_cl_cond_obs_{figure}.csv", index_col=0)
    for heldout, splits in HELDOUT_SPLITS.items():
        matched, coverage = matched_combo_rmse(traj, context, features, splits)
        print(f"== held-out set: {heldout}\n{coverage.to_string(index=False)}")
        for label, combo in (
            ("matched", matched),
            ("as aggregated", combo_rmse(by_cl, context, features, splits)),
        ):
            print(
                f"-- {label}\n",
                paired_tests(combo).round(4).to_string(index=False),
            )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
