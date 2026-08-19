# Shared configuration for figure generation
import matplotlib
import matplotlib.pyplot as _plt
import seaborn as sns

from training_configuration import EXTRA_MARKERS_5B_PROT, EXTRA_MARKERS_5B_TX

# ── tab20 palette ─────────────────────────────────────────────────────
# Slots 0–7 (dark even indices) → 4 contexts
# Slots 8–14 → 3 transcript/protein pairs (dark/light) + estimated EGFR
# Remaining models → Set2 palette
_tab20 = _plt.cm.tab20.colors
_set2 = _plt.cm.Set2.colors
_tab10 = _plt.cm.tab10.colors

# ── Global font sizes ─────────────────────────────────────────────────
FONTSIZE = 8
FONTSIZE_HEADING = 10

# ── Line widths (0.5 increments) ─────────────────────────────────────
LW_THICK = 1.5  # primary lines: experimental data, DMM mean
LW_MEDIUM = 1.0  # secondary lines: references, error bars, diagonal
LW_THIN = 0.5  # background: seeds, quantile markers

# ── Embedding scatter colours ─────────────────────────────────────────
EMBEDDING_COLORS = {
    "Luminal": "#7b2d8e",  # purple
    "Basal": "#1b9e77",  # teal
    "Normal": "#d95f02",  # orange
}


def apply_figure_style():
    """Apply consistent figure styling across all paper figures.

    Sets editable-text PDF/SVG output, seaborn theme (whitegrid, notebook
    context), and compact font sizes suitable for multi-panel figures.
    Call once at the top of every figure notebook.
    """
    # Emit editable text in PDF/SVG (not paths) so Illustrator can edit it
    matplotlib.rcParams["pdf.fonttype"] = 42  # TrueType fonts in PDF
    matplotlib.rcParams["ps.fonttype"] = 42  # TrueType fonts in PS/EPS
    matplotlib.rcParams["svg.fonttype"] = "none"  # raw text in SVG

    # Base seaborn theme – whitegrid with notebook-sized elements
    sns.set_theme(style="whitegrid", context="talk")

    # Override seaborn's "talk" context with compact font sizes
    sns.set_context(
        "notebook",
        rc={
            "font.size": FONTSIZE,
            "axes.titlesize": FONTSIZE_HEADING,
            "axes.labelsize": FONTSIZE_HEADING,
            "xtick.labelsize": FONTSIZE,
            "ytick.labelsize": FONTSIZE,
            "legend.fontsize": FONTSIZE,
            "legend.title_fontsize": FONTSIZE,
        },
    )


# Apply style on import so every notebook that loads figure_config gets it
apply_figure_style()

# Global model aliases
MODEL_LABELS = {
    "EGFR_MAPK__logobs": "DMM",
    "EGFR_MAPK__logobs_fegfr_aggavg": "DMM + EGFR est",
    "EGFR_MAPK__logobs_tegfr_aggavg": "DMM + EGFR tx",
    "EGFR_MAPK__logobs_pegfr_aggavg": "DMM + EGFR prot",
    "EGFR_MAPK__logobs_terbb2_aggavg": "DMM + ERBB2 tx",
    "EGFR_MAPK__logobs_perbb2_aggavg": "DMM + ERBB2 prot",
    "EGFR_MAPK__logobs_terbb3_aggavg": "DMM + ERBB3 tx",
    "EGFR_MAPK__logobs_perbb3_aggavg": "DMM + ERBB3 prot",
    "EGFR_MAPK__logobs_mbraf_mkras": "DMM + mutations",
    "EGFR_MAPK__logobs_tegfr_mbraf_mkras_aggavg": "DMM + EGFR tx + mutations",
    # Growth factors (figure4)
    "EGFR_MAPK__logobs_tegfr_ttgfa_tbtc_tereg_tnrg1_tnrg2_aggavg": "DMM + EGFR tx + all growth factors",
    "EGFR_MAPK__logobs_tegfr_tbtc_aggavg": "DMM + EGFR tx + BTC",
    "EGFR_MAPK__logobs_tegfr_tereg_aggavg": "DMM + EGFR tx + EREG",
    "EGFR_MAPK__logobs_tegfr_tnrg1_aggavg": "DMM + EGFR tx + NRG1",
    "EGFR_MAPK__logobs_tegfr_tnrg2_aggavg": "DMM + EGFR tx + NRG2",
    "EGFR_MAPK__logobs_tegfr_ttgfa_aggavg": "DMM + EGFR tx + TGFA",
    # Individual mutations (figure4)
    "EGFR_MAPK__logobs_tegfr_mbraf_aggavg": "DMM + EGFR tx + BRAF mut.",
    "EGFR_MAPK__logobs_tegfr_mkras_aggavg": "DMM + EGFR tx + KRAS mut.",
    "elasticnet": "linear regression",
}

# Model colors
# ── tab20 slots 8–14: transcript/protein pairs (dark/light) + estimated EGFR
# ── Set2 for remaining models (mutations, growth factors, linear regression)
MODEL_COLORS = {
    # EGFR pair:
    "DMM + EGFR tx": _tab20[12],
    "DMM + EGFR prot": _tab20[13],
    # ERBB2 pair:
    "DMM + ERBB2 tx": _tab20[16],
    "DMM + ERBB2 prot": _tab20[17],
    # ERBB3 pair:
    "DMM + ERBB3 tx": _tab20[18],
    "DMM + ERBB3 prot": _tab20[19],
    # Estimated EGFR:
    "DMM + EGFR est": _tab20[10],
    # Mutations – (fix label names: add "tx")
    "DMM + mutations": _set2[0],
    "DMM + EGFR tx + mutations": "#888888",  # gray    (combined)
    "DMM + EGFR tx + BRAF mut.": _tab10[3],  # red
    "DMM + EGFR tx + KRAS mut.": _tab10[5],  # brown
    # Growth factors – tab10 qualitative (fix label names: add "tx")
    "DMM + EGFR tx + all growth factors": "#555555",  # dark gray (combined)
    "DMM + EGFR tx + BTC": _tab10[0],  # blue
    "DMM + EGFR tx + EREG": _tab10[2],  # green
    "DMM + EGFR tx + NRG1": _tab10[4],  # purple
    "DMM + EGFR tx + NRG2": _tab10[9],  # cyan
    "DMM + EGFR tx + TGFA": _tab10[1],  # orange
    # Linear regression – black
    "linear regression": (0.0, 0.0, 0.0),
}

# Model groups for spacing in plots
MODEL_GROUPS = [
    [
        "DMM + EGFR est",
        "DMM + EGFR tx",
        "DMM + EGFR prot",
    ],
    ["DMM + ERBB2 tx", "DMM + ERBB2 prot"],
    ["DMM + ERBB3 tx", "DMM + ERBB3 prot"],
    ["DMM + mutations", "DMM + EGFR tx + mutations"],
    # Growth factors (figure4)
    [
        "DMM + EGFR tx + all growth factors",
        "DMM + EGFR tx + BTC",
        "DMM + EGFR tx + EREG",
        "DMM + EGFR tx + NRG1",
        "DMM + EGFR tx + NRG2",
        "DMM + EGFR tx + TGFA",
    ],
    # Individual mutations (figure4)
    ["DMM + EGFR tx + BRAF mut.", "DMM + EGFR tx + KRAS mut."],
    ["linear regression"],
]

# Context labels
CONTEXT_LABELS = {
    "cytof_init": "CyTOF",
    "transcriptomics": "RNAseq",
    "proteomics": "MassSpec",
    "multimodal": "Multimodal",
}

# Context colors: dark tab20 slots (even indices)
_CONTEXT_ORDER = ["CyTOF", "RNAseq", "MassSpec", "Multimodal", "MOSA"]
CONTEXT_COLORS = {ctx: _tab20[i * 2] for i, ctx in enumerate(_CONTEXT_ORDER)}

# Context colors (keyed by raw context code)
CONTEXT_COLORS_RAW = {k: CONTEXT_COLORS[v] for k, v in CONTEXT_LABELS.items()}

MODALITY_COLORS = {
    "transcript": CONTEXT_COLORS["RNAseq"],
    "protein": CONTEXT_COLORS["MassSpec"],
}

# Context labels for figure2 feature augmentations
CONTEXT_LABELS_2 = {
    "cytof_init_plus_tEGFR": "+ EGFR tx",
    "cytof_init_plus_pEGFR": "+ EGFR prot",
    "cytof_init_plus_lb": "+ L/B subt.",
    "cytof_init_plus_intr": "+ intr. subt.",
}

# Context labels for figure5b transcript/protein augmentations
CONTEXT_LABELS_5B = {
    f"cytof_init_plus_t{m}": f"+ t{m}" for m in EXTRA_MARKERS_5B_TX
} | {f"cytof_init_plus_p{m}": f"+ p{m}" for m in EXTRA_MARKERS_5B_PROT}

# CyTOF marker labels
CYTOF_MARKER_LABELS = {
    "p.ERBB2": "pERBB2 [a.u.]",
    "p.MEK": "pMEK [a.u.]",
    "p.ERK": "pERK [a.u.]",
    "p.p90RSK": r"pp90$^{rsk}$ [a.u.]",
}

# Observable → CyTOF marker mapping for consistent annotation
OBS_TO_CYTOF_MARKER = {
    "pERBB2_Y1248_obs": "p.ERBB2",
    "pERK_Y204_obs": "p.ERK",
    "pMEK_S222_obs": "p.MEK",
    "pRPS6KA1_S380_obs": "p.p90RSK",
}

# Observable display labels (harmonized with CYTOF_MARKER_LABELS)
OBS_LABELS = {
    obs: CYTOF_MARKER_LABELS.get(marker, obs)
    for obs, marker in OBS_TO_CYTOF_MARKER.items()
}

# Condition display labels
CONDITION_LABELS = {
    "EGF": "EGF",
    "iEGFR": "+iRTK",
    "iMEK": "+iMEK",
}

# Reference palette and labels for trajectory plots
REF_PALETTE = {
    "obs": "#2ca02c",  # green – experimental observations (obs)
    "sample": "#98df8a",  # light green – sample reference (not experimental)
    "DMM": CONTEXT_COLORS["CyTOF"],
    "avg_model": "#9467bd",  # purple – population average
    "linreg": "#ff7f0e",  # orange
    "lasso": "#d62728",  # red
    "elasticnet": MODEL_COLORS.get("linear regression", "#8c564b"),
}

REF_LABELS = {
    "obs": "Experimental (obs)",
    "sample": "Sample ref",
    "DMM": "DMM",
    "avg_model": "Avg. model",
    "linreg": "Linear reg.",
    "lasso": "Lasso",
    "elasticnet": MODEL_LABELS.get("elasticnet", "Elastic net"),
}

# Reference renaming for control baselines
REF_DISPLAY_MAP = {
    "avg_model": "negative control",
    "sample": "positive control",
}

# Shared styling for barplots
BARPLOT_CONTEXT_ORDER = ["RNAseq", "MassSpec", "CyTOF", "Multimodal", "MOSA"]
BARPLOT_REF_LINE_STYLES = {
    "negative control": "-.",
    "positive control": ":",
}
BARPLOT_REF_LINE_COLORS = {
    "negative control": "#555555",
    "positive control": "#000000",
}
BARPLOT_REF_LINE_WIDTH = 1.5
AXIS_SPINE_LINEWIDTH = 1
AXIS_SPINE_COLOR = "black"

# Default dark-center diverging colormap for binned plots
BINNED_CMAP = sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)

# Default context order
DEFAULT_CONTEXT_ORDER = [
    "CyTOF",
    "Transcriptomics",
    "Proteomics",
    "Multimodal",
    "MOSA",
]


def get_model_label(model: str) -> str:
    """Get display label for a model."""
    return MODEL_LABELS.get(model, model)


def get_context_label(context: str) -> str:
    """Get display label for a context."""
    return CONTEXT_LABELS.get(context, context)


def get_cytof_marker_label(marker: str) -> str:
    """Get display label for a CyTOF marker."""
    return CYTOF_MARKER_LABELS.get(marker, marker)


def configure_axis_spines(
    ax,
    top=False,
    right=False,
    left=True,
    bottom=True,
    linewidth=AXIS_SPINE_LINEWIDTH,
    color=AXIS_SPINE_COLOR,
):
    """
    Configure axis spines with consistent styling across figures.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to configure
    top : bool, optional
        Show top spine (default: False)
    right : bool, optional
        Show right spine (default: False)
    left : bool, optional
        Show left spine (default: True)
    bottom : bool, optional
        Show bottom spine (default: True)
    linewidth : float, optional
        Line width for visible spines (default: AXIS_SPINE_LINEWIDTH)
    color : str, optional
        Color for visible spines (default: AXIS_SPINE_COLOR)
    """
    ax.spines["top"].set_visible(top)
    ax.spines["right"].set_visible(right)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    ax.grid(False, which="both", axis="x")  # Disable vertical grid lines

    # Set linewidth and color for visible spines
    for spine_name in ["top", "right", "left", "bottom"]:
        spine = ax.spines[spine_name]
        if spine.get_visible():
            spine.set_linewidth(linewidth)
            spine.set_color(color)

    # Ensure tick marks are visible (seaborn whitegrid hides them)
    ax.tick_params(
        which="both",
        direction="out",
        length=4,
        width=linewidth,
        color=color,
        left=left,
        bottom=bottom,
        top=top,
        right=right,
    )


# ---------------------------------------------------------------------------
# Barplot colour helpers (tab20-based)
# ---------------------------------------------------------------------------


def get_box_color(model_label: str, context: str | None = None) -> tuple:
    """Return the colour for a boxplot box.

    * **DMM** (the base model) is coloured by *context* using the dark tab20
      palette.  If *context* is ``None``, falls back to the CyTOF colour.
    * Composite labels like ``"DMM \u00b7 CyTOF"`` are resolved automatically.
    * Every other model gets a fixed colour from ``MODEL_COLORS``.
    """
    # Handle composite labels produced by _expand_dmm_by_context
    if " \u00b7 " in model_label:
        base, ctx = model_label.split(" \u00b7 ", 1)
        if base == "DMM":
            return CONTEXT_COLORS.get(ctx, CONTEXT_COLORS["CyTOF"])
    if model_label == "DMM":
        if context is not None:
            return CONTEXT_COLORS.get(context, CONTEXT_COLORS["CyTOF"])
        return CONTEXT_COLORS["CyTOF"]
    return MODEL_COLORS.get(model_label, (0.5, 0.5, 0.5))


def build_barplot_legend(
    contexts: list[str],
    model_labels: list[str],
    *,
    include_refs: bool = True,
) -> tuple[list, list]:
    """Build combined legend handles for context + model colours.

    Returns ``(handles, labels)`` suitable for ``ax.legend(handles, labels)``.

    The legend is grouped:
      1. **Contexts** (dark tab20) – shown as filled squares.
      2. **Models** (light tab20, excl. DMM) – shown as filled squares.
      3. Optionally **reference lines** (negative / positive control).

    Parameters
    ----------
    contexts : list[str]
        Display-label context names that appear in the plot (e.g.
        ``["CyTOF", "Multimodal"]``).
    model_labels : list[str]
        Display-label model names in the plot **excluding** ``"DMM"``
        (which is already represented via contexts).
    include_refs : bool
        Whether to append negative/positive control legend entries.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles, labels = [], []

    # 1. Context colours (dark tab20)
    for ctx in contexts:
        color = CONTEXT_COLORS.get(ctx, (0.5, 0.5, 0.5))
        handles.append(
            Patch(facecolor=color, edgecolor="white", linewidth=0.5)
        )
        labels.append(ctx)

    # 2. Model colours (light tab20, no DMM)
    for ml in model_labels:
        if ml == "DMM":
            continue
        color = MODEL_COLORS.get(ml, (0.5, 0.5, 0.5))
        handles.append(
            Patch(facecolor=color, edgecolor="white", linewidth=0.5)
        )
        labels.append(ml)

    # 3. Reference lines
    if include_refs:
        for ref_label, ls in BARPLOT_REF_LINE_STYLES.items():
            color = BARPLOT_REF_LINE_COLORS.get(ref_label, "gray")
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=color,
                    linestyle=ls,
                    linewidth=BARPLOT_REF_LINE_WIDTH,
                )
            )
            labels.append(ref_label)

    return handles, labels
