# Shared configuration for figure generation
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")

# Global model aliases
MODEL_LABELS = {
    "EGFR_MAPK__logobs": "DMM",
    "EGFR_MAPK__logobs_fegfr_aggavg": "DMM + estimated EGFR levels",
    "EGFR_MAPK__logobs_tegfr_aggavg": "DMM + EGFR transcript levels",
    "EGFR_MAPK__logobs_pegfr_aggavg": "DMM + EGFR protein levels",
    "EGFR_MAPK__logobs_terbb2_aggavg": "DMM + ERBB2 transcript levels",
    "EGFR_MAPK__logobs_perbb2_aggavg": "DMM + ERBB2 protein levels",
    "EGFR_MAPK__logobs_terbb3_aggavg": "DMM + ERBB3 transcript levels",
    "EGFR_MAPK__logobs_perbb3_aggavg": "DMM + ERBB3 protein levels",
    "EGFR_MAPK__logobs_mbraf_mkras": "DMM + mutations",
    "EGFR_MAPK__logobs_tegfr_mbraf_mkras_aggavg": "DMM + EGFR transcripts + mutations",
    "elasticnet": "linear regression",
}

# Color palette for matching transcripts/proteins
MODEL_COLORS = {
    "DMM": "#7f7f7f",  # gray
    "DMM + estimated EGFR levels": "#8c564b",  # brown
    "DMM + EGFR transcript levels": "#1f77b4",  # blue
    "DMM + EGFR protein levels": "#aec7e8",  # light blue
    "DMM + ERBB2 transcript levels": "#ff7f0e",  # orange
    "DMM + ERBB2 protein levels": "#ffbb78",  # light orange
    "DMM + ERBB3 transcript levels": "#2ca02c",  # green
    "DMM + ERBB3 protein levels": "#98df8a",  # light green
    "DMM + mutations": "#d62728",  # red
    "DMM + EGFR transcripts + mutations": "#9467bd",  # purple
    "linear regression": "#000000",  # cyan
}

# Model groups for spacing in plots
MODEL_GROUPS = [
    ["DMM"],
    [
        "DMM + estimated EGFR levels",
        "DMM + EGFR transcript levels",
        "DMM + EGFR protein levels",
    ],
    ["DMM + ERBB2 transcript levels", "DMM + ERBB2 protein levels"],
    ["DMM + ERBB3 transcript levels", "DMM + ERBB3 protein levels"],
    ["DMM + mutations", "DMM + EGFR transcripts + mutations"],
    ["linear regression"],
]

# Context labels
CONTEXT_LABELS = {
    "cytof_init": "CyTOF",
    "transcriptomics": "RNAseq",
    "proteomics": "MassSpec",
    "multimodal": "Multimodal",
}

# CyTOF marker labels
CYTOF_MARKER_LABELS = {
    "p.HER2": "phospho ERBB2",
    "p.MEK": "phospho MEK",
    "p.ERK": "phospho ERK",
    "p.p90RSK": r"phospho p90$^{rsk}$",
}

# Observable → CyTOF marker mapping for consistent annotation
OBS_TO_CYTOF_MARKER = {
    "pERBB2_Y1248_obs": "p.HER2",
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
    "iEGFR": "EGF + iEGFR",
    "iMEK": "EGF + iMEK",
}

# Reference palette and labels for trajectory plots
REF_PALETTE = {
    "obs": "#2ca02c",  # green – experimental observations (obs)
    "sample": "#98df8a",  # light green – sample reference (not experimental)
    "DMM": MODEL_COLORS.get("DMM", "#1f77b4"),
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
BARPLOT_CONTEXT_ORDER = ["RNAseq", "MassSpec", "CyTOF", "Multimodal"]
BARPLOT_REF_LINE_STYLES = {
    "negative control": "--",
    "positive control": ":",
}
BARPLOT_REF_LINE_COLORS = {
    "negative control": "#555555",
    "positive control": "#000000",
}
AXIS_SPINE_LINEWIDTH = 2
AXIS_SPINE_COLOR = "black"

# Default context order
DEFAULT_CONTEXT_ORDER = [
    "CyTOF",
    "Transcriptomics",
    "Proteomics",
    "Multimodal",
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

    # Set linewidth and color for visible spines
    for spine_name in ["top", "right", "left", "bottom"]:
        spine = ax.spines[spine_name]
        if spine.get_visible():
            spine.set_linewidth(linewidth)
            spine.set_color(color)
