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
