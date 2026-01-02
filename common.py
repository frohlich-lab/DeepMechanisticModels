import os
import re
from collections import namedtuple
from pathlib import Path

import numpy as np

from cytof import get_samples
from dmm.config_options import scan_attributes


# moved from Snakefile
class SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


# Get the DEBUG environment variable
debug_mode = os.getenv("DEBUG", "false").lower() in ["true", "1", "yes"]

MODEL_FEATURE_PREFIX = "INPUT_"

Wildcards = namedtuple("Wildcards", ["data", "samples"])

basedir: Path = Path(__file__).resolve().parent
fig_dir = basedir / "figures"
evaluations_dir = basedir / "eval"
results_dir = basedir / "res"
data_dir = basedir / "data"
pretrain_dir = basedir / "pretrain"
features_dir = basedir / "features"

PER_SAMPLE_OUTFILE_PARS = str(
    pretrain_dir / "{model}" / "{data}" / "{sample}.csv"
)
PER_SAMPLE_OUTFILE_RESULTS = str(
    pretrain_dir / "{model}" / "{data}" / "{sample}.hdf"
)

FEATURES_OUTFILE = str(
    features_dir
    / "{model}"
    / "{data}"
    / "{dataset}"
    / (
        "__".join(
            {
                x: f"{{{x}}}" for x in ["context", "samples", "features"]
            }.values()
        )
        + ".csv"
    )
)

FEATURES_PIPELINE = str(
    features_dir
    / "{model}"
    / "{data}"
    / "{context}__{samples}__{features}__trained_pca_pipeline.joblib"
)

defaults = {
    x: f"{{{x}}}" for x in scan_attributes if x not in ["model", "data"]
}

tpl_results_file = "__".join(defaults.values())

TRAINING_OUTFILE_RESULTS = str(
    results_dir / "{model}" / "{data}" / (tpl_results_file + ".hdf5")
)

PRETRAINED_BEST_MODELS = str(
    results_dir / "{model}" / "{data}" / (tpl_results_file + "_nn_pre_bm.eqx")
)

# TODO @GiacomoFabrini check this works and replace how this is handled everywhere!
TRAINED_MODEL = str(
    results_dir / "{model}" / "{data}" / (tpl_results_file + ".eqx")
)

TRAINED_MODEL_WEIGHT_PLOTS = str(
    results_dir
    / "{model}"
    / "{data}"
    / (tpl_results_file + "_weight_plot.png")
)

COLLECTED_TRAINING_RESULTS = str(
    results_dir
    / "{model}"
    / "{data}"
    / (tpl_results_file.format(**{**defaults, "job": "full"}) + ".hdf5")
)

tpl_petab_file = str(data_dir / "{model}__{data}__{file}.tsv")
MEASUREMENTS_FILE = tpl_petab_file.format(
    file="measurements", data="{data}", model="{model}"
)
MEASUREMENTS_FILE_RW = MEASUREMENTS_FILE.replace(".tsv", "_rw_{samples}.tsv")
CONDITIONS_FILE = tpl_petab_file.format(
    file="conditions", data="{data}", model="{model}"
)
OBSERVABLES_FILE = tpl_petab_file.format(
    file="observables", data="{data}", model="{model}"
)

EVALUATION_REFERENCE = str(
    evaluations_dir
    / "{model}"
    / "{data}"
    / "references"
    / "{samples}_{mode}_{dataset}.csv"
)

# Regressor template and files
tpl_regressor = str(
    evaluations_dir
    / "{model}"
    / "{data}"
    / "regressors"
    / (
        "__".join(
            f"{{{x}}}"
            for x in [
                "context",
                "samples",
                "mode",
                "features",
            ]
        )
    )
)

EVALUATION_REGRESSOR = tpl_regressor + "__{dataset}.csv"

# using same defaults as above
tpl_evaluation_file = "__".join(defaults.values())

EVALUATION_TRAINING = str(
    evaluations_dir
    / "{model}"
    / "{data}"
    / "training"
    / "{dataset}"
    / (tpl_evaluation_file + ".csv")
)

EVALUATION_EMBEDDING = str(
    evaluations_dir
    / "{model}"
    / "{data}"
    / "embeddings"
    / "{dataset}"
    / (tpl_evaluation_file + ".csv")
)

EVALUATION_FULL_PARAMETERS = str(
    evaluations_dir
    / "{model}"
    / "{data}"
    / "trained_parameters"
    / "{dataset}"
    / (tpl_evaluation_file + ".csv")
)

EVALUATION_PARAMETER_DEVIATIONS = str(
    evaluations_dir
    / "{model}"
    / "{data}"
    / "trained_param_dev"
    / "{dataset}"
    / (tpl_evaluation_file + ".csv")
)

EVALUATION_PLOT_FILE = "{dataset}__" + tpl_evaluation_file
EVALUATE_ALL = str(fig_dir / "{model}" / "{data}" / "evaluate_all_{group}.pdf")
EVALUATE_ALL_CSVS = str(
    evaluations_dir / "{model}" / "{data}" / "{filename}.csv"
)

hardest_cell_lines = ["cBT20", "cHCC1500", "cHCC2185", "cMCF7", "cUACC3199"]
test_samples = ["cCAL120", "cCAMA1", "cHCC1143", "cKPL1", "cZR75B"]


def training_samples(wildcards) -> list[str]:
    samples = get_samples(wildcards.data)
    # filter out samples with missing transcriptomics data
    if "plus_missingtx" not in wildcards.samples:
        missing_transcriptomics = ["cHCC2157", "cMCF10F", "cMDAkb2"]
        samples = [
            sample
            for sample in samples
            if sample not in missing_transcriptomics
        ]
    # filter out test samples
    samples = [sample for sample in samples if sample not in test_samples]
    # filter out validation samples
    samples = [
        sample for sample in samples if sample not in val_samples(wildcards)
    ]
    # subsample if specified
    if m := re.search(r"_(\d+)pct", wildcards.samples):
        n_samples = int(len(samples) * int(m.group(1)) / 100)
        np.random.seed(0)
        samples = np.random.choice(samples, n_samples, replace=False).tolist()
    return sorted(samples)


def val_samples(wildcards) -> list[str]:
    return (
        [f"c{wildcards.samples}"]
        if not wildcards.samples.startswith("all")
        else sorted(test_samples)
    )


def per_sample_pretraining_train(wildcards) -> list[str]:
    return [
        PER_SAMPLE_OUTFILE_PARS.format(
            sample=sample, model=wildcards.model, data=wildcards.data
        )
        for sample in training_samples(wildcards)
    ]


def per_sample_pretraining_test(wildcards) -> list[str]:
    return [
        PER_SAMPLE_OUTFILE_PARS.format(
            sample=sample, model=wildcards.model, data=wildcards.data
        )
        for sample in val_samples(wildcards)
    ]


# Tognetti et al. PAM 50 & luminal/basal, Figure 2, manually extracted
subtypes_tognetti = {
    "c184A1": {"PAM50": "Normal", "Luminal/Basal": "Normal"},
    "cBT20": {"PAM50": "Basal", "Luminal/Basal": "Basal"},
    "cBT474": {"PAM50": "LB", "Luminal/Basal": "Luminal"},
    "cBT549": {"PAM50": "Basal", "Luminal/Basal": "Basal"},
    "cCAL148": {"PAM50": "HER2", "Luminal/Basal": "Luminal"},
    "cCAL51": {"PAM50": "Basal", "Luminal/Basal": "Basal"},
    "cCAL851": {"PAM50": "Basal", "Luminal/Basal": "Basal"},
    "cDU4475": {"PAM50": "Other", "Luminal/Basal": "Basal"},
    "cEFM192A": {"PAM50": "HER2", "Luminal/Basal": "Luminal"},
    "cEVSAT": {"PAM50": "HER2", "Luminal/Basal": "Luminal"},
    "cHBL100": {"PAM50": "Basal", "Luminal/Basal": "Basal"},
    "cHCC1187": {"PAM50": "Basal", "Luminal/Basal": "Basal"},
    "cHCC1395": {"PAM50": "Basal", "Luminal/Basal": "Basal"},
    "cHCC1419": {"PAM50": "LB", "Luminal/Basal": "Luminal"},
    "cHCC1500": {"PAM50": "LB", "Luminal/Basal": "Luminal"},
    "cHCC1569": {"PAM50": "Basal", "Luminal/Basal": "Basal"},
    "cHCC1937": {"PAM50": "Basal", "Luminal/Basal": "Basal"},
    "cHCC1954": {"PAM50": "LB", "Luminal/Basal": "Basal"},
    "cHCC2157": {"PAM50": "Other", "Luminal/Basal": "Basal"},
    "cHCC2185": {"PAM50": "LA", "Luminal/Basal": "Luminal"},
    "cHCC3153": {"PAM50": "Basal", "Luminal/Basal": "Basal"},
    "cHCC38": {"PAM50": "Basal", "Luminal/Basal": "Basal"},
    "cHCC70": {"PAM50": "Basal", "Luminal/Basal": "Basal"},
    "cHDQP1": {"PAM50": "Normal", "Luminal/Basal": "Basal"},
    "cJIMT1": {"PAM50": "Basal", "Luminal/Basal": "Basal"},
    "cMCF10A": {"PAM50": "Normal", "Luminal/Basal": "Normal"},
    "cMCF10F": {"PAM50": "Other", "Luminal/Basal": "Normal"},
    "cMCF7": {"PAM50": "LB", "Luminal/Basal": "Luminal"},
    "cMDAMB134VI": {"PAM50": "LA", "Luminal/Basal": "Luminal"},
    "cMDAMB157": {"PAM50": "Basal", "Luminal/Basal": "Basal"},
    "cMDAMB175VII": {"PAM50": "LB", "Luminal/Basal": "Luminal"},
    "cMDAMB361": {"PAM50": "HER2", "Luminal/Basal": "Luminal"},
    "cMDAMB415": {"PAM50": "LB", "Luminal/Basal": "Luminal"},
    "cMDAMB453": {"PAM50": "HER2", "Luminal/Basal": "Luminal"},
    "cMDAkb2": {"PAM50": "Other", "Luminal/Basal": "Luminal"},
    "cMFM223": {"PAM50": "LA", "Luminal/Basal": "Luminal"},
    "cMPE600": {"PAM50": "LA", "Luminal/Basal": "Luminal"},
    "cMX1": {"PAM50": "Basal", "Luminal/Basal": "Basal"},
    "cOCUBM": {"PAM50": "HER2", "Luminal/Basal": "Luminal"},
    "cT47D": {"PAM50": "LA", "Luminal/Basal": "Luminal"},
    "cUACC812": {"PAM50": "LB", "Luminal/Basal": "Luminal"},
    "cUACC893": {"PAM50": "HER2", "Luminal/Basal": "Luminal"},
    "cZR7530": {"PAM50": "LB", "Luminal/Basal": "Luminal"},
    # added PAM50 and luminal/basal annotations for subchallenge II cell-lines
    "c184B5": {"PAM50": "Normal", "Luminal/Basal": "Normal"},
    "cBT483": {"PAM50": "LB", "Luminal/Basal": "Luminal"},
    "cHCC1428": {"PAM50": "LA", "Luminal/Basal": "Luminal"},
    "cHCC1806": {"PAM50": "Basal", "Luminal/Basal": "Basal"},
    "cHCC202": {"PAM50": "HER2", "Luminal/Basal": "Luminal"},
    "cHs578T": {"PAM50": "Basal", "Luminal/Basal": "Basal"},
    "cMCF12A": {"PAM50": "Normal", "Luminal/Basal": "Normal"},
    "cMDAMB231": {"PAM50": "Basal", "Luminal/Basal": "Basal"},
    "cMDAMB468": {"PAM50": "Basal", "Luminal/Basal": "Basal"},
    "cSKBR3": {"PAM50": "HER2", "Luminal/Basal": "Luminal"},
    "cUACC3199": {"PAM50": "Basal", "Luminal/Basal": "Basal"},
    "cZR751": {"PAM50": "LB", "Luminal/Basal": "Luminal"},
}

# From https://bmcmedgenomics.biomedcentral.com/articles/10.1186/1755-8794-5-44, Figure 1
pam50_genelist = [
    "FGFR4",
    "ERBB2",
    "GRB7",
    "BLVRA",
    "BAG1",
    "BCL2",
    "CXXC5",
    "ESR1",
    "GPR160",
    "FOXA1",
    "MLPH",
    "NAT1",
    "SLC39A6",
    "MAPT",
    "PGR",
    "MDM2",
    "TMEM45B",
    "MMP11",
    "ACTR3B",
    "CDC6",
    "CCNE1",
    "EXO1",
    "CDCA1",
    "KNTC2",
    "BIRC5",
    "CENPF",
    "ANLN",
    "CDC20",
    "CCNB1",
    "CEP55",
    "MYBL2",
    "MKI67",
    "UBE2C",
    "RRM2",
    "KIF2C",
    "MELK",
    "TYMS",
    "PTTG1",
    "ORC6L",
    "UBE2T",
    "CDH3",
    "EGFR",
    "KRT17",
    "KRT14",
    "KRT5",
    "FOXC1",
    "MIA",
    "SFRP1",
    "PHGDH",
    "MYC",
]

REGRESSION_MODES = ["linreg", "lasso", "elasticnet"]
