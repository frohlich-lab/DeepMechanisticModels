import numpy as np
import os

from collections import namedtuple
from cytof import get_samples
from dmm.config_options import default_attributes
from pathlib import Path
from training_configuration import CONTEXTS_FEATURES
from typing import List


# moved from Snakefile
class SafeDict(dict):
    def __missing__(self, key):
        return '{' + key + '}'


# Get the DEBUG environment variable
debug_mode = os.getenv('DEBUG', 'false').lower() in ['true', '1', 'yes']

CONTEXT_SET = sorted(list(set([context for context, _ in CONTEXTS_FEATURES])))

MODEL_FEATURE_PREFIX = "INPUT_"

Wildcards = namedtuple("Wildcards", ["data", "samples"])

basedir: Path = Path(__file__).resolve().parent
fig_dir = basedir / "figures"
evaluations_dir = basedir / "eval_2908_linscan_nowandb"  # TODO @GiacomoFabrini rename to `evaluations`
results_dir = basedir / "res_2908_linscan_nowandb"  # TODO @GiacomoFabrini rename to `results`
data_dir = basedir / "data"
pretrain_dir = basedir / "pretraining"
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
    / "{samples}_{features}_{context}_trained_pca_pipeline.joblib"
)

defaults = {
    x: f"{{{x}}}"
    for x in default_attributes
}

tpl_results_file = "__".join(defaults.values())

TRAINING_OUTFILE_RESULTS = str(
    results_dir / "{model}" / "{data}" / (tpl_results_file + ".hdf5")
)

PRETRAINED_BEST_MODELS = str(
    results_dir / "{model}" / "{data}" / (tpl_results_file + "_nn_pre_bm.eqx")
)

# TODO @GiacomoFabrini check this works and replace how this is handled everywhere!
TRAINED_BEST_MODELS = str(
    results_dir / "{model}" / "{data}" / (tpl_results_file + "_bm_{ensemble_id}.eqx")
)

TRAINED_MODEL_WEIGHT_PLOTS = str(
    results_dir / "{model}" / "{data}" / (tpl_results_file + "_weight_plot.png")
)

COLLECTED_TRAINING_RESULTS = str(
    results_dir
    / "{model}"
    / "{data}"
    / (tpl_results_file.format(**{**defaults, "job": "full"}) + ".hdf5")
)

tpl_petab_file = str(data_dir / "{model}_{data}_{file}.tsv")
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
    evaluations_dir / "{model}" / "{data}" / "{samples}_{mode}_{dataset}.csv"
)

EVALUATION_REGRESSOR = str(
    evaluations_dir
    / "{model}"
    / "{data}"
    / "{samples}_{mode}_{context}_{dataset}.csv"
)

REGR_TRAINED_PIPELINE = str(
    evaluations_dir
    / "{model}"
    / "{data}"
    / "{samples}_{mode}_{context}_trained_pipeline.joblib"
)

REGR_FEATURES_TRAIN = str(
    evaluations_dir
    / "{model}"
    / "{data}"
    / "{samples}_{mode}_{context}_features_train.joblib"
)

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

EVALUATION_PLOT_FILE = "{dataset}__" + tpl_evaluation_file
EVALUATE_ALL = str(fig_dir / "{model}" / "{data}" / "evaluate_all_{group}.pdf")
EVALUATE_ALL_CSVS = str(evaluations_dir / "{model}" / "{data}" / "{filename}.pdf")


def training_samples(wildcards, mode: str = "leave_one_out") -> List[str]:
    samples = get_samples(wildcards.data)
    split, n_splits = wildcards.samples.split("of")
    if mode != "leave_one_out":
        splits = np.array_split(np.asarray(samples), int(n_splits))
        return list(
            np.concatenate([s for i, s in enumerate(splits) if i != int(split)])
        )
    else:
        hardest_samples = ['cMCF7', 'cBT20', 'cHCC1500', 'cEVSAT', 'cHCC2185']
        return [sample for sample in samples if sample != hardest_samples[int(split)]]


def test_samples(wildcards, mode: str = "leave_one_out") -> List[str]:
    samples = get_samples(wildcards.data)
    split, n_splits = wildcards.samples.split("of")
    if mode != "leave_one_out":
        splits = np.array_split(np.asarray(samples), int(n_splits))
        return list(splits[int(split)])
    else:
        hardest_samples = ['cMCF7', 'cBT20', 'cHCC1500', 'cEVSAT', 'cHCC2185']
        return [hardest_samples[int(split)]]


def per_sample_pretraining_train(wildcards) -> List[str]:
    return [
        PER_SAMPLE_OUTFILE_PARS.format(
            sample=sample, model=wildcards.model, data=wildcards.data
        )
        for sample in training_samples(wildcards)
    ]


def per_sample_pretraining_test(wildcards) -> List[str]:
    return [
        PER_SAMPLE_OUTFILE_PARS.format(
            sample=sample, model=wildcards.model, data=wildcards.data
        )
        for sample in test_samples(wildcards)
    ]

# Does not appear to be used?!
# def select_values(data, num_selected: int):
#     # Convert the generator to a list
#     data_list = list(data)
#
#     # Generate log-spaced indices
#     num_values = len(data_list)
#
#     if num_values <= 1:
#         return data_list
#
#     indices = set(
#         np.logspace(
#             0,
#             np.log10(num_values - 1),
#             num=min(num_selected, num_values),
#             endpoint=True,
#             base=10,
#             dtype=int,
#         )
#     )
#
#     # Select values based on the indices
#     selected_values = [data_list[i] for i in indices]
#
#     return selected_values
