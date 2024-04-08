import numpy as np
import dataclasses

from collections import namedtuple
from cytof import get_samples
from pathlib import Path
from training_configuration import CONTEXTS_FEATURES
from typing import List


@dataclasses.dataclass
class Conf(dict):
    model: str
    data: str
    context: str = None
    features: str = None
    samples: str = None
    sample: str = None
    n_hidden: int = None
    encoder_layer_sizes: List[int]
    encoder_layer_biases: List[bool]
    inflater_layer_sizes: List[int]
    inflater_layer_biases: List[bool]
    decoder_layer_biases: List[bool]
    activation_fn_name: str
    reconstruct: bool
    nn_pretrain: bool  # dictates when KinParams_Combiner params are switched to non-zero, learnable values
    orth_reg_strategy: str = None  # values: "L1" / "L2"
    l1reg_inflate: float = 0.0
    oreg_inflate: float = 0.0
    l1reg_encode: float = 0.0
    oreg_encode: float = 0.0
    job: int = None
    threads: int = 1
    n_starts: int = None

CONTEXT_SET = set([context for context, _ in CONTEXTS_FEATURES])

MODEL_FEATURE_PREFIX = "INPUT_"

Wildcards = namedtuple("Wildcards", ["data", "samples"])

basedir: Path = Path(__file__).resolve().parent
fig_dir = basedir / "figures"
evaluations_dir = basedir / "evaluations"
results_dir = basedir / "results"
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


defaults = {
    x: f"{{{x}}}"
    for x in [
        "context",
        "features",
        "samples",
        # "pretrain",
        "n_hidden",
        "orth_reg_strategy",
        "l1reg_inflate",
        "oreg_inflate",
        "l1reg_encode",
        "oreg_encode",
        "job",
    ]
}
tpl_results_file = "__".join(defaults.values())


TRAINING_OUTFILE_RESULTS = str(
    results_dir / "{model}" / "{data}" / (tpl_results_file + ".hdf5")
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

defaults = {
    x: f"{{{x}}}"
    for x in [
        "context",
        "samples",
        "n_hidden",
        "job", # need job field in EVALUATION_TRAINING
        "features",
        "orth_reg_strategy",
        "l1reg_inflate",
        "oreg_inflate",
        "l1reg_encode",
        "oreg_encode",
    ]
}
tpl_evaluation_file = "__".join(defaults.values())
EVALUATION_TRAINING = str(
    evaluations_dir
    / "{model}"
    / "{data}"
    / "training"
    / "{dataset}"
    / (tpl_evaluation_file + ".csv")
)
EVALUATE_ALL = str(fig_dir / "{model}" / "{data}" / "evaluate_all_{group}.pdf")
EVALUATE_ALL_CSVS = str(evaluations_dir / "{model}" / "{data}" / "{filename}.pdf")


def training_samples(wildcards) -> List[str]:
    samples = get_samples(wildcards.data)
    split, n_splits = wildcards.samples.split("_")
    splits = np.array_split(np.asarray(samples), int(n_splits))
    return list(
        np.concatenate([s for i, s in enumerate(splits) if i != int(split)])
    )


def test_samples(wildcards) -> List[str]:
    samples = get_samples(wildcards.data)
    split, n_splits = wildcards.samples.split("_")
    splits = np.array_split(np.asarray(samples), int(n_splits))
    return list(splits[int(split)])


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


def select_values(data, num_selected: int):
    # Convert the generator to a list
    data_list = list(data)

    # Generate log-spaced indices
    num_values = len(data_list)

    if num_values <= 1:
        return data_list

    indices = set(
        np.logspace(
            0,
            np.log10(num_values - 1),
            num=min(num_selected, num_values),
            endpoint=True,
            base=10,
            dtype=int,
        )
    )

    # Select values based on the indices
    selected_values = [data_list[i] for i in indices]

    return selected_values
