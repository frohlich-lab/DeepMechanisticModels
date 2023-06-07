from collections import namedtuple
from pathlib import Path
from typing import List

import numpy as np

from cytof import get_samples

MODEL_FEATURE_PREFIX = "INPUT_"

Wildcards = namedtuple("Wildcards", ["data", "samples"])

basedir: Path = Path(__file__).resolve().parent
fig_dir = basedir / "figures"
results_dir = basedir / "results"
data_dir = basedir / "data"
pretrain_dir = basedir / "pretraining"

PER_SAMPLE_OUTFILE_PARS = str(
    pretrain_dir / "{model}" / "{data}" / "{sample}.csv"
)
PER_SAMPLE_OUTFILE_RESULTS = str(
    pretrain_dir / "{model}" / "{data}" / "{sample}.hdf"
)

defaults = {
    x: f"{{{x}}}" for x in ["context", "samples", "n_hidden", "alpha", "job"]
}
tpl_results_file = "__".join(defaults.values())
CROSS_SAMPLE_OUTFILE_PARS = str(
    pretrain_dir / "{model}" / "{data}" / (tpl_results_file + ".csv")
)
CROSS_SAMPLE_OUTFILE_RESULTS = str(
    pretrain_dir / "{model}" / "{data}" / (tpl_results_file + ".hdf5")
)
CROSS_SAMPLE_OUTFILE_TRACE = str(
    pretrain_dir / "{model}" / "{data}" / (tpl_results_file + "_trace.hdf5")
)

TRAINING_OUTFILE_TRACE = str(
    results_dir / "{model}" / "{data}" / (tpl_results_file + "_trace.hdf5")
)
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
CONDITIONS_FILE = tpl_petab_file.format(
    file="conditions", data="{data}", model="{model}"
)
OBSERVABLES_FILE = tpl_petab_file.format(
    file="observables", data="{data}", model="{model}"
)

EVALUATION_REFERENCE = str(
    fig_dir / "{model}" / "{data}" / "{samples}_pretrain_{mode}_{dataset}.csv"
)
EVALUATION_PRETRAINING = str(
    fig_dir
    / "{model}"
    / "{data}"
    / "{samples}_pretrain_{mode}_{dataset}_{alpha}_{n_hidden}_{context}.csv"
)
EVALUATION_TRAINING = str(
    fig_dir
    / "{model}"
    / "{data}"
    / "{samples}_training_{dataset}_{alpha}_{n_hidden}_{context}.csv"
)
EVALUATE_ALL = str(fig_dir / "{model}" / "{data}" / "evaluate_all_{group}.pdf")


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


def select_values(data, percent=0.1):
    # Convert the generator to a list
    data_list = list(data)

    # Generate log-spaced indices
    num_values = len(data_list)
    num_selected = int(num_values * percent)
    indices = set(
        np.logspace(
            0,
            np.log10(num_values - 1),
            num=num_selected,
            endpoint=True,
            base=10,
            dtype=int,
        )
    )

    # Select values based on the indices
    selected_values = [data_list[i] for i in indices]

    return selected_values
