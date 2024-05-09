import dataclasses
import numpy as np
import optax

from collections import namedtuple
from cytof import get_samples
from optax import adam, adamw
from pathlib import Path
from training_configuration import CONTEXTS_FEATURES
from typing import Dict, List, Optional


# define abbreviations/labels for logging of loss terms
L1EREG = "l1reg_encode"
OEREG = "oreg_encode"
L1IREG = "l1reg_inflate"
OIREG = "oreg_inflate"
RECON_LOSS = "recon_loss"
SYMM_LOSS = "symm_reg"

# Optimisers to choose from
optimisers = {
    "adam": adam,
    "adamw": adamw,
}


@dataclasses.dataclass(repr=True)
class Conf(dict):
    model: str
    data: str
    context: str = None
    features: str = None
    samples: str = None
    sample: str = None
    n_hidden: int = None
    encoder_layer_sizes: List[int] = None
    encoder_layer_biases: List[bool] = None
    encoder_weight_init_fn: str = "eqx_default"
    encoder_bias_init_fn: str = "eqx_default"
    inflater_layer_sizes: List[int] = None
    inflater_layer_biases: List[bool] = None
    inflater_weight_init_fn: str = "eqx_default"
    inflater_bias_init_fn: str = "eqx_default"
    decoder_layer_biases: List[bool] = None
    decoder_weight_init_fn: str = "eqx_default"
    decoder_bias_init_fn: str = "eqx_default"
    activation_fn_name: str = "relu"
    optimiser: str = "adam"
    reconstruct: bool = None
    orth_reg_strategy: str = None
    l1reg_encode: float = 0.0
    oreg_encode: float = 0.0
    l1reg_inflate: float = 0.0
    oreg_inflate: float = 0.0
    recon_loss: float = 0.0
    symm_reg: float = 0.0
    job: int = None
    threads: int = 1
    n_starts: int = None
    linear_benchmark: str = None
    use_early_stopping: bool = True
    use_simple_linear_schedule: bool = True
    max_lrate: float = 0.01  # absolute maximum learning rate (max in first schedule or in all schedules without decay)
    lrate_span: float = 1e0  # ratio between max and min learning rates in a given schedule
    lrate_decay: float = 0.98  # if < 1, the learning rate decays between schedules.
    # # 0.98 will reduce 1e-2 to 1e-3 in 100 epochs, similarly to our original linear schedule
    warmup_fct: float = 0.0  # fraction of schedule epochs to be used for warmup
    opt_steps: int = 10  # Number of steps in the first schedule (they multiply each time in length by opt_mult)
    opt_mult: int = 2  # Multiplier for the number of steps in each schedule

    def __str__(
            self,
            replace: Optional[Dict[str, str]] = None,
    ):
        """
        Return string representation with selected hyperparameters.
        """
        # Generate a dictionary of field names and values
        field_dict = {field.name: getattr(self, field.name) for field in dataclasses.fields(self)}

        # Update the field_dict with any replacements specified
        if replace is not None:
            field_dict.update(replace)

        # Filter out unwanted fields from the final string representation
        unwanted_fields = [
            "model", "data", "sample", "context", "features",
            "encoder_layer_biases", "inflater_layer_biases", "decoder_layer_biases",
            "encoder_output_bias", "inflater_output_bias", "decoder_output_bias",
            "threads", "n_starts", "linear_benchmark", "use_early_stopping"
        ]

        # Avoid including in run name weight and bias initialisation strategies if default
        for init_strategy, label in zip(
                [self.encoder_weight_init_fn, self.encoder_bias_init_fn,
                 self.inflater_weight_init_fn, self.inflater_bias_init_fn,
                 self.decoder_weight_init_fn, self.decoder_bias_init_fn],
                ["encoder_weight_init_fn", "encoder_bias_init_fn",
                 "inflater_weight_init_fn", "inflater_bias_init_fn",
                 "decoder_weight_init_fn", "decoder_bias_init_fn"]
        ):
            if init_strategy == "eqx_default":
                unwanted_fields += [label]

        # Create a list of values for the fields that are not in the unwanted list
        filtered_values = [
            field_dict[field] for field in field_dict if field not in unwanted_fields
        ]

        # Return the joined string of the filtered values
        return '__'.join(map(str, filtered_values))


@dataclasses.dataclass
class EarlyStoppingParams(dict):
    use_early_stopping: bool = True
    patience: int = 9
    min_improvement: float = 0


@dataclasses.dataclass
class ModuleParams(dict):
    layer_sizes: List[int]
    layer_biases: Optional[List[bool]] = None  # no learnable bias
    weight_init_fn: str = "eqx_default"  # eqx.nn.Linear layers
    bias_init_fn: str = "eqx_default"  # eqx.nn.Linear layers


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


def get_scheduler(
        conf: Dict,
        n_epoch: int,
) -> optax.Schedule:
    """Get the learning rate scheduler.

    Parameters
    ----------
    conf : configuration object
    n_epoch : int - total number of training epochs

    Returns
    ----------
    optax.sgdr_schedule
        The learning rate scheduler.
    """
    if conf["use_simple_linear_schedule"]:
        # Define custom steps to use the same machinery as below - schedule config should
        # be entirely within conf object
        schedules = [
            {
                'init_value': conf["max_lrate"] / conf["lrate_span"],  # before warm-up
                'peak_value': conf["max_lrate"],  # after warm-up
                'warmup_steps': int(n_epoch * conf["warmup_fct"]),
                'decay_steps': n_epoch,  # entire n_epoch
                'end_value': conf["max_lrate"] * conf["lrate_decay"]**n_epoch, # after decay
            }  # single linear schedule
        ]
    else:
        epochs_per_schedule = np.array([
            conf["opt_steps"] * (conf["opt_mult"] ** i)
            for i in range(int(n_epoch // conf["opt_steps"]))
            if conf["opt_steps"] * (conf["opt_mult"] ** i) <= n_epoch
        ])
        schedules = [
            {
                'init_value': conf["max_lrate"] / conf["lrate_span"] * conf["lrate_decay"] ** i_schedule,
                'peak_value': conf["max_lrate"] * conf["lrate_decay"] ** i_schedule,
                'warmup_steps': int(
                    (conf["opt_steps"] * (conf["opt_mult"] ** i_schedule))
                    * conf["warmup_fct"]
                ),
                'decay_steps': int(conf["opt_steps"]  * (conf["opt_mult"] ** i_schedule)),
                'end_value': conf["max_lrate"] / conf["lrate_span"] * conf["lrate_decay"] ** (i_schedule + 1),
            }
            for i_schedule in range(len(epochs_per_schedule))
        ]
    return optax.sgdr_schedule(schedules)
