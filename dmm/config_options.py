import dataclasses
from typing import Dict, List, Optional


@dataclasses.dataclass(repr=True, init=True, frozen=True, eq=True)
class Conf(dict):
    model: str = ""
    data: str = "dream_cytof"
    context: str = ""
    features: str = ""
    samples: str = ""
    sample: str = ""
    # Standard scaling
    standardise_features: bool = False
    # Train/freeze medians
    freeze_medians: bool = False
    # Network structure
    n_hidden: int = 2
    nn_structure_multiplier: int = 2
    depth: int = 0
    multiheaded: bool = False
    use_layer_bias: list[bool] | bool = False
    last_layer_activation: bool = False
    nn_init_fn: str = "custom"
    nn_init_scale: float = (
        0.1  # variance scaling parameter when using custom init
    )
    dropout_rate: float = 0.0  # default: no entries set to 0
    # Training
    activation_fn_name: str = "swish"
    optimiser: str = "adam"
    # Regularisation
    orth_reg_strategy: str = "L2"
    l1reg_encode: float = 0.0
    oreg_encode: float = 0.0
    l1reg_inflate: float = 0.0
    oreg_inflate: float = 0.0
    l1reg_inflater_output: float = 0.0
    l2reg_inflater_output: float = 0.0
    inflater_output_reg_epoch: int = (
        200  # after, regularisation is lifted but the sparsity pattern is kept
    )
    sparse_threshold_perc: str = "gmm"  # specified as top percentage to keep as cell-line specific, default: keep top 50%
    recon_loss: float = 0.01
    symm_reg: float = 0.0
    median_reg: float = 0.0
    # Learning schedule hyperparameters
    max_lrate: float = 0.01  # maximum learning rate (max in first schedule or in all without decay)
    lrate_span: float = (
        1e0  # ratio between max and min learning rates in a given schedule
    )
    lrate_decay: float = (
        1.0  # if < 1, the learning rate decays between schedules.
    )
    # # 0.98 will reduce 1e-2 to 1e-3 in 100 epochs, similarly to our original linear schedule
    warmup_fct: float = (
        0.0  # fraction of schedule epochs to be used for warmup
    )
    opt_steps: int = 10  # Number of steps in the first schedule
    opt_mult: int = 2  # Multiplier for the number of steps in each schedule
    momentum: float = 0.9  # momentum for AdamW
    weight_decay: float = 0.0  # controls weight decay for AdamW
    use_simple_linear_schedule: bool = False
    n_epoch: int = 1000
    inflater_bound: float = 5.0
    # Early-stopping
    use_early_stopping: bool = False
    # Other hyperparams
    job: int = 0
    threads: int = 1
    n_starts: int = None
    date_tag: str = None
    figure: str = None

    def to_dict(self) -> dict:
        """
        Convert the configuration to a dictionary.
        """
        return {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
        }

    def __getitem__(self, key):
        """
        Custom __getitem__ to allow access to configuration parameters as dict.
        """
        if key in self.__dict__:
            return self.__dict__[key]
        raise AttributeError(f"Conf has no attribute '{key}")

    def __str__(
        self,
        replace: Dict[str, str] | None = None,
    ):
        """
        Return string representation with selected hyperparameters.
        """
        # Generate a dictionary of field names and values
        field_dict = {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
        }

        # Update the field_dict with any replacements specified
        if replace is not None:
            field_dict.update(replace)

        # Filter out unwanted fields from the final string representation
        unwanted_fields = [
            "model",
            "data",
            "sample",
            "samples",
            "context",
            "features",
            "pretrain",
            "freeze_medians",
            "use_layer_bias",
            "linear_benchmark",
            "nn_init_fn",
            "max_lrate",
            "lrate_span",
            "lrate_decay",
            "warmup_fct",
            "opt_steps",
            "opt_mult",
            "weight_decay",
            "momentum",
            "use_simple_linear_schedule",
            "use_early_stopping",
            "threads",
            "n_starts",
            "date_tag",
            "figure",
        ]

        # Create a list of values for the fields that are not in the unwanted list
        filtered_values = [
            field_dict[field]
            for field in field_dict
            if field not in unwanted_fields
        ]

        # Return the joined string of the filtered values
        return "__".join(map(str, filtered_values))


@dataclasses.dataclass
class ModuleParams(dict):
    layer_sizes: List[int]
    layer_biases: Optional[List[bool]] = None  # no learnable bias
    weight_init_fn: str = "eqx_default"  # eqx.nn.Linear layers
    bias_init_fn: str = "eqx_default"  # eqx.nn.Linear layers
    last_layer_activation: bool = (
        "False"  # no activation function in last layer of each module
    )
    dropout_rate: float = 0.0
    weight_init_scale: float = 0.1  # only used if weight_init_fn is "custom"


@dataclasses.dataclass
class EarlyStoppingParams(dict):
    use_early_stopping: bool = True
    patience: int = 9
    min_improvement: float = 0


# define abbreviations/labels for logging of loss terms
L1EREG = "l1reg_encode"
OEREG = "oreg_encode"
L1DREG = "l1reg_decode"  # uses the same scale as l1reg_encode
ODREG = "oreg_decode"  # uses the same scale as oreg_encode
L1IREG = "l1reg_inflate"
OIREG = "oreg_inflate"
L1REG_IO = "l1reg_inflater_output"
L2REG_IO = "l2reg_inflater_output"
IO_SPARSITY = "inflater_output_sparsity"
RECON_LOSS = "recon_loss"
SYMM_LOSS = "symm_reg"
MEDIAN_REG = "median_reg"

scan_attributes = [
    "model",
    "samples",
    "context",
    "features",
    "n_hidden",
    "depth",
    "dropout_rate",
    "nn_init_scale",
    L1IREG,
    OIREG,
    L1EREG,
    OEREG,
    L1REG_IO,
    L2REG_IO,
    RECON_LOSS,
    SYMM_LOSS,
    MEDIAN_REG,
    "inflater_output_reg_epoch",
    "job",
    "n_epoch",
    "inflater_bound",
]

for attr in scan_attributes:
    assert hasattr(Conf, attr), f"Conf does not have attribute {attr}"
