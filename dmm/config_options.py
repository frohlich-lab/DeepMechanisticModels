import dataclasses
from typing import Dict, List, Optional


@dataclasses.dataclass(repr=True, init=True)
class Conf(dict):
    model: str
    data: str
    context: str = None
    features: str = None
    features_selection: str = None
    # Transform input features via standard scaling and PCA
    features_transform: str = None
    samples: str = None
    pretrain: bool = False
    sample: str = None
    # Initialisation
    median_init: str = "None"  # can be either `per_sample` or `avg_model` -> initialises params of KinParamsCombiner
    # Train/freeze medians
    freeze_medians: bool = False
    # Network structure
    n_hidden: int = 0
    nn_structure_multiplier: int = 0
    depth: int = 0
    linear_benchmark: bool = False
    use_layer_bias: List[bool] = False
    last_layer_activation: bool = False
    nn_init_fn: str = "None"
    reconstruct: bool = False
    # Training
    activation_fn_name: str = "None"
    optimiser: str = "None"
    # Regularisation
    orth_reg_strategy: str = "None"
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
    recon_loss: float = 0.0
    symm_reg: float = 0.0
    median_reg: float = 0.0
    # Learning schedule hyperparameters
    max_lrate: Optional[
        float
    ] = 0.01  # maximum learning rate (max in first schedule or in all without decay)
    lrate_span: Optional[
        float
    ] = 1e0  # ratio between max and min learning rates in a given schedule
    lrate_decay: Optional[
        float
    ] = 0.98  # if < 1, the learning rate decays between schedules.
    # # 0.98 will reduce 1e-2 to 1e-3 in 100 epochs, similarly to our original linear schedule
    warmup_fct: Optional[
        float
    ] = 0.0  # fraction of schedule epochs to be used for warmup
    opt_steps: Optional[int] = 0  # Number of steps in the first schedule
    opt_mult: Optional[
        int
    ] = 0  # Multiplier for the number of steps in each schedule
    momentum: Optional[float] = 0.9  # momentum for AdamW
    weight_decay: Optional[float] = 1e-4  # controls weight decay for AdamW
    use_simple_linear_schedule: bool = False
    # Early-stopping
    use_early_stopping: bool = False
    # Other hyperparams
    job: int = 0
    threads: int = 1
    n_starts: int = None
    run_mode_tag: str = None
    date_tag: str = None

    def __str__(
        self,
        replace: Optional[Dict[str, str]] = None,
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
            "median_init",
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
            "run_mode_tag",
            "date_tag",
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


@dataclasses.dataclass
class EarlyStoppingParams(dict):
    use_early_stopping: bool = True
    patience: int = 9
    min_improvement: float = 0


unwanted_attributes = [
    "model",
    "data",
    "sample",
    "threads",
    "n_starts",
    "run_mode_tag",
    "date_tag",
]

default_attributes = [
    k
    for k, v in vars(Conf).items()
    if not k.startswith("__") and k not in unwanted_attributes
]

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
