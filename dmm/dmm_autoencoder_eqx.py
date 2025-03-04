import equinox as eqx
import jax
import jax.numpy as jnp
import json
import pandas as pd
import pypesto.petab

from . import MODEL_FEATURE_PREFIX
from .config_options import ModuleParams
from .deepcomponent_eqx import KinParamsCombiner
from .model_utils import generate_layer_sizes
from .petab_subproblem import load_petab
from .problem import Problem
from .two_headed_deep_autoencoder_eqx import TwoHeadedDeepAutoencoder
from jaxtyping import Array
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union


def get_reg_exp(orth_reg_strategy):
    reg_exp_dict = {
        "L1": 1,
        "L2": 2,
    }
    if orth_reg_strategy not in reg_exp_dict.keys():
        raise ValueError(f"Invalid orth_reg_strategy: {orth_reg_strategy}")
    return reg_exp_dict[orth_reg_strategy]


def mse(
        predictions: Array,
        targets: Array,
):
    """
    Computes the Mean Squared Error (MSE) between predictions and targets.
    """
    return jnp.mean(jnp.square(predictions - targets))


def init_biases(biases, num_layers):
    if (biases is None) or (not biases):  # None or False
        biases = [False] * num_layers
    elif biases:
        biases = [True] * num_layers
    # removed - either all layers have biases or none have biases
    # elif len(biases) != num_layers:
    #     raise ValueError("Biases must have the same length as layer_sizes!")
    return biases


def update_module_params_dict(
        module_params: ModuleParams,
        new_layer_sizes: List[int],
) -> ModuleParams:
    # Initialise biases (in case of None or single value definitions)
    new_layer_biases = init_biases(
        biases=module_params.layer_biases,
        num_layers=(len(new_layer_sizes) - 1),
    )
    # Produce updated module parameters dictionary
    updated_module_params = ModuleParams(
        layer_sizes=new_layer_sizes,
        layer_biases=new_layer_biases,
        weight_init_fn=module_params.weight_init_fn,
        bias_init_fn=module_params.bias_init_fn
    )
    return updated_module_params


class DeepMechanisticModel(TwoHeadedDeepAutoencoder):
    kin_params_combiner: KinParamsCombiner
    sparsity_binary_mask: Tuple  # prevent undesired updates

    dataset_name: str = eqx.static_field()
    # pathway_name: str = eqx.static_field()  # not used?!
    module_depth: int = eqx.static_field()
    module_structure_multiplier: int = eqx.static_field()
    use_layer_bias: bool = eqx.static_field()
    last_layer_activation: bool = eqx.static_field()
    weight_init_fn: str = eqx.static_field()
    bias_init_fn: str = eqx.static_field()
    sample_name_list: List[str] = eqx.static_field()
    n_input_features: int = eqx.static_field()
    n_latent: int = eqx.static_field()
    n_threads: int = eqx.static_field()
    orth_reg_strategy: str = eqx.static_field()
    activation_fn_name: str = eqx.static_field()
    reconstruct: bool = eqx.static_field()
    model_key: Any = eqx.static_field()


    petab_importer: pypesto.petab.PetabImporter = eqx.static_field()
    pypesto_subproblem: pypesto.Problem = eqx.static_field()
    n_inflated_specific_kin_params: int = eqx.static_field()
    n_global_kin_params: int = eqx.static_field()

    def __init__(
            self,
            problem: Problem,
            dataset: str,
            module_depth: int,
            module_structure_multiplier: int,
            use_layer_bias: bool,
            last_layer_activation: bool,
            weight_init_fn: str,
            bias_init_fn: str,
            key: Any,
            measurement_table: pd.DataFrame,
            observable_table: pd.DataFrame,
            condition_table: pd.DataFrame,
            sample_name_list: List[str],
            n_input_features: int,
            n_latent: int,
            n_threads: int = 1,
            orth_reg_strategy: str = "L2",
            activation_fn_name: str = "relu",  # ReLU = Rectified Linear Unit
            reconstruct: bool = False,  # default: single head, no decoder (encoder->inflater)
    ):
        """

        :param problem:
            problem.pathway_name contains the name of pathway to use for model.

        :param dataset:
            name of dataset to use for model.

        :param module_depth:
            number of hidden layers for encoder/inflater/decoder modules.

        :param module_structure_multiplier:
            multiplier for the width of subsequent hidden layers in encoder (reversed order)/inflater/decoder modules.

        :param use_layer_bias:
            boolean flag regulating the use/lack of biases in module layers.

        :param last_layer_activation:
            boolean flag regulating the use of a non-linear activation function in the last layer.

        :param weight_init_fn:
            weight initialisation function to use for module layers.

        :param bias_init_fn:
            bias initialisation function to use for module layers.

        :param key:
            PRNG key.

        :param measurement_table:
            petab measurement table (pandas DataFrame).

        :param observable_table:
            petab observable table (pandas DataFrame).

        :param condition_table:
            petab condition table (pandas DataFrame).

        :param sample_name_list:
            list of sample names (previously `features.index`).

        :param n_input_features:
            Number of features (not sure if needed).

        :param n_latent:
            Number of latent features / dimension of the bottleneck, compressed representation.

        :param n_threads:
            number of threads to use for pypesto.

        :param orth_reg_strategy:
            orthogonal regularisation strategy to be used: L1 vs L2 (default).

        :param activation_fn_name:
            choice of activation function.
            Default: ReLU.

        :param reconstruct:
            boolean flag. If set to True, adds a second, autoencoding head to the network
            (encoder->decoder) on top of the first head (encoder->inflater).
            Default: single head (False).

        """

        self.dataset_name = dataset

        self.module_depth = module_depth
        self.module_structure_multiplier = module_structure_multiplier
        self.use_layer_bias = use_layer_bias
        self.last_layer_activation = last_layer_activation
        self.weight_init_fn = weight_init_fn
        self.bias_init_fn = bias_init_fn
        self.reconstruct = reconstruct
        self.orth_reg_strategy = orth_reg_strategy
        self.activation_fn_name = activation_fn_name

        self.n_input_features = n_input_features
        self.n_latent = n_latent

        self.n_threads = n_threads

        self.model_key = key

        # self.pathway_name = problem.pathway_name  # not used?!

        # Get petab_importer and pypesto_subproblem
        self.petab_importer = load_petab(
            problem=problem,
            dataset=self.dataset_name,
            measurement_table=measurement_table,
            condition_table=condition_table,
            observable_table=observable_table,
            samples=sample_name_list,  # these will get sorted within petab_importer
        )
        self.pypesto_subproblem = self.petab_importer.create_problem()

        # extract sample names, ordering of those is important since samples
        # must match when reshaping the inflated matrix
        petab_samples = []
        for name in self.pypesto_subproblem.x_names:
            if not name.startswith(MODEL_FEATURE_PREFIX):
                continue

            sample = name.split("__")[-1]
            if sample not in petab_samples and sample in sample_name_list:
                petab_samples.append(sample)

        # Store sample names
        self.sample_name_list = petab_samples
        n_samples = len(self.sample_name_list)

        # n_inflated_specific_kin_params = number of cell-line-specific parameters (per cell-line = sample)
        # these kinetic parameters are the targets of the inflater module (previously model_inputs)
        self.n_inflated_specific_kin_params = int(
            sum(
                name.startswith(MODEL_FEATURE_PREFIX)
                for name in self.pypesto_subproblem.x_names
            )
            / n_samples
        )

        # n_global_kin_params = number of NON cell-line specific parameters (previously n_kin_params)
        self.n_global_kin_params = (
                self.pypesto_subproblem.dim - self.n_inflated_specific_kin_params * n_samples
        )

        # Generate layer_sizes for whole modules (input, hidden, output)
        encoder_layer_sizes = [
            self.n_input_features, *generate_layer_sizes(
                latent_dim=self.n_latent,
                depth=self.module_depth,
                max_width=self.n_input_features,  # TODO check whether we need to double (*2) this
                multiplier=self.module_structure_multiplier,
                reverse=True,
            ), self.n_latent
        ]
        inflater_layer_sizes = [
            self.n_latent, *generate_layer_sizes(
                latent_dim=self.n_latent,
                depth=self.module_depth,
                max_width=self.n_inflated_specific_kin_params,  # TODO check whether we need to double (*2) this
                multiplier=self.module_structure_multiplier,
                reverse=False,
            ), self.n_inflated_specific_kin_params
        ]

        # Define encoder, inflater and decoder parameters
        params = {
            f"{module}_params": ModuleParams(
                layer_sizes=layer_sizes,
                layer_biases=[self.use_layer_bias]*len(layer_sizes),
                weight_init_fn=self.weight_init_fn,
                bias_init_fn=self.bias_init_fn,
                last_layer_activation=self.last_layer_activation,
                # TODO @GiacomoFabrini: discuss with Fabian which modules should have a last layer activation (all?)
            )
            for module, layer_sizes in zip(
                ["encoder", "inflater", "decoder"],
                [encoder_layer_sizes, inflater_layer_sizes, encoder_layer_sizes[::-1]]
            )
        }

        # Instantiate Kinetic Parameters Combiner module
        self.kin_params_combiner = KinParamsCombiner(
            component_name='kin_params_combiner',
            n_global_kin_params=self.n_global_kin_params
        )

        # Initialise dummy sparsity binary mask with a tuple of ones the same size as inflater kinetic param dev
        self.sparsity_binary_mask = tuple(
            [1 for _ in range(self.n_inflated_specific_kin_params)]
        )

        # Initialise TwoHeadedDeepAutoencoder
        super().__init__(
            **params,
            key=self.model_key,
            activation_fn_name=self.activation_fn_name,
            reconstruct=self.reconstruct,
        )

        problem.apply_objective_settings(
            self.pypesto_subproblem.objective, n_threads=self.n_threads
        )


    def __call__(self, x):
        # Call the parent __call__ method to get the original outputs
        outputs = super().__call__(x)
        # Apply the sparsity binary mask element-wise -- since it's a Tuple, it's not learnt/updated
        outputs["inflated"] = outputs["inflated"] * jnp.array(self.sparsity_binary_mask)
        # Finally, introduce soft constrain within ±5 (hardcoded) range through rescaled tanh: a * tanh(x/a), a=5
        outputs["inflated"] = 5 * jnp.tanh(outputs["inflated"] / 5)
        return outputs


    def update_sparsity_binary_mask(self, x, threshold_perc: int = 50, round_up: bool = False):
        """
        Update the sparsity binary mask based on the median parameter deviation across samples.
        :param x:
            input data.
        :param threshold_perc:
            percentage of the median parameter deviations to retain as cell-line-specific (default: 50%, specified as 50).
        :param round_up:
            boolean flag to round up or down when computing the threshold (default: False).

        :return:
            new instance of DMM with updated sparsity binary mask.
        """
        # Compute absolute median param dev across samples
        absolute_param_dev_median = jnp.abs(jnp.median(jax.vmap(self)(x)["inflated"], axis=0))

        # Sort absolute median param dev in descending order
        sorted_deviations = jnp.sort(absolute_param_dev_median)[::-1]

        # Compute threshold to keep threshold_perc values and ensure within bounds
        # Given the number of cell-line-specific params is odd, we can choose whether to round up or down
        # Considering we want sparsity, I have opted to round down by default - behaviour can be changed via round_up.
        threshold = sorted_deviations[jnp.clip(
            int(jnp.floor(len(sorted_deviations) * (1 - threshold_perc/100))) - 1 if not round_up else
            int(jnp.ceil(len(sorted_deviations) * (1 - threshold_perc/100))) - 1,
            0,
            len(sorted_deviations) - 1
        )]

        # Check kinetic parameter deviation and zero out entries in the sparsity mask if below threshold
        new_sparsity_binary_mask = tuple(jnp.where(
            absolute_param_dev_median < threshold,
            0.0,
            jnp.array(self.sparsity_binary_mask)
        ).tolist())
        return eqx.tree_at(
            lambda model: model.sparsity_binary_mask,
            self,
            new_sparsity_binary_mask,
            is_leaf=lambda leaf: type(leaf) is tuple  # is this needed?
        )


    def embedding(self, input_data: jnp.ndarray) -> jnp.ndarray:
        return self(input_data)["inflated"]  # inflated kinetic parameters (global first, cell-line-specific second)

    def l1_encode_reg(
            self,
            scale: float = 1.0
    ):
        """
        L1 regularization of deep encoder weights.
        """
        return l1reg(self.deep_encoder, scale)

    def orth_encode_reg(
            self,
            scale: float = 1.0
    ):
        """
        Orthogonal regularization of deep encoder weights.
        """
        return orth_reg(self.deep_encoder, self.orth_reg_strategy, "encoder", scale)

    def l1_decode_reg(
            self,
            scale: float = 1.0
    ):
        """
        L1 regularization of deep decoder weights.
        """
        return l1reg(self.deep_decoder, scale)

    def orth_decode_reg(
            self,
            scale: float = 1.0
    ):
        """
        Orthogonal regularization of deep encoder weights.
        """
        return orth_reg(self.deep_decoder, self.orth_reg_strategy, "decoder", scale)

    def l1_inflate_reg(
            self,
            scale: float = 1.0
    ):
        """
        L1 regularization of deep inflater weights.
        """
        return l1reg(self.deep_inflater, scale)

    def orth_inflate_reg(
            self,
            scale: float = 1.0
    ):

        """
        Orthogonal regularization of deep inflater weights.
        """
        return orth_reg(self.deep_inflater, self.orth_reg_strategy, "inflater", scale)

    def l1reg_inflater_output(
            self,
            x: Array,
            scale: float = 1.0
    ):
        """
        L1 regularization of inflater output - number of cell-specific deviations/log fold-changes.
        """
        # Introduced 1e-6 multiplier to investigate lower regularisation strengths without formatting issues
        return scale * 1e-6 * jnp.sum(
            jnp.abs(
                jax.vmap(self)(x)["inflated"]
            )
        )

    def reconstruction_loss(
            self,
            x: Array,  # TODO @GiacomoFabrini is this ok?
            scale: float = 1.0
    ):
        """
        Reconstruction loss of the autoencoder (in case `self.reconstruct` == True).
        Simple Mean Squared Error (without the sqrt for now!)
        """
        reconstructed_x = jax.vmap(self)(x)["decoded"]
        # fval contains MSE (not RMSE) - using MSE in reconstruction loss
        # TODO @GiacomoFabrini: fval and reconstruction loss use MSEs - need to move to RMSEs?!
        #  Are they on the same scale/order of magnitude as L1 terms if we leave them squared?!
        return scale * mse(predictions=reconstructed_x, targets=x)

    def symmetry_loss(
            self,
            scale: float = 1.0
    ):
        """
        Symmetry loss for the autoencoder (in case `self.reconstruct` == True),
        pushes the decoder weights to be the transposed of the encoder weights.
        """
        symmetry_reg = 0
        num_layers = len(self.deep_encoder.layers)
        # Iterate over the encoder and decoder layers
        for encoder_layer, decoder_layer in zip(
                self.deep_encoder.layers, self.deep_decoder.layers[::-1]  # zip them in reverse order
        ):
            # Compute the weight difference for each pair of corresponding layers
            diff = encoder_layer.weight - decoder_layer.weight.T
            # Then compute mean squares differences per layer
            symmetry_reg += jnp.mean(jnp.square(diff))
        return scale * symmetry_reg/num_layers  # mean across layers - should be on the same order of magnitude as MSE


    def constrain_median(self, x: Array, scale: float = 1.0):
        """
        Constrain median of global parameters to be close to initialisation (avg_model/per_sample), x.
        """
        return scale * mse(predictions=self.kin_params_combiner.learned_global_kin_params, targets=x)


    # inspired from Fabian's NeuralCoarseGraining
    # see: https://github.com/frohlich-lab/NeuralCoarseGraining/blob/main/ncg/static.py
    def get_hyperparams(self, samples_list_dict: dict = None) -> dict[str, Union[int, dict]]:
        """
        Get the hyperparameters of the model.

        Note: used in model serialisation
        """
        return {
            'dataset': self.dataset_name,
            'module_depth': self.module_depth,
            'module_structure_multiplier': self.module_structure_multiplier,
            'use_layer_bias': self.use_layer_bias,
            'last_layer_activation': self.last_layer_activation,
            'weight_init_fn': self.weight_init_fn,
            'bias_init_fn': self.bias_init_fn,
            'sample_name_list': self.sample_name_list if samples_list_dict is None else samples_list_dict,
            'n_input_features': self.n_input_features,
            'n_latent': self.n_latent,
            'n_threads': self.n_threads,
            'orth_reg_strategy': self.orth_reg_strategy,
            'activation_fn_name': self.activation_fn_name,
            'reconstruct': self.reconstruct,
            'key': self.model_key.tolist()
        }

    def save(self, filename: Path, samples_list_dict: dict = None) -> None:
        """
        Save the model to a file.

        :param filename: path of file
        :param samples_list_dict: dictionary of samples list (train/test)
        """
        filename.parent.mkdir(exist_ok=True, parents=True)
        with Path.open(filename, 'wb') as f:
            # Save model hyperparameters
            hyperparam_str = json.dumps(self.get_hyperparams(samples_list_dict))
            f.write((hyperparam_str + '\n').encode())
            # Save model parameters (weights, biases)
            eqx.tree_serialise_leaves(f, self)

    @classmethod
    def load(
            cls,
            filename: Union[Path, str],
            problem: Problem,  # not serialisable in json
            dataset: str,
            petab_base_files: Dict[str, pd.DataFrame],
    ) -> 'DeepMechanisticModel':
        """
        Loads DMM model from a file.

        :param filename: path of file
        :param problem: CytofProblem instance
        :param dataset: dataset name (train/test)
        :param petab_base_files: petab base files (measurement, observable, condition tables)
        :return: Model instance
        """
        # Ensure filename is a Path object
        filename = Path(filename)
        with Path.open(filename, 'rb') as f:
            # Load model hyperparameters
            hyperparam_str = f.readline().decode().strip()
            hyperparams = json.loads(hyperparam_str)
            # Handle parameters that require conversion
            # Key: convert back to ArrayImpl with expected dtype
            # TODO @GiacomoFabrini: is this necessary?
            hyperparams['key'] = jnp.array(hyperparams['key'], dtype=jnp.uint32)
            # Subset sample_name_list dictionary to corresponding dataset
            if isinstance(hyperparams['sample_name_list'], dict) and dataset is not None:
                hyperparams['sample_name_list'] = hyperparams['sample_name_list'][dataset]
            # Make model skeleton
            model = cls(
                **hyperparams,
                problem=problem,
                **petab_base_files,
            )
            # Apply serialised weights and biases to model skeleton
            model = eqx.tree_deserialise_leaves(f, model)
        return model


def l1reg(
    module,
    scale: float = 1.0
):
    """
    L1 regularization of generic module weights.
    """
    l1reg_loss = 0
    for layer in module.layers:
        w = layer.weight
        l1reg_loss += scale * jnp.mean(
            jnp.abs(w)
        )
    return l1reg_loss / len(module.layers)  # mean across all layers


def orth_reg(
    module,
    orth_reg_strategy,
    mode: str,
    scale: float = 1.0
):
    """
    Orthogonal regularization of generic module weights.
    """
    oreg_loss = 0
    reg_exponent = get_reg_exp(orth_reg_strategy)
    if mode == "encoder":
        for layer in module.layers:
            w = layer.weight
            m = jnp.dot(w.T, w)
            oreg_loss += scale * jnp.mean(
                jnp.abs(m - jnp.eye(m.shape[0])) ** reg_exponent
            )
    else:  # decoder, inflater
        for layer in module.layers:
            w = layer.weight
            m = jnp.dot(w, w.T)
            oreg_loss += scale * jnp.mean(
                jnp.abs(m - jnp.diag(jnp.diag(m))) ** reg_exponent
            )
    return oreg_loss/len(module.layers)  # mean across all layers