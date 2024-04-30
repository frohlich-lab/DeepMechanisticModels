from typing import List

import equinox as eqx
import jax.numpy as jnp
# import numpy as np
import pandas as pd
import pypesto.petab
import petab
from jax import config
from jaxtyping import Array
from . import MODEL_FEATURE_PREFIX
from dmm.janus_autoencoder_eqx import TwoHeadedDeepAutoencoder
from .petab_subproblem import load_petab
from .problem import Problem


config.update("jax_enable_x64", True)


def init_biases(biases, layer_sizes, component_name):
    if biases is None:
        biases = [False] * len(layer_sizes)
    elif len(biases) != len(layer_sizes):
        raise ValueError(f"{component_name}: biases must have the same length as layer_sizes")
    return biases


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


class DeepMechanisticModel(TwoHeadedDeepAutoencoder):
    data_name: str = eqx.static_field()
    pathway_name: str = eqx.static_field()
    n_input_features: int = eqx.static_field()
    n_latent: int = eqx.static_field()
    n_inflated_specific_kin_params: int = eqx.static_field()
    n_global_kin_params: int = eqx.static_field()
    sample_names: List[str] = eqx.static_field()
    x_names: List[str] = eqx.static_field()
    petab_importer: pypesto.petab.PetabImporter = eqx.static_field()
    pypesto_subproblem: pypesto.Problem = eqx.static_field()
    # encoder_params_dict: dict = eqx.static_field()
    # inflater_params_dict: dict = eqx.static_field()
    # decoder_params_dict: dict = eqx.static_field()
    orth_reg_strategy: str = eqx.static_field()
    activation_fn_name: str = eqx.static_field()
    reconstruct: bool = eqx.static_field()

    def __init__(
            self,
            problem: Problem,
            dataset: str,
            encoder_layer_sizes: List[int],  # decoder_layer_sizes = encoder_layer_sizes[::-1]
            inflater_layer_sizes: List[int],
            key: int,
            measurement_table: pd.DataFrame,
            observable_table: pd.DataFrame,
            condition_table: pd.DataFrame,
            samples_list: List[str],
            n_input_features: int,
            n_latent: int,
            n_threads=1,
            # default for all modules: use eqx.nn.Linear layers
            encoder_weight_init_fn: str = "eqx_default",
            encoder_bias_init_fn: str = "eqx_default",
            inflater_weight_init_fn: str = "eqx_default",
            inflater_bias_init_fn: str = "eqx_default",
            decoder_weight_init_fn: str = "eqx_default",
            decoder_bias_init_fn: str = "eqx_default",
            # default: no learnable biases
            encoder_layer_biases: List[bool] = None,
            inflater_layer_biases: List[bool] = None,
            decoder_layer_biases: List[bool] = None,
            # default: input and output layers have no bias - only matters if layer_biases is not None
            encoder_input_output_bias: List[bool] = None,
            inflater_input_output_bias: List[bool] = None,
            decoder_input_output_bias: List[bool] = None,
            orth_reg_strategy: str = "L2",
            activation_fn_name: str = "relu",  # ReLU = Rectified Linear Unit
            reconstruct: bool = False,  # default: single head, no decoder (encoder->inflater)
    ):
        """

        :param dataset:
            name of dataset to use for model.

        :param problem:
            problem.pathway_name contains the name of pathway to use for model.


        -- ENCODER-specific params
        :param encoder_layer_sizes:
            list of hidden layer sizes for encoder component (and decoder component, in reverse).

        :param encoder_weight_init_fn:
            encoder weight initialisation strategy.

        :param encoder_bias_init_fn:
            encoder bias initialisation strategy.

        :param encoder_layer_biases:
            list of bool values indicating whether to add a learnable bias or not for encoder layers.

        -- INFLATER-specific params
        :param inflater_layer_sizes:
            list of hidden layer sizes for inflater component.

        :param inflater_weight_init_fn:
            inflater weight initialisation strategy.

        :param inflater_bias_init_fn:
            inflater bias initialisation strategy.

        :param inflater_layer_biases:
            list of bool values indicating whether to add a learnable bias or not for inflater layers.

        -- DECODER-specific params
        :param decoder_weight_init_fn:
            decoder weight initialisation strategy.

        :param decoder_bias_init_fn:
            decoder bias initialisation strategy.

        :param decoder_layer_biases:
            list of bool values indicating whether to add a learnable bias or not for decoder layers.

        -- OTHER params
        :param key:
            PRNG key.

        :param activation_fn_name:
            choice of activation function.
            Default: ReLU.

        :param reconstruct:
            boolean flag. If set to True, adds a second, autoencoding head to the network
            (encoder->decoder) on top of the first head (encoder->inflater).
            Default: single head (False).

        :param orth_reg_strategy:
            orthogonal regularisation strategy to be used: L1 vs L2 (default).

        :param n_threads:
            number of threads to use for pypesto.

         :param samples_list:
            List of samples (previously features.index).

        :param n_input_features:
            Number of features (not sure if needed).

        """

        self.data_name = dataset
        self.pathway_name = problem.pathway_name

        # TODO @GiacomoFabrini n_params needs to come from petab problem

        # subset samples
        self.petab_importer = load_petab(
            problem=problem,
            dataset=dataset,
            measurement_table=measurement_table,
            condition_table=condition_table,
            observable_table=observable_table,
            samples=samples_list,  # features needed here!
        )
        self.pypesto_subproblem = self.petab_importer.create_problem()

        # extract sample names, ordering of those is important since samples
        # must match when reshaping the inflated matrix
        petab_samples = []
        for name in self.pypesto_subproblem.x_names:
            if not name.startswith(MODEL_FEATURE_PREFIX):
                continue

            sample = name.split("__")[-1]
            if sample not in petab_samples and sample in samples_list:
                petab_samples.append(sample)

        n_samples = len(samples_list)
        self.n_input_features = n_input_features
        self.n_latent = n_latent

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

        # set sample names
        self.sample_names = samples_list

        # set regularisation strategy, activation function and reconstruct flag
        self.orth_reg_strategy = orth_reg_strategy
        self.activation_fn_name = activation_fn_name
        self.reconstruct = reconstruct

        # Update layer_sizes (hidden layers) to include input and output layers
        encoder_layer_sizes = [self.n_input_features] + encoder_layer_sizes + [self.n_latent]
        inflater_layer_sizes = [self.n_latent] + inflater_layer_sizes + [self.n_inflated_specific_kin_params]

        # Same for biases - default
        # if both lists are defined, augment them / if not, simply keep None and they will be initialised to False
        if (encoder_layer_biases is not None) and (encoder_input_output_bias is not None):
            encoder_layer_biases = ([encoder_input_output_bias[0]]
                                    + encoder_layer_biases
                                    + [encoder_input_output_bias[-1]])
        if (decoder_layer_biases is not None) and (decoder_input_output_bias is not None):
            decoder_layer_biases = ([decoder_input_output_bias[0]]
                                    + decoder_layer_biases
                                    + [decoder_input_output_bias[-1]])
        if (inflater_layer_biases is not None) and (inflater_input_output_bias is not None):
            inflater_layer_biases = ([inflater_input_output_bias[0]]
                                     + inflater_layer_biases
                                     + [inflater_input_output_bias[-1]])

        # Initialise module biases to default value if None (i.e. use_bias = False for all)
        # Check for shape mismatches between layer_sizes and layer_biases
        encoder_layer_biases = init_biases(
            biases=encoder_layer_biases,
            layer_sizes=encoder_layer_sizes,
            component_name="encoder"
        )
        inflater_layer_biases = init_biases(
            biases=inflater_layer_biases,
            layer_sizes=inflater_layer_sizes,
            component_name="inflater"
        )
        decoder_layer_biases = init_biases(
            biases=decoder_layer_biases,
            layer_sizes=encoder_layer_sizes,
            component_name="decoder"
        )

        # encoder parameters/properties
        encoder_params_dict = {
            "encoder_layer_sizes": encoder_layer_sizes,
            "encoder_layer_biases": encoder_layer_biases,
            "encoder_weight_init_fn": encoder_weight_init_fn,
            "encoder_bias_init_fn": encoder_bias_init_fn,
        }
        # inflater parameters/properties
        inflater_params_dict = {
            "inflater_layer_sizes": inflater_layer_sizes,
            "inflater_layer_biases": inflater_layer_biases,
            "inflater_weight_init_fn": inflater_weight_init_fn,
            "inflater_bias_init_fn": inflater_bias_init_fn,
        }
        # decoder parameters/properties
        decoder_params_dict = {
            "decoder_layer_biases": decoder_layer_biases,
            "decoder_weight_init_fn": decoder_weight_init_fn,
            "decoder_bias_init_fn": decoder_bias_init_fn,
        }

        # Initialise TwoHeadedDeepAutoencoder
        super().__init__(
            n_input_features=self.n_input_features,
            n_inflated_specific_kin_params=self.n_inflated_specific_kin_params,
            n_global_kin_params=self.n_global_kin_params,
            **encoder_params_dict,
            **inflater_params_dict,
            **decoder_params_dict,
            key=key,
            activation_fn_name=self.activation_fn_name,
            orth_reg_strategy=self.orth_reg_strategy,
            reconstruct=self.reconstruct,
        )

        problem.apply_objective_settings(
            self.pypesto_subproblem.objective, n_threads=n_threads
        )

        # augment TwoHeadedDeepAutoencoder.x_names with ODE x_names
        # self.x_names = self.x_names + [
        #     name
        #     for ix, name in enumerate(self.pypesto_subproblem.x_names)
        #     if not name.startswith(MODEL_FEATURE_PREFIX)
        #     and ix in self.pypesto_subproblem.x_free_indices
        # ]

    def embedding(self, input_data: jnp.ndarray) -> jnp.ndarray:
        return self(input_data)[0]  # array containing all kinetic parameters (global first, cell-line-specific second)

    def l1_encode_reg(
            self,
            scale: float = 1.0
    ):
        """
        L1 regularization of deep encoder weights.
        """
        l1reg_encode_loss = 0
        for layer_num in range(len(self.deep_encoder.layers)):
            w = self.deep_encoder.layers[layer_num].weight
            l1reg_encode_loss += scale * jnp.mean(
                jnp.abs(w)
            )
        return l1reg_encode_loss

    def orth_encode_reg(
            self,
            scale: float = 1.0
    ):
        """
        Orthogonal regularization of deep encoder weights.
        """
        oreg_encode_loss = 0
        reg_exponent = get_reg_exp(self.orth_reg_strategy)
        for layer_num in range(len(self.deep_encoder.layers)):
            w = self.deep_encoder.layers[layer_num].weight
            m = jnp.dot(w.T, w)
            oreg_encode_loss += scale * jnp.mean(
                jnp.abs(m - jnp.eye(m.shape[0]))**reg_exponent
            )
        return oreg_encode_loss

    def l1_inflate_reg(
            self,
            scale: float = 1.0
    ):
        """
        L1 regularization of deep inflater weights.
        """
        l1reg_inflate_loss = 0
        for layer_num in range(len(self.deep_inflater.layers)):
            w = self.deep_inflater.layers[layer_num].weight
            l1reg_inflate_loss += scale * jnp.mean(
                jnp.abs(w)
            )
        return l1reg_inflate_loss

    def orth_inflate_reg(
            self,
            scale: float = 1.0
    ):

        """
        Orthogonal regularization of deep inflater weights.
        """
        oreg_inflate_loss = 0
        reg_exponent = get_reg_exp(self.orth_reg_strategy)
        for layer_num in range(len(self.deep_inflater.layers)):
            w = self.deep_inflater.layers[layer_num].weight
            m = jnp.dot(w, w.T)
            oreg_inflate_loss += scale * jnp.mean(
                jnp.abs(m - jnp.diag(jnp.diag(m)))**reg_exponent
            )
        return oreg_inflate_loss

    def reconstruction_loss(
            self,
            x: Array,  # TODO @GiacomoFabrini is this ok?
            scale: float = 1.0
    ):
        """
        Reconstruction loss of the autoencoder (in case self.reconstruct == True).
        Simple Mean Squared Error (without the sqrt for now!)
        """
        reconstructed_x = self(x=x)[1]  # decoded
        # TODO @GiacomoFabrini: consider moving all MSEs to RMSEs?!
        #  Are they on the same scale/order of magnitude as
        #  L1 terms if we leave them squared?!
        return scale*mse(predictions=reconstructed_x, targets=x)

    def symmetry_loss(
            self,
            scale: float = 1.0
    ):
        """
        Symmetry loss for the autoencoder (in case self.reconstruct == True),
        pushes the decoder weights to be the transposed of the encoder weigths.
        """
        symmetry_reg = 0
        num_layers = len(self.deep_encoder.layers)
        # Iterate over the encoder and decoder layers
        for encoder_layer, decoder_layer in zip(
                self.deep_encoder.layers, self.deep_decoder.layers[::-1]  # zip them in reverse order
        ):
            # Compute the weight difference for each pair of corresponding layers
            diff = encoder_layer.weight - decoder_layer.weight.T
            # Then compute sum of squares differences
            symmetry_reg += jnp.sum(jnp.square(diff))
        symmetry_reg /= num_layers  # turns into mean square error
        return scale * symmetry_reg
