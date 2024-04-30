import equinox as eqx
import jax.numpy as jnp

from dmm.deepcomponent_eqx import (
    DeepComponent,
    KinParamsCombiner,
)
from jax import config, random
from jaxtyping import Array
from typing import (
    List,
)

config.update("jax_enable_x64", True)


class TwoHeadedDeepAutoencoder(eqx.Module):
    """
    A potentially deep, non-linear and two-headed Autoencoder.
        - First head: encoder -> inflater (kinetic parameters of ODE model).
        - Second head: encoder -> decoder (input reconstruction): traditional autoencoder behaviour.

    -- ENCODER PARAMS
    :param encoder_layer_sizes:
        list of layer sizes for encoder component (and decoder component, in reverse)

    :param encoder_weight_init_fn:
        encoder weight initialisation strategy.

    :param encoder_bias_init_fn:
        encoder bias initialisation strategy.

    :param encoder_layer_biases:
        list of bool values indicating whether to add a learnable bias or not for encoder layers

    -- INFLATER PARAMS
    :param inflater_layer_sizes:
        list of layer sizes for inflater component

    :param inflater_weight_init_fn:
        inflater weight initialisation strategy.

    :param inflater_bias_init_fn:
        inflater bias initialisation strategy.

    :param inflater_layer_biases:
        list of bool values indicating whether to add a learnable bias or not for inflater layers

    -- KineticParametersCombiner PARAMS
    :param n_global_kin_params:
        Number of global kinetic parameters (ODE non cell-line-specific model parameters).

    -- DECODER PARAMS
    :param decoder_weight_init_fn:
        decoder weight initialisation strategy.

    :param decoder_bias_init_fn:
        decoder bias initialisation strategy.

    :param decoder_layer_biases:
        list of bool values indicating whether to add a learnable bias or not for decoder layers

    -- OTHER PARAMETERS
    :param key:
        PRNG key.

    :param activation_fn_name:
        name of the activation function (selected from act_fn_by_name dictionary)

    :param reconstruct:
        boolean flag. If set to True, the output of the encoder is fed both
        into the inflater network to inform the kinetic parameters of the ODE and
        into the decoder network to ensure latent space can reconstruct the input information
        via reconstruction loss in true autoencoder spirit.

    :param orth_reg_strategy:
        orthogonal regularisation strategy to be used: L1 vs L2 (default)

    :param n_input_features:
        Number of input features (encoder input feature space).

    :param n_inflated_specific_kin_params:
        Number of kinetic parameters to inflate to (inflater output size).


    """

    n_input_features: int = eqx.static_field()  # encoder input size
    n_inflated_specific_kin_params: int = eqx.static_field()  # inflater output size
    n_global_kin_params: int = eqx.static_field()
    x_names: List[str] = eqx.static_field()
    orth_reg_strategy: str = eqx.static_field()
    reconstruct: bool = eqx.static_field()

    deep_encoder: DeepComponent
    deep_inflater: DeepComponent
    deep_decoder: eqx.Module
    kin_params_combiner: KinParamsCombiner

    # TODO @GiacomoFabrini do we need self.x_names?
    def __init__(
            self,
            n_input_features: int,
            n_inflated_specific_kin_params: int,
            n_global_kin_params: int,
            encoder_layer_sizes: List[int],  # decoder layers are just going to be encoder_layer_sizes mirrored
            encoder_weight_init_fn: str,
            encoder_bias_init_fn: str,
            encoder_layer_biases: List[bool],
            inflater_layer_sizes: List[int],
            inflater_weight_init_fn: str,
            inflater_bias_init_fn: str,
            inflater_layer_biases: List[bool],
            decoder_weight_init_fn: str,
            decoder_bias_init_fn: str,
            decoder_layer_biases: List[bool],
            key,
            activation_fn_name: str,  # default activation_fn_name is ReLU if more than one layer is present
            orth_reg_strategy: str,  # default orthogonal regularisation strategy is L2
            reconstruct: bool,  # current default behaviour uses a single head (encoder->inflater),
    ):

        self.n_input_features = n_input_features
        self.n_inflated_specific_kin_params = n_inflated_specific_kin_params
        self.n_global_kin_params = n_global_kin_params

        # VALIDITY CHECKS
        # input size (self.n_features) must match input layer size of encoder (equal to decoder output size, if any)
        # AUTOMATICALLY TRUE - we set the hidden layer sizes, but the input layer is set via n_features
        # if encoder_layer_sizes[0] != self.n_features:
        #     raise ValueError("Input layer size must be the same as input feature space size!")
        # encoder layers must shrink towards bottleneck/latent representation
        if encoder_layer_sizes[-1] > encoder_layer_sizes[0]:
            raise ValueError("Latent space size cannot be larger than input feature space size!")
        # TODO @GiacomoFabrini: need to implement this check in training/train - features.ndim not available here
        # elif features.ndim != 2:
        #     raise ValueError("features expected to be two-dimensional!")
        # encoder output size must match inflater input size (interface between components/modules)
        # AUTOMATICALLY TRUE - we set bottlenect layer based on n_latent
        # elif encoder_layer_sizes[-1] != inflater_layer_sizes[0]:
        #     raise ValueError("Encoder output size must match inflater input size!")
        # inflater output size must match the number of kinetic parameters to inflate to
        # AUTOMATICALLY TRUE - we set output layer of inflater based on n_inflated_kin_params
        # elif inflater_layer_sizes[-1] != self.n_inflated_kin_params:
        #     raise ValueError("Last inflater layer size must match the number of kinetic parameters to inflate to!")

        # Set orthogonal regularisation strategy
        self.orth_reg_strategy = orth_reg_strategy

        # Set reconstruct flag
        self.reconstruct = reconstruct

        # Split random key
        key_encoder, key_inflater, key_decoder = random.split(key, num=3)

        # Instantiate encoder component
        self.deep_encoder = DeepComponent(
            component_name="encoder",
            layer_sizes=encoder_layer_sizes,
            biases=encoder_layer_biases,
            key=key_encoder,
            activation_fn_name=activation_fn_name,
            weight_init_fn=encoder_weight_init_fn,
            bias_init_fn=encoder_bias_init_fn,
        )

        # Instantiate inflater component
        self.deep_inflater = DeepComponent(
            component_name="inflater",
            layer_sizes=inflater_layer_sizes,
            biases=inflater_layer_biases,
            key=key_inflater,
            activation_fn_name=activation_fn_name,
            weight_init_fn=inflater_weight_init_fn,
            bias_init_fn=inflater_bias_init_fn,
        )

        # Instantiate global kinetic parameters component
        self.kin_params_combiner = KinParamsCombiner(
            component_name='kin_params_combiner',
            n_inflated_specific_kin_params=n_inflated_specific_kin_params,
            n_global_kin_params=n_global_kin_params
        )

        self.x_names = self.deep_encoder.x_names + self.deep_inflater.x_names + self.kin_params_combiner.x_names

        if self.reconstruct:
            decoder_layer_sizes = encoder_layer_sizes[::-1]
            self.deep_decoder = DeepComponent(
                component_name="decoder",
                layer_sizes=decoder_layer_sizes,
                biases=decoder_layer_biases,
                key=key_decoder,
                activation_fn_name=activation_fn_name,
                weight_init_fn=decoder_weight_init_fn,
                bias_init_fn=decoder_bias_init_fn,
            )

            self.x_names += self.deep_decoder.x_names

        else:
            self.deep_decoder = eqx.nn.Identity()  # no decoder head

    def __call__(self, x):
        encoded = self.deep_encoder(x)
        inflated = self.deep_inflater(encoded)
        # If using decoding head, pass encoding through decoder, else just leave second output blank (None)
        decoded = self.deep_decoder(encoded) if self.reconstruct else None
        augmented_inflated = self.kin_params_combiner(inflated)
        # augmented_inflated: concatenation of global kinetic parameters
        # and flattened cell-line-specific params (inflated deviations + learned medians)
        return augmented_inflated, decoded
