from typing import Any, Union

import equinox as eqx
from jax import config, random

from .config_options import ModuleParams
from .deepcomponent_eqx import DeepComponent

config.update("jax_enable_x64", True)


class TwoHeadedDeepAutoencoder(eqx.Module):
    """
    A potentially deep, non-linear and two-headed Autoencoder.
        - First head: encoder -> inflater (kinetic parameters/cell-line-specific parameter deviations of ODE model).
        - Second head: encoder -> decoder (input reconstruction): traditional autoencoder behaviour.

    :param encoder_params:
        Parameters for the encoder module.

    :param inflater_params:
        Parameters for the inflater module.

    :param decoder_params:
        Parameters for the decoder module.

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
    """

    deep_encoder: DeepComponent
    deep_inflater: DeepComponent
    deep_decoder: Union[eqx.Module, DeepComponent]

    encoder_params: ModuleParams = eqx.field(static=True)
    inflater_params: ModuleParams = eqx.field(static=True)
    decoder_params: ModuleParams = eqx.field(static=True)
    reconstruct: bool = eqx.field(static=True)

    def __init__(
        self,
        encoder_params: ModuleParams,
        inflater_params: ModuleParams,
        decoder_params: ModuleParams,
        key: Any,
        activation_fn_name: str,  # default activation_fn_name is ReLU if more than one layer is present
        reconstruct: bool,  # current default behaviour uses a single head (encoder->inflater),
    ):
        # CHECKS
        # encoder layers must shrink towards bottleneck/latent representation -- by default - remove?
        if encoder_params.layer_sizes[-1] > encoder_params.layer_sizes[0]:
            raise ValueError(
                "Latent space size cannot be larger than input feature space size!"
            )
        elif inflater_params.layer_sizes[0] > inflater_params.layer_sizes[1]:
            raise ValueError(
                "Latent space size cannot be larger than output/kinetic parameters feature space size!"
            )

        # Set module parameters
        self.encoder_params = encoder_params
        self.inflater_params = inflater_params
        self.decoder_params = decoder_params

        # Set reconstruct flag
        self.reconstruct = reconstruct

        # Split random key
        key_encoder, key_inflater, key_decoder = random.split(key, num=3)

        # Instantiate encoder component
        self.deep_encoder = DeepComponent(
            component_name="encoder",
            layer_sizes=self.encoder_params.layer_sizes,
            biases=self.encoder_params.layer_biases,
            key=key_encoder,
            activation_fn_name=activation_fn_name,
            last_layer_activation=self.encoder_params.last_layer_activation,
            weight_init_fn=self.encoder_params.weight_init_fn,
            bias_init_fn=self.encoder_params.bias_init_fn,
            use_dropout=self.encoder_params.use_dropout,
            dropout_rate=self.encoder_params.dropout_rate,
        )

        # Instantiate inflater component
        self.deep_inflater = DeepComponent(
            component_name="inflater",
            layer_sizes=self.inflater_params.layer_sizes,
            biases=self.inflater_params.layer_biases,
            key=key_inflater,
            activation_fn_name=activation_fn_name,
            last_layer_activation=self.inflater_params.last_layer_activation,
            weight_init_fn=self.inflater_params.weight_init_fn,
            bias_init_fn=self.inflater_params.bias_init_fn,
            # no dropout
            use_dropout=False,
            dropout_rate=0.0,
        )

        # Instantiate decoder component if two-headed autoencoder
        if self.reconstruct:
            self.deep_decoder = DeepComponent(
                component_name="decoder",
                layer_sizes=self.decoder_params.layer_sizes,
                biases=self.decoder_params.layer_biases,
                key=key_decoder,
                activation_fn_name=activation_fn_name,
                last_layer_activation=self.decoder_params.last_layer_activation,
                weight_init_fn=self.decoder_params.weight_init_fn,
                bias_init_fn=self.decoder_params.bias_init_fn,
                # no dropout
                use_dropout=False,
                dropout_rate=0.0,
            )
        else:
            self.deep_decoder = eqx.nn.Identity()  # no decoder head

    def __call__(self, x, key):
        encoded = self.deep_encoder(x, key)
        inflated = self.deep_inflater(encoded, None)
        # If using decoding head, pass encoding through decoder, else just leave second output blank (None)
        decoded = self.deep_decoder(encoded, None) if self.reconstruct else None
        return {"inflated": inflated, "decoded": decoded}
