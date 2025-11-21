from typing import Any

import equinox as eqx
import jax.numpy as jnp
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

    deep_encoder: DeepComponent | list[DeepComponent]
    deep_inflater: DeepComponent
    deep_decoder: eqx.Module | DeepComponent | list[DeepComponent]
    multiheaded: bool

    def __init__(
        self,
        encoder_params: ModuleParams,
        inflater_params: ModuleParams,
        decoder_params: ModuleParams,
        key: Any,
        activation_fn_name: str,
        reconstruct: bool,
        multiheaded: bool,
    ):
        # # encoder layers must shrink towards bottleneck/latent representation -- by default - remove?
        # if encoder_params.layer_sizes[-1] > encoder_params.layer_sizes[0]:
        #     raise ValueError(
        #         "Latent space size cannot be larger than input feature space size!"
        #     )
        if inflater_params.layer_sizes[0] > inflater_params.layer_sizes[1]:
            raise ValueError(
                "Latent space size cannot be larger than output/kinetic parameters feature space size!"
            )

        self.multiheaded = multiheaded

        # Split random key
        key_encoder, key_inflater, key_decoder = random.split(key, num=3)

        if self.multiheaded:
            keys_encoder = random.split(key_encoder, num=3)
            keys_decoder = random.split(key_decoder, num=3)

        # Instantiate encoder component
        if not self.multiheaded:
            self.deep_encoder = DeepComponent(
                component_name="encoder",
                layer_sizes=encoder_params.layer_sizes,
                biases=encoder_params.layer_biases,
                key=key_encoder,
                activation_fn_name=activation_fn_name,
                last_layer_activation=encoder_params.last_layer_activation,
                weight_init_fn=encoder_params.weight_init_fn,
                bias_init_fn=encoder_params.bias_init_fn,
                dropout_rate=encoder_params.dropout_rate,
                weight_init_scale=encoder_params.weight_init_scale,
            )
        else:
            self.deep_encoder = [
                DeepComponent(
                    component_name="encoder",
                    layer_sizes=encoder_params.layer_sizes,
                    biases=encoder_params.layer_biases,
                    key=key,
                    activation_fn_name=activation_fn_name,
                    last_layer_activation=encoder_params.last_layer_activation,
                    weight_init_fn=encoder_params.weight_init_fn,
                    bias_init_fn=encoder_params.bias_init_fn,
                    dropout_rate=encoder_params.dropout_rate,
                    weight_init_scale=encoder_params.weight_init_scale,
                )
                for key in keys_encoder
            ]

        # Instantiate inflater component
        self.deep_inflater = DeepComponent(
            component_name="inflater",
            layer_sizes=inflater_params.layer_sizes,
            biases=inflater_params.layer_biases,
            key=key_inflater,
            activation_fn_name=activation_fn_name,
            last_layer_activation=inflater_params.last_layer_activation,
            weight_init_fn=inflater_params.weight_init_fn,
            bias_init_fn=inflater_params.bias_init_fn,
            dropout_rate=0.0,  # no dropout
            weight_init_scale=inflater_params.weight_init_scale,
        )

        # Instantiate decoder component if two-headed autoencoder
        if reconstruct:
            if not self.multiheaded:
                self.deep_decoder = DeepComponent(
                    component_name="decoder",
                    layer_sizes=decoder_params.layer_sizes,
                    biases=decoder_params.layer_biases,
                    key=key_decoder,
                    activation_fn_name=activation_fn_name,
                    last_layer_activation=decoder_params.last_layer_activation,
                    weight_init_fn=decoder_params.weight_init_fn,
                    bias_init_fn=decoder_params.bias_init_fn,
                    dropout_rate=0.0,  # no dropout
                    weight_init_scale=decoder_params.weight_init_scale,
                )
            else:
                self.deep_decoder = [
                    DeepComponent(
                        component_name="decoder",
                        layer_sizes=decoder_params.layer_sizes,
                        biases=decoder_params.layer_biases,
                        key=key,
                        activation_fn_name=activation_fn_name,
                        last_layer_activation=decoder_params.last_layer_activation,
                        weight_init_fn=decoder_params.weight_init_fn,
                        bias_init_fn=decoder_params.bias_init_fn,
                        dropout_rate=0.0,  # no dropout
                        weight_init_scale=decoder_params.weight_init_scale,
                    )
                    for key in keys_decoder
                ]
        else:
            self.deep_decoder = eqx.nn.Identity()  # no decoder head

    def encode(self, x, key):
        if not self.multiheaded:
            return self.deep_encoder(x, key)
        else:
            keys = random.split(key, num=3)
            inputs = jnp.split(x.reshape(1, -1), 3, axis=1)
            # Mean pool (average) encoded contexts to get a single latent embedding
            return jnp.mean(
                jnp.stack(
                    [
                        encoder(input_arr.reshape(-1), subkey)
                        for encoder, input_arr, subkey in zip(
                            self.deep_encoder, inputs, keys
                        )
                    ],
                    axis=-1,
                ),
                axis=-1,
            )

    def decode(self, x, key):
        if not self.multiheaded:
            key_encoder, key_decoder = random.split(key, num=2)
            return self.deep_decoder(self.encode(x, key_encoder), key_decoder)
        else:
            key_encoder, key_decoder = random.split(key, num=2)
            keys_encoder = random.split(key_encoder, num=3)
            keys_decoder = random.split(key_decoder, num=3)
            inputs = jnp.split(x.reshape(1, -1), 3, axis=1)
            # Concatenate decoded contexts
            return jnp.concatenate(
                [
                    decoder(
                        encoder(input_arr.reshape(-1), subkey_encoder),
                        subkey_decoder,
                    )
                    for encoder, decoder, input_arr, subkey_encoder, subkey_decoder in zip(
                        self.deep_encoder,
                        self.deep_decoder,
                        inputs,
                        keys_encoder,
                        keys_decoder,
                    )
                ],
                axis=-1,
            )

    def inflate(self, x, key):
        key_encoder, key_inflater = random.split(key, num=2)
        return self.deep_inflater(self.encode(x, key_encoder), key_inflater)
