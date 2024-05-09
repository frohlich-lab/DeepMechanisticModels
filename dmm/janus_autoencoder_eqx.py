import equinox as eqx
import jax.numpy as jnp

from common import ModuleParams
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

    :param encoder_params:
        Parameters for the encoder module.

    :param inflater_params:
        Parameters for the inflater module.

    :param decoder_params:
        Parameters for the decoder module.

    -- KineticParametersCombiner PARAMS
    :param n_global_kin_params:
        Number of global kinetic parameters (ODE non cell-line-specific model parameters).

    :param n_inflated_specific_kin_params:
        Number of kinetic parameters to inflate to (inflater output size).

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

    n_inflated_specific_kin_params: int = eqx.static_field()  # inflater output size
    n_global_kin_params: int = eqx.static_field()
    x_names: List[str] = eqx.static_field()
    reconstruct: bool = eqx.static_field()

    deep_encoder: DeepComponent
    deep_inflater: DeepComponent
    deep_decoder: eqx.Module
    kin_params_combiner: KinParamsCombiner

    # TODO @GiacomoFabrini do we need self.x_names?
    def __init__(
            self,
            n_inflated_specific_kin_params: int,
            n_global_kin_params: int,
            encoder_params: ModuleParams,
            inflater_params: ModuleParams,
            decoder_params: ModuleParams,
            key,
            activation_fn_name: str,  # default activation_fn_name is ReLU if more than one layer is present
            reconstruct: bool,  # current default behaviour uses a single head (encoder->inflater),
    ):

        self.n_inflated_specific_kin_params = n_inflated_specific_kin_params
        self.n_global_kin_params = n_global_kin_params

        # CHECKS
        # encoder layers must shrink towards bottleneck/latent representation
        if encoder_params.layer_sizes[-1] > encoder_params.layer_sizes[0]:
            raise ValueError("Latent space size cannot be larger than input feature space size!")
        # TODO @GiacomoFabrini: need to implement this check in training/train - features.ndim not available here
        # elif features.ndim != 2:
        #     raise ValueError("features expected to be two-dimensional!")

        # Set reconstruct flag
        self.reconstruct = reconstruct

        # Split random key
        key_encoder, key_inflater, key_decoder = random.split(key, num=3)

        # Instantiate encoder component
        self.deep_encoder = DeepComponent(
            component_name="encoder",
            layer_sizes=encoder_params.layer_sizes,
            biases=encoder_params.layer_biases,
            key=key_encoder,
            activation_fn_name=activation_fn_name,
            weight_init_fn=encoder_params.weight_init_fn,
            bias_init_fn=encoder_params.bias_init_fn,
        )

        # Instantiate inflater component
        self.deep_inflater = DeepComponent(
            component_name="inflater",
            layer_sizes=inflater_params.layer_sizes,
            biases=inflater_params.layer_biases,
            key=key_inflater,
            activation_fn_name=activation_fn_name,
            weight_init_fn=inflater_params.weight_init_fn,
            bias_init_fn=inflater_params.bias_init_fn,
        )

        # Instantiate global kinetic parameters component
        self.kin_params_combiner = KinParamsCombiner(
            component_name='kin_params_combiner',
            # TODO @GiacomoFabrini reinstate if reinstating .learned_median_params
            # n_inflated_specific_kin_params=n_inflated_specific_kin_params,
            n_global_kin_params=n_global_kin_params
        )

        self.x_names = self.deep_encoder.x_names + self.deep_inflater.x_names + self.kin_params_combiner.x_names

        if self.reconstruct:
            self.deep_decoder = DeepComponent(
                component_name="decoder",
                layer_sizes=decoder_params.layer_sizes,
                biases=decoder_params.layer_biases,
                key=key_decoder,
                activation_fn_name=activation_fn_name,
                weight_init_fn=decoder_params.weight_init_fn,
                bias_init_fn=decoder_params.bias_init_fn,
            )

            self.x_names += self.deep_decoder.x_names

        else:
            self.deep_decoder = eqx.nn.Identity()  # no decoder head

    def __call__(self, x):
        encoded = self.deep_encoder(x)
        inflated = self.deep_inflater(encoded)
        # If using decoding head, pass encoding through decoder, else just leave second output blank (None)
        decoded = self.deep_decoder(encoded) if self.reconstruct else None
        # augmented_inflated = self.kin_params_combiner(inflated)
        # augmented_inflated: concatenation of global kinetic parameters
        # and flattened cell-line-specific params (inflated deviations + learned medians)
        # Removed for now after introducing jax.vmap to handle multiple training examples (cell-lines)
        return inflated, decoded
