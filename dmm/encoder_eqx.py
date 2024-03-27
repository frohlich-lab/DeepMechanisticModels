from typing import Callable, List, Sequence
# from typing import List, Union
import equinox as eqx
# import jax.numpy as jnp
import numpy as np
from jax import config, nn, random

config.update("jax_enable_x64", True)

# TODO @GiacomoFabrini consider adding eqx.nn.PReLU as parametric leaky ReLU
    # add tanh option
act_fn_by_name = {
    "identity": eqx.nn.Identity(),  # simply returns the input
    "relu": nn.relu,
    "leakyrelu": nn.leaky_relu,
    "elu": nn.elu,
    "gelu": nn.gelu,
    "swish": nn.swish,
}

class DeepComponent(eqx.Module):
    """
    Deep Component module: can be Encoder/Inflater/Decoder.

    :param component_name:
        module name: "encoder" / "inflater" / "decoder"

    :param layer_sizes:
        number of units/neurons in DeepComponent layers

    :param key:
        random key

    :param activation_fn_name:
        name of the activation function (selected from act_fn_by_name dictionary)

    """

    component_name: str = eqx.static_field()
    layers: Sequence[eqx.nn.Linear]
    x_names: List[str] = eqx.static_field()
    activation: Callable

    def __init__(
        self,
        component_name,
        layer_sizes,
        key,
        activation_fn_name,
    ):
        # component/module name (encoder/inflater/decoder)
        self.component_name = component_name
        # Initialise layers and x_names
        self.layers = []
        self.x_names = []
        # Prepare keys for layer initialisation
        layer_keys = random.split(key, num=len(layer_sizes)-1)
        # Define layer-wise architecture
        for layer_num, (
                (fan_in, fan_out),
                key
        ) in enumerate(
                zip(
                    zip(layer_sizes[:-1], layer_sizes[1:]),
                    layer_keys
                )
        ):
            self.layers.append(
                eqx.nn.Linear(fan_in, fan_out, use_bias=False, key=key)
            )
            self.x_names.extend(
                [
                    f"{self.component_name}_{layer_num}_{ind}_weight"
                    for ind in range(fan_in * fan_out)
                ]
            )

        # TODO @GiacomoFabrini if we do NOT need self.x_names, change to code below
        # self.layers = [
        #     eqx.nn.Linear(fan_in, fan_out, use_bias=False, key=key)
        #     for key, (fan_in, fan_out) in zip(
        #         layer_keys, zip(layer_sizes[:-1], layer_sizes[1:])
        #     )
        # ]

        # activation function
        if activation_fn_name in act_fn_by_name.keys():
            self.activation = act_fn_by_name[activation_fn_name]
        else:
            raise ValueError(f"Unknown activation function: {activation_fn_name}")

    def __call__(self, x):
        a = x
        # if more than one layer (deep architecture), applies non-linearities to all layers but the last
        # if single layer, does not apply any non-linearities
        if len(self.layers) > 1:
            for layer in self.layers[:-1]:
                a = self.activation(layer(a))
        return self.layers[-1](a)  # if only one layer, self.layers[0] == self.layers[-1]


class TwoHeadedDeepAutoencoder(eqx.Module):
    """
    A potentially deep, non-linear and two-headed Autoencoder.
        First head: encoder -> inflater (kinetic parameters of ODE model).
        Second head: encoder -> decoder (input reconstruction): proper Autoencoder behaviour.

    :param features:
        input data for the encoder

    :param encoder_layer_sizes:
        list of layer sizes for encoder component (and decoder component, in reverse)

    :param inflater_layer_sizes:
        list of layer sizes for inflater component

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

    """
    n_features: int = eqx.static_field()  # input size
    n_latent: int = eqx.static_field()  # bottleneck layer size
    n_params: int = eqx.static_field()  # number of kinetic parameters = output layer size
    n_encode_weights: int = eqx.static_field()  # known from input size and bottleneck layer size
    n_inflate_weights: int = eqx.static_field()  # known from bottleneck layer size and output layer size
    n_encoder_pars: int = eqx.static_field()  # known from two above (sum)
    data: np.ndarray = eqx.static_field()
    x_names: List[str] = eqx.static_field()

    deep_encoder: DeepComponent
    deep_inflater: DeepComponent
    deep_decoder: eqx.Module | None  # could be None if not reconstructing (hence not using DeepComponent)
    # can also set deep_decoder: DeepComponent if changing to eqx.nn.Identity() rather than None down below
    orth_reg_strategy: str = eqx.static_field()
    reconstruct: bool

    # TODO @GiacomoFabrini: do we need features, self.data, self.n_latent, self.n_params,
        # self.n_encode_weights, self.n_inflate_weights, self.n_decode_weights, self.n_encoder_pars? If not, remove!
    def __init__(
        self,
        features: np.ndarray,
        encoder_layer_sizes: List,  # decoder layers are just going to be encoder_layer_sizes mirrored
        inflater_layer_sizes: List,
        key,
        activation_fn_name: str = "relu", # default activation_fn_name is ReLU if more than one layer is present
        orth_reg_strategy: str = "L2",  # default orthogonal regularisation strategy is L2
        reconstruct: bool = False  # current default behaviour uses a single head (encoder->inflater),
    ):

        self.n_features = features.shape[1]
        # input size (self.n_features) must match input layer size of encoder (equal to decoder output size, if any)
        if encoder_layer_sizes[0] != self.n_features:
            raise ValueError("Input layer size must be the same as input feature space size!")
        # encoder layers must shrink towards bottleneck/latent representation
        elif encoder_layer_sizes[-1] > encoder_layer_sizes[0]:
            raise ValueError("Latent space size cannot be larger than input feature space size!")
        elif features.ndim != 2:
            raise ValueError("features expected to be two-dimensional!")
        # Make sure encoder output size matches inflater input size (interface between components/modules)
        elif encoder_layer_sizes[-1] != inflater_layer_sizes[0]:
            raise ValueError("Encoder output size must match inflater input size!")

        self.data = features
        # self.n_latent = encoder_layer_sizes[-1]
        # self.n_params = inflater_layer_sizes[-1]

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
            key=key_encoder,
            activation_fn_name=activation_fn_name,
        )
        self.n_encode_weights = len(self.deep_encoder.x_names)

        # Instantiate inflater component
        self.deep_inflater = DeepComponent(
            component_name="inflater",
            layer_sizes=inflater_layer_sizes,
            key=key_inflater,
            activation_fn_name=activation_fn_name,
        )
        self.n_inflate_weights = len(self.deep_inflater.x_names)

        if self.reconstruct:
            decoder_layer_sizes = encoder_layer_sizes[::-1]
            self.deep_decoder = DeepComponent(
                component_name="decoder",
                layer_sizes=decoder_layer_sizes,
                key=key_decoder,
                activation_fn_name=activation_fn_name,
            )
            self.n_decode_weights = len(self.deep_decoder.x_names)

            self.n_encoder_pars = self.n_encode_weights + self.n_inflate_weights + self.n_decode_weights
            self.x_names = self.deep_encoder.x_names + self.deep_inflater.x_names + self.deep_decoder.x_names

        else:
            self.deep_decoder = None  # can potentially make this eqx.nn.Identity() to set deep_decoder to DeepComponent

            self.n_encoder_pars = self.n_encode_weights + self.n_inflate_weights
            self.x_names = self.deep_encoder.x_names + self.deep_inflater.x_names

    def __call__(self, x):
        encoded = self.deep_encoder(x)
        inflated = self.deep_inflater(encoded)
        # If using decoding head, pass encoding through decoder, else just leave second output blank (None)
        decoded = self.deep_decoder(encoded) if self.reconstruct else None
        return inflated, decoded