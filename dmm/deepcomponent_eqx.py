import equinox as eqx
import jax.numpy as jnp

from dmm.custom_layers_eqx import (
    CustomInitLinear,
    init_fn
)
from jax import config, nn, random
from typing import (
    Callable,
    List,
)

config.update("jax_enable_x64", True)

# TODO @GiacomoFabrini LATER: consider adding eqx.nn.PReLU (parametric leaky ReLU)
act_fn_by_name = {
    "identity": eqx.nn.Identity(),  # simply returns the input
    "tanh": jnp.tanh,
    "relu": nn.relu,
    "leaky_relu": nn.leaky_relu,
    "elu": nn.elu,
    "gelu": nn.gelu,
    "swish": nn.swish,
}


class DeepComponent(eqx.Module):
    """
    Deep Component module.

    :param component_name:
        module name: "encoder" / "inflater" / "decoder".

    :param layer_sizes:
        number of units/neurons in DeepComponent layers.

    :param biases:
        list of bool values indicating whether to add a learnable bias to a specific layer or not.
        This enables, for instance, to add a learnable bias array/vector to the last layer of the inflater only.

    :param key:
        random key.

    :param activation_fn_name:
        name of the activation function (selected from act_fn_by_name dictionary).

    :param weight_init_fn (Optional):
        weight initialisation function: either "eqx_default" to select eqx.nn.Linear layers or
        one from jax.nn.initializers to build CustomInitLayer. If latter, needs to be a key
        of `init_fn` dictionary.

    :param bias_init_fn (Optional):
        bias initialisation function: either "eqx_default" to select eqx.nn.Linear layers or
        one from jax.nn.initializers to build CustomInitLayer. If latter, needs to be a key
        of `init_fn` dictionary.
    """

    component_name: str = eqx.static_field()
    x_names: List[str] = eqx.static_field()
    layers: List[eqx.nn.Linear]
    activation: Callable

    def __init__(
        self,
        component_name,
        layer_sizes,
        biases,
        key,
        activation_fn_name,
        weight_init_fn="eqx_default",  # use eqx.nn.Linear layers by default
        bias_init_fn="eqx_default",
    ):
        # component/module name (encoder/inflater/decoder)
        self.component_name = component_name

        # Initialise layers and x_names
        self.layers = []
        self.x_names = []

        # Prepare keys for layer initialisation
        layer_keys = random.split(key, num=len(layer_sizes)-1)

        # Define layer-wise architecture
        # Always specify either both weight_init_fn and bias_init_fn
        # or both as "eqx_default" -- mixed combinations will result in ValueError
        if (weight_init_fn == "eqx_default") and (bias_init_fn == "eqx_default"):
            # Default option: eqx.nn.Linear (similar to He/Xavier without constant scaling factors)
            for layer_num, (
                    (fan_in, fan_out),
                    (key, bias)
            ) in enumerate(
                    zip(
                        zip(layer_sizes[:-1], layer_sizes[1:]),
                        zip(layer_keys, biases)
                    )
            ):
                self.layers.append(
                    eqx.nn.Linear(
                        in_features=fan_in,
                        out_features=fan_out,
                        use_bias=bias,
                        key=key
                    )
                )
                # TODO @GiacomoFabrini: do we need these? (pt.1)
                self.x_names.extend(
                    [
                        f"{self.component_name}_{layer_num}_{ind}_weight"
                        for ind in range(fan_in * fan_out)
                    ]
                )
        elif (weight_init_fn in init_fn.keys()) and (bias_init_fn in init_fn.keys()):
            # Select initializer from jax.nn.initializers() via init_fn dict
            for layer_num, (
                    (fan_in, fan_out),
                    (key, bias)
            ) in enumerate(
                zip(
                    zip(layer_sizes[:-1], layer_sizes[1:]),
                    zip(layer_keys, biases)
                )
            ):
                self.layers.append(
                    CustomInitLinear(
                        fan_in=fan_in,
                        fan_out=fan_out,
                        use_bias=bias,  # default: no bias
                        key=key,
                        weight_init=init_fn[weight_init_fn],
                        bias_init=init_fn[bias_init_fn],
                    )
                )
                # TODO @GiacomoFabrini pt.2 from above
                self.x_names.extend(
                    [
                        f"{self.component_name}_{layer_num}_{ind}_weight"
                        for ind in range(fan_in * fan_out)
                    ]
                )
        else:
            # In case of mixed combinations or unknown init_fn names, raise ValueError
            raise ValueError(f"Incorrect or unknown {weight_init_fn} or {bias_init_fn}.")

        # activation function
        if activation_fn_name in act_fn_by_name.keys():
            self.activation = act_fn_by_name[activation_fn_name]
        else:
            raise ValueError(f"Unknown activation function: {activation_fn_name}")

    def __call__(self, x):
        a = x
        # if more than one layer, applies non-linear activations to all layers but the last
        if len(self.layers) > 1:
            for layer in self.layers[:-1]:
                a = self.activation(layer(a))
        # if single layer (self.layers[0] == self.layers[-1]), fully linear behaviour (no non-linear activations)
        return self.layers[-1](a)
