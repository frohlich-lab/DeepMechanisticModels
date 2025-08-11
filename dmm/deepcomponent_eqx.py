import equinox as eqx
import jax.numpy as jnp
from jax import nn, random
from jaxtyping import Array

from .custom_layers_eqx import CustomInitLinear, init_fn

act_fn_by_name = {
    "identity": eqx.nn.Identity(),  # simply returns the input
    "tanh": jnp.tanh,
    "relu": nn.relu,
    "leaky_relu": nn.leaky_relu,
    "elu": nn.elu,
    "gelu": nn.gelu,
    "swish": nn.swish,
    "softplus": nn.softplus,
}


def generate_layer(
    in_features,
    out_features,
    use_bias,
    key,
    weight_init_fn="eqx_default",
    bias_init_fn="eqx_default",
):
    """
    Produces either a Linear (eqx.nn.Linear) layer or a CustomInitLayer (where
    weight and bias initialisations are performed through a chosen initialiser
    from jax.nn.initializers()).
    """
    # Default option: eqx.nn.Linear (similar to He/Xavier without constant scaling factors)
    if (weight_init_fn == "eqx_default") and (bias_init_fn == "eqx_default"):
        return eqx.nn.Linear(
            in_features=in_features,
            out_features=out_features,
            use_bias=use_bias,
            key=key,
        )
    # Select initializer from jax.nn.initializers() via init_fn dict
    elif (weight_init_fn in init_fn.keys()) and (
        bias_init_fn in init_fn.keys()
    ):
        return CustomInitLinear(
            in_features=in_features,
            out_features=out_features,
            use_bias=use_bias,  # default: no bias
            key=key,
            weight_init=init_fn[weight_init_fn],
            bias_init=init_fn[bias_init_fn],
        )
    else:
        # TODO @GiacomoFabrini consider improving this?!
        # In case of mixed combinations or unknown init_fn names, raise ValueError
        raise ValueError(
            f"Incorrect or unknown {weight_init_fn} or {bias_init_fn}."
        )


class DeepComponent(eqx.Module):
    """
    Deep Component module.

    :param component_name:
        module name: "encoder" / "inflater" / "decoder".

    :param layer_sizes:
        number of units/neurons in DeepComponent layers.

    :param biases:
        list of bool values indicating whether to add a learnable bias to a specific layer or not.
        This makes it possible, for instance, to add a learnable bias to the last layer of the inflater only.

    :param key:
        random key.

    :param activation_fn_name:
        name of the activation function (selected from act_fn_by_name dictionary).

    :param last_layer_activation:
        boolean flag regulating whether to use a non-linear activation function in the last layer.

    :param weight_init_fn (Optional):
        weight initialisation function: either "eqx_default" to select eqx.nn.Linear layers or
        one from jax.nn.initializers to build CustomInitLayer. If latter, needs to be a key
        of `init_fn` dictionary.

    :param bias_init_fn (Optional):
        bias initialisation function: either "eqx_default" to select eqx.nn.Linear layers or
        one from jax.nn.initializers to build CustomInitLayer. If latter, needs to be a key
        of `init_fn` dictionary.

    :param dropout_rate:
        dropout rate.
    """

    layers: list[eqx.nn.Linear | CustomInitLinear]
    component_name: str
    activation_fn_name: str
    last_layer_activation: bool
    dropout_rate: float = eqx.field(static=True)
    dropout_layers: list[eqx.nn.Dropout] | None

    def __init__(
        self,
        component_name,
        layer_sizes,
        biases,
        key,
        activation_fn_name="relu",
        last_layer_activation: bool = "False",
        weight_init_fn="eqx_default",  # use eqx.nn.Linear layers by default
        bias_init_fn="eqx_default",
        dropout_rate=0.0,
    ):
        # component/module name (encoder/inflater/decoder)
        self.component_name = component_name

        # Initialise layers and prepare keys for layer initialisation
        self.layers = []
        layer_keys = random.split(key, num=len(layer_sizes) - 1)

        # Define layer-wise architecture
        # Always specify either both weight_init_fn and bias_init_fn
        # or both as "eqx_default" -- mixed combinations will result in ValueError
        for fan_in, fan_out, key, bias in zip(
            layer_sizes[:-1], layer_sizes[1:], layer_keys, biases
        ):
            self.layers.append(
                generate_layer(
                    in_features=fan_in,
                    out_features=fan_out,
                    use_bias=bias,
                    key=key,
                    weight_init_fn=weight_init_fn,
                    bias_init_fn=bias_init_fn,
                )
            )

        self.dropout_rate = dropout_rate
        # Create list of dropout layers
        self.dropout_layers = []
        for _ in layer_sizes[:-1]:
            self.dropout_layers.append(
                eqx.nn.Dropout(dropout_rate) if self.dropout_rate > 0
                else None
            )

        # activation function
        if activation_fn_name in act_fn_by_name.keys():
            self.activation_fn_name = activation_fn_name
        else:
            raise ValueError(
                f"Unknown activation function: {activation_fn_name}"
            )
        self.last_layer_activation = last_layer_activation

    def __call__(self, x, key):
        if (self.dropout_rate > 0) and (key is not None):
            dropout_keys = random.split(key, num=len(self.dropout_layers))
        elif (self.dropout_rate > 0) and (key is None):
            raise ValueError("Key cannot be None for dropout!")
        a = x
        # Input dropout
        if self.dropout_rate > 0:
            input_dropout = self.dropout_layers[0]
            a = input_dropout(a, key=dropout_keys[0])
        activation = act_fn_by_name[self.activation_fn_name]
        # if more than one layer, applies non-linear activations to all layers but the last
        if len(self.layers) > 1:
            for i, (layer, dropout) in enumerate(zip(self.layers[:-1], self.dropout_layers[1:])):
                a = activation(layer(a))
                # Hidden layer dropout
                if self.dropout_rate > 0:
                    a = dropout(a, key=dropout_keys[i+1])
        # if single layer (self.layers[0] == self.layers[-1]), fully linear behaviour (no non-linear activations)
        return (
            self.layers[-1](a)
            if not self.last_layer_activation
            else activation(self.layers[-1](a))
        )


class KinParamsCombiner(eqx.Module):
    """
    Kinetic Parameters Combiner module: combines cell-line specific kinetic parameter components (deviations)
    with the corresponding global (non-cell-specific) kinetic parameter components (medians).

    :param component_name:
        module name: "encoder" / "inflater" / "decoder".

    :param n_global_kin_params:
        number of global kinetic parameters.
    """

    learned_global_kin_params: Array
    component_name: str = eqx.field(static=True)

    def __init__(self, component_name, n_global_kin_params):
        # Initialise the learned global (non-cell-specific) parameters to zeros (in log10 scale, so ones in linear)
        self.component_name = component_name
        self.learned_global_kin_params = jnp.zeros(
            shape=(n_global_kin_params,)
        )

    def __call__(self, x):
        # input x is the inflated parameter deviations
        # output is the concatenation of the global parameters and the flattened specific ones
        return jnp.concatenate([self.learned_global_kin_params, x.flatten()])
