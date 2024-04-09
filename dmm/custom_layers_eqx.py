import equinox as eqx
import jax.nn.initializers as initializers
# import jax.numpy as jnp
from jax import config, random
from jaxtyping import Array
from typing import Optional, Union


config.update("jax_enable_x64", True)

# Dictionary mapping initialization strategies to JAX initializers (or potentially custom functions)
init_fn = {
    "he_normal": initializers.he_normal(),
    "he_uniform": initializers.he_uniform(),
    "lecun_normal": initializers.lecun_normal(),
    "lecun_uniform": initializers.lecun_uniform(),
    "xavier_normal": initializers.glorot_normal(),
    "xavier_uniform": initializers.glorot_uniform(),
}


class CustomInitLinear(eqx.Module):
    # same notation as eqx.nn.Linear layers: access with .weight and .bias, enable bias with use_bias
    weight: Array
    bias: Optional[Array]
    use_bias: bool = eqx.static_field()

    def __init__(
            self,
            in_features,  # again, follows the same notation as eqx.nn.Linear
            out_features,
            key,
            weight_init,
            bias_init,
            use_bias=False,  # default: no bias
    ):
        weight_key, bias_key = random.split(key)
        self.use_bias = use_bias
        self.weight = weight_init(
            weight_key,
            (out_features, in_features)
        )
        self.bias = bias_init(
            bias_key,
            (out_features,)
        ) if self.use_bias else None

    def __call__(self, x):
        out = self.weight @ x  # for consistency with equinox.nn.Linear definition
        # doc: https://github.com/patrick-kidger/equinox/blob/main/equinox/nn/_linear.py
        if self.use_bias:
            out += self.bias
        return out
