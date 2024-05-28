import equinox as eqx
import jax.nn.initializers as initializers
import jax.random as jr

from jaxtyping import Array
from typing import Optional, Literal, Union


# Dictionary mapping initialization strategies to JAX initializers (or potentially custom functions)
init_fn = {
    "HN": initializers.he_normal(),
    "HU": initializers.he_uniform(),
    "LN": initializers.lecun_normal(),
    "LU": initializers.lecun_uniform(),
    "XN": initializers.glorot_normal(),
    "XU": initializers.glorot_uniform(),
}


class CustomInitLinear(eqx.nn.Linear):
    # same notation as eqx.nn.Linear layers: access with .weight and .bias, enable bias with use_bias
    in_features: Union[int, Literal["scalar"]] = eqx.static_field()
    out_features: Union[int, Literal["scalar"]] = eqx.static_field()
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
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            use_bias=use_bias,
            key=key,
        )

        weight_key, bias_key = jr.split(key)
        self.weight = weight_init(
            weight_key,
            (out_features, in_features)
        )
        self.bias = bias_init(
            bias_key,
            (out_features,)
        ) if self.use_bias else None
