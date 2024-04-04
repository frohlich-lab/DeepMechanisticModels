import equinox as eqx
import jax.nn.initializers as initializers
import jax.numpy as jnp
from jax import config, random


config.update("jax_enable_x64", True)

# Dictionary mapping initialisation strategies to JAX initializer functions (or potentially custom ones)
init_fn = {
    "he_normal": initializers.he_normal(),
    "he_uniform": initializers.he_uniform(),
    "lecun_normal": initializers.lecun_normal(),
    "lecun_uniform": initializers.lecun_uniform(),
    "xavier_normal": initializers.glorot_normal(),
    "xavier_uniform": initializers.glorot_uniform(),
}


class CustomInitLinear(eqx.Module):
    weights: jnp.ndarray
    biases: jnp.ndarray
    use_bias: bool

    def __init__(
            self,
            fan_in,
            fan_out,
            key,
            weight_init,
            bias_init,
            use_bias=False,  # default: no bias
    ):
        weight_key, bias_key = random.split(key)
        self.weights = weight_init(
            weight_key,
            (fan_out, fan_in)
        )
        self.biases = bias_init(
            bias_key,
            (fan_out,)
        ) if use_bias else None
        self.use_bias = use_bias

    def __call__(self, x):
        out = self.weights @ x  # for consistency with equinox.nn.Linear definition
        # https://github.com/patrick-kidger/equinox/blob/main/equinox/nn/_linear.py
        if self.use_bias:
            out += self.biases
        return out
