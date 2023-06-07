"""
Materials for a simple linear encoder, and its analytical reverse.
"""

from typing import List, Union

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax.config import config

config.update("jax_enable_x64", True)


class AutoEncoder(eqx.Module):
    """
    A simple linear autoencoder.

    :param input_data:
        input data for the encoder

    :param n_latent:
        number of latent variables

    :param n_params:
        number of parameters that to which the embedding will be inflated to
    """

    n_features: int = eqx.static_field()
    n_latent: int = eqx.static_field()
    n_params: int = eqx.static_field()
    n_encode_weights: int = eqx.static_field()
    n_inflate_weights: int = eqx.static_field()
    n_encoder_pars: int = eqx.static_field()
    data: np.ndarray = eqx.static_field()
    x_names: List[str] = eqx.static_field()

    def __init__(
        self, input_data: np.ndarray, n_latent: int = 1, n_params: int = 12
    ):
        self.n_features = input_data.shape[1]
        assert n_latent <= self.n_features
        assert input_data.ndim == 2
        self.data = input_data
        self.n_latent = n_latent
        self.n_params = n_params
        self.n_encode_weights = self.n_features * self.n_latent
        self.n_inflate_weights = self.n_latent * self.n_params
        self.n_encoder_pars = self.n_encode_weights + self.n_inflate_weights

        # self.par_modulation_scale = par_modulation_scale

        self.x_names = [
            f"encoder_{iw}_weight" for iw in range(self.n_encode_weights)
        ] + [f"inflate_{iw}_weight" for iw in range(self.n_inflate_weights)]

    def encode(self, parameters: jnp.ndarray):
        """
        Run the input through the encoder.

        :param parameters:
            parametrization of full autoencoder
        """
        weights = jnp.reshape(parameters, (self.n_features, self.n_latent))
        return jnp.dot(self.data, weights)

    def encode_sample(self, sample: np.ndarray, parameters: jnp.ndarray):
        """
        Run the input through the encoder.

        :param parameters:
            parametrization of full autoencoder
        """
        weights = jnp.reshape(parameters, (self.n_features, self.n_latent))
        return jnp.dot(sample, weights)

    def decode(
        self,
        embedding: Union[np.ndarray, jnp.ndarray],
        parameters: jnp.ndarray,
    ):
        """
        Run the input through the decoder.

        :param parameters:
            parametrization of full autoencoder
        """
        weights = jnp.reshape(parameters, (self.n_features, self.n_latent))
        return jnp.dot(embedding, jnp.linalg.pinv(weights.T).T)

    def inflate_params(
        self,
        embedding: Union[np.ndarray, jnp.ndarray],
        parameters: jnp.ndarray,
    ):
        """Inflate the input to parameters (partial parameter vector)"""
        weights = jnp.reshape(parameters, (self.n_latent, self.n_params))
        return jnp.dot(embedding, weights)

    def encode_params(self, parameters: jnp.ndarray):
        """
        Run the encoder and then inflate to parameters.

        :param parameters:
            parametrization of full autoencoder
        """
        return self.inflate_params(self.encode(parameters), parameters)
