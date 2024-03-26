"""
Materials for a simple linear encoder, and its analytical reverse.
"""

from typing import List, Union

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import config

config.update("jax_enable_x64", True)

class AutoEncoder(eqx.Module):
    """
    A simple linear autoencoder.

    :param features:
        input data for the encoder

    :param n_latent:
        number of latent variables

    :param n_params:
        number of parameters that to which the embedding will be inflated to
    """

    n_features: int = eqx.static_field()  # input size
    n_latent: int = eqx.static_field()  # bottleneck layer size
    n_params: int = eqx.static_field()  # number of kinetic parameters = output layer size
    n_encode_weights: int = eqx.static_field()  # known from input size and bottleneck layer size
    n_inflate_weights: int = eqx.static_field()  # known from bottleneck layer size and output layer size
    n_encoder_pars: int = eqx.static_field()  # known from two above (sum)
    data: np.ndarray = eqx.static_field()
    x_names: List[str] = eqx.static_field()
    orth_reg_strategy: str = eqx.static_field()

    def __init__(
        self,
        features: np.ndarray,
        n_latent: int = 1,
        n_params: int = 12,
        orth_reg_strategy: str = "L2"  # default orthogonal regularisation strategy is L2
    ):
        self.n_features = features.shape[1]
        if n_latent > self.n_features:
            raise ValueError("Latent space size cannot be larger than input feature space size!")
        # assert n_latent <= self.n_features
        elif features.ndim != 2:
            raise ValueError("features expected to be two-dimensional!")
        # assert features.ndim == 2
        # self.data = features
        self.n_latent = n_latent
        self.n_params = n_params
        self.n_encode_weights = self.n_features * self.n_latent
        self.n_inflate_weights = self.n_latent * self.n_params
        self.n_encoder_pars = self.n_encode_weights + self.n_inflate_weights

        # orthogonal regularisation strategy
        self.orth_reg_strategy = orth_reg_strategy

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
