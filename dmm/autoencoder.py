from typing import List, Optional

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pypesto.petab
from jax.config import config
from sklearn.decomposition import PCA

from . import MODEL_FEATURE_PREFIX
from .encoder import AutoEncoder
from .petab_subproblem import load_petab
from .problem import Problem

config.update("jax_enable_x64", True)


class DeepMechanisticModel(AutoEncoder):
    data_name: str = eqx.static_field()
    pathway_name: str = eqx.static_field()
    features: np.ndarray = eqx.static_field()
    features_pca: np.ndarray = eqx.static_field()
    pca: PCA = eqx.static_field()
    n_model_inputs: int = eqx.static_field()
    n_kin_params: int = eqx.static_field()
    n_samples: int = eqx.static_field()
    sample_names: List[str] = eqx.static_field()
    x_names: List[str] = eqx.static_field()
    feature_cols: List[str] = eqx.static_field()
    # general PetabImporter compared to old PetabImporterPysb
    petab_importer: pypesto.petab.PetabImporter = eqx.static_field()
    pypesto_subproblem: pypesto.Problem = eqx.static_field()

    def __init__(
        self,
        problem: Problem,
        dataset: str,
        n_latent: int,
        measurement_table: pd.DataFrame,
        observable_table: pd.DataFrame,
        condition_table: pd.DataFrame,
        features: pd.DataFrame,
        n_threads=1,
        pca: Optional[PCA] = None,
    ):
        """
        loads the mechanistic model as theano operator with loss as output and
        decoder output as input

        :param pathway_name:
            name of pathway to use for model

        :param n_latent:
            number of nodes in the hidden layer of the encoder

        :param l1reg:
            currently this parameter only influences the strength of l1
            regularization on the inflate layer (the respective laplace
            prior has its standard deviation defined based on the value of
            this parameter). For bounded inflate functions, this parameter
            is also intended to rescale the inputs accordingly.

        """
        self.data_name = dataset
        self.pathway_name = problem.pathway_name

        # subset samples

        self.petab_importer = load_petab(
            problem=problem,
            dataset=dataset,
            measurement_table=measurement_table,
            condition_table=condition_table,
            observable_table=observable_table,
            samples=list(features.index),
        )

        self.pypesto_subproblem = self.petab_importer.create_problem()

        # extract sample names, ordering of those is important since samples
        # must match when reshaping the inflated matrix
        petab_samples = []
        for name in self.pypesto_subproblem.x_names:
            if not name.startswith(MODEL_FEATURE_PREFIX):
                continue

            sample = name.split("__")[-1]
            if sample not in petab_samples and sample in features.index:
                petab_samples.append(sample)

        self.features = features.loc[petab_samples, :].values

        if pca is None:
            self.pca = PCA(n_components=n_latent).fit(self.features)
        else:
            self.pca = pca
        self.features_pca = self.pca.transform(self.features)

        self.n_samples, self.n_features = self.features_pca.shape
        self.n_model_inputs = int(
            sum(
                name.startswith(MODEL_FEATURE_PREFIX)
                for name in self.pypesto_subproblem.x_names
            )
            / self.n_samples
        )
        self.n_kin_params = (
            self.pypesto_subproblem.dim - self.n_model_inputs * self.n_samples
        )

        self.sample_names = list(features.index)
        self.feature_cols = [
            f"PC{i}" for i in range(self.features_pca.shape[1])
        ]
        super().__init__(
            features=self.features,
            n_latent=n_latent,
            n_params=self.n_model_inputs,
        )

        problem.apply_objective_settings(
            self.pypesto_subproblem.objective, n_threads=n_threads
        )

        self.x_names = self.x_names + [
            name
            for ix, name in enumerate(self.pypesto_subproblem.x_names)
            if not name.startswith(MODEL_FEATURE_PREFIX)
            and ix in self.pypesto_subproblem.x_free_indices
        ]

    def embedding(self, params: np.ndarray) -> jnp.ndarray:
        encode_weights, inflate_weights, kin_params = jnp.split(
            params,
            np.array(
                (
                    self.n_encode_weights,
                    self.n_inflate_weights + self.n_encode_weights,
                )
            ),
        )
        return jnp.concatenate(
            [
                kin_params,
                self.inflate_params(
                    self.encode(encode_weights), inflate_weights
                ).flatten(),
            ]
        )

    def orth_encode_reg(self, params: jnp.ndarray, scale: float = 1.0):
        """
        Orthogonal regularization of the encoder weights.
        """
        encode_weights, _, _ = jnp.split(
            params,
            np.array(
                (
                    self.n_encode_weights,
                    self.n_encoder_pars,
                )
            ),
        )
        w = jnp.reshape(encode_weights, (self.n_features, self.n_latent))
        m = jnp.dot(w.T, w)
        return scale * jnp.mean(jnp.abs(m - jnp.eye(self.n_latent))**2)

    def orth_inflate_reg(self, params: jnp.ndarray, scale: float = 1.0):
        """
        Orthogonal regularization of the inflate weights.
        """
        _, inflate_weights, _ = jnp.split(
            params,
            np.array(
                (
                    self.n_encode_weights,
                    self.n_inflate_weights + self.n_encode_weights,
                )
            ),
        )
        w = jnp.reshape(inflate_weights, (self.n_latent, self.n_params))
        m = jnp.dot(w, w.T)
        return scale * jnp.mean(jnp.abs(m - jnp.diag(jnp.diag(m)))**2)

    def l1_inflate_reg(self, params: jnp.ndarray, scale: float = 1.0):
        """
        l1 regularization of the inflate output.
        """
        _, inflate_weights, _ = jnp.split(
            params,
            np.array(
                (
                    self.n_encode_weights,
                    self.n_inflate_weights + self.n_encode_weights,
                )
            ),
        )
        return scale * jnp.mean(jnp.abs(inflate_weights))

    def l1_encode_reg(self, params: jnp.ndarray, scale: float = 1.0):
        """
        l1 regularization of the inflate output.
        """
        encode_weights, _, _ = jnp.split(
            params,
            np.array(
                (
                    self.n_encode_weights,
                    self.n_inflate_weights + self.n_encode_weights,
                )
            ),
        )
        return scale * jnp.mean(jnp.abs(encode_weights))
