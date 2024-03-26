from typing import List, Optional

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pypesto.petab
from jax import config
from sklearn.decomposition import PCA

from . import MODEL_FEATURE_PREFIX
from dmm.encoder_eqx import TwoHeadedDeepAutoencoder
from .petab_subproblem import load_petab
from .problem import Problem
# from optax import power_iteration


config.update("jax_enable_x64", True)

class DeepMechanisticModel_v2(TwoHeadedDeepAutoencoder):
    # TODO @GiacomoFabrini check attributes! What is missing?
    data_name: str = eqx.static_field()
    pathway_name: str = eqx.static_field()
    features: np.ndarray = eqx.static_field()
    features_pca: np.ndarray = eqx.static_field()
    pca: PCA = eqx.static_field()
    n_model_inputs: int = eqx.static_field()
    n_kin_params: int = eqx.static_field()
    n_samples: int = eqx.static_field()
    orth_reg_strategy: str = eqx.static_field()
    sample_names: List[str] = eqx.static_field()
    x_names: List[str] = eqx.static_field()
    feature_cols: List[str] = eqx.static_field()
    petab_importer: pypesto.petab.PetabImporter = eqx.static_field()
    pypesto_subproblem: pypesto.Problem = eqx.static_field()
    encoder_layer_sizes: List[int] = eqx.static_field()
    inflater_layer_sizes: List[int] = eqx.static_field()
    decoder_layer_sizes: List[int] = eqx.static_field()
    activation_fn_name: str =  eqx.static_field()
    reconstruct: bool =  eqx.static_field()

    def __init__(
        self,
        problem: Problem,
        dataset: str,
        n_latent: int,
        n_params: int,
        encoder_layer_sizes: List,
        inflater_layer_sizes: List,
        orth_reg_strategy: str = "L2",
        measurement_table: pd.DataFrame,
        observable_table: pd.DataFrame,
        condition_table: pd.DataFrame,
        features: pd.DataFrame,
        n_threads=1,
        pca: Optional[PCA] = None,
        activation_fn_name: str = "relu",  # default activation function = Rectified Linear Unit
        reconstruct: bool = False, # whether to add decoder head (single head by default)
    ):
        """
        loads the mechanistic model as theano operator with loss as output and
        decoder output as input

        :param pathway_name:
            name of pathway to use for model

        :param n_latent:
            number of nodes in the hidden layer of the encoder

        :param n_params:
        number of parameters to which the embedding will be inflated to ???!

        :param encoder_layer_sizes:
            list of layer sizes for encoder component (and decoder component, in reverse)
            Needed to define encoder and, potentially, decoder modules.

        :param inflater_layer_sizes:
            list of layer sizes for inflater component
            Needed to define inflater module.

        :param key:
            PRNG key.

        :param activation_fn_name:
            choice of activation function
            Default: ReLU

        :param reconstruct:
            boolean flag. If set to True, adds a second, autoencoding head to the network
            (encoder->decoder) on top of the first head (encoder->inflater)
            Default: single head (False)

        :param orth_reg_strategy:
        orthogonal regularisation strategy to be used: L1 vs L2 (default)

        """

        self.data_name = dataset
        self.pathway_name = problem.pathway_name

        # Set bottleneck layer size
        self.n_latent = n_latent
        # Set inflater layer output size
        self.n_params = n_params

        # set regularisation strategy
        self.orth_reg_strategy = orth_reg_strategy

        # set reconstruct flag
        self.reconstruct = reconstruct

        # define layer sizes
        self.encoder_layer_sizes = encoder_layer_sizes
        self.inflater_layer_sizes = inflater_layer_sizes

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


        # Make sure encoder output size and inflater input size match n_latent
        if encoder_layer_sizes[-1] != self.n_latent:
            raise ValueError("Size of encoder bottleneck layer must match n_latent!")
        if inflater_layer_sizes[0] != self.n_latent:
            raise ValueError("Size of inflater input layer must match n_latent!")
        # Make sure inflater output size matches n_params
        if inflater_layer_sizes[-1] != self.n_params:
            raise ValueError("Size of inflater output layer must match number of kinetic parameters!")

        # Initialise TwoHeadedDeepAutoencoder
        super().__init__(
            features=self.features,
            encoder_layer_sizes=self.encoder_layer_sizes,
            inflater_layer_sizes=self.inflater_layer_sizes,
            key=key,
            activation_fn_name=activation_fn_name,
            orth_reg_strategy=self.orth_reg_strategy,
            reconstruct=self.reconstruct,
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

    # TODO @GiacomoFabrini ask Fabian about this?!
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

    # TODO @GiacomoFabrini code single loss function incorporating all 4 components + reconstruction loss
    def loss_fn(self,
                l1reg_encode, l1reg_inflate, l1reg_decode,
                oreg_encode, oreg_inflate, oreg_decode,
                ):
        encode_weights = eqx.filter(model.deep_encoder, eqx.is_array)
        inflate_weights = eqx.filter(model.deep_inflater, eqx.is_array)

        l1reg_encode_reg_loss = l1reg_encode * jnp.mean(jnp.abs(encode_weights))
        l1reg_inflate_reg_loss = l1reg_inflate * jnp.mean(jnp.abs(inflate_weights))


        if self.reconstruct:
            decode_weights = eqx.filter(model.deep_decoder, eqx.is_array)
            l1reg_decode_reg_loss = l1reg_encode * jnp.mean(jnp.abs(decode_weights))

        if self.reconstruct:
            return l1reg_encode_reg_loss + l1reg_inflate_reg_loss
        else:
            return l1reg_encode_reg_loss + l1reg_inflate_reg_loss

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
        if self.orth_reg_strategy == "L1":
            # L1 norm - originally used in Fabian's runs
            return scale * jnp.mean(jnp.abs(m - jnp.eye(self.n_latent)))
        elif self.orth_reg_strategy == "L2":
            # L2 norm
            return scale * jnp.mean(jnp.abs(m - jnp.eye(self.n_latent))**2)
        else:
            raise ValueError(f"Invalid orth_reg_strategy: {self.orth_reg_strategy}")
        # SRIP - minimise max singular value/eigenvalue of the same matrix above
        # Reference: "Can we gain more from orthogonality regularizations in training Deep CNNs?"
        # Reference DOI: https://doi.org/10.48550/arXiv.1810.09102
        # Implementation: https://github.com/google-deepmind/optax/blob/main/optax/_src/linear_algebra.py
        # Unfortunately, optax.power_iteration() appears not to be differentiable!
        # _, max_eig = power_iteration(m - jnp.eye(self.n_latent))
        # return scale * max_eig


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
        if self.orth_reg_strategy == "L1":
            # L1 norm - originally used in Fabian's runs
            return scale * jnp.mean(jnp.abs(m - jnp.diag(jnp.diag(m))))
        elif self.orth_reg_strategy == "L2":
            # L2 norm
            return scale * jnp.mean(jnp.abs(m - jnp.diag(jnp.diag(m)))**2)
        else:
            raise ValueError(f"Invalid orth_reg_strategy: {self.orth_reg_strategy}")
        # SRIP - minimise max singular value/eigenvalue of the same matrix above
        # Reference: "Can we gain more from orthogonality regularizations in training Deep CNNs?"
        # Reference DOI: https://doi.org/10.48550/arXiv.1810.09102
        # Implementation: https://github.com/google-deepmind/optax/blob/main/optax/_src/linear_algebra.py
        # Unfortunately, optax.power_iteration() appears not to be differentiable!
        # _, max_eig = power_iteration(m - jnp.diag(jnp.diag(m)))
        # return scale * max_eig

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