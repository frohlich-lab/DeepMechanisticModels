from typing import List

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pypesto.petab
from jax import config
from . import MODEL_FEATURE_PREFIX
from dmm.janus_autoencoder_eqx import TwoHeadedDeepAutoencoder
from .petab_subproblem import load_petab
from .problem import Problem

# TODO @GiacomoFabrini idea: pretrain whole network on n_params x n_samples
# matrix coming from ODE pretraining
# then train end-to-end differentiable DMM
# can pretrain encoder-inflater or encoder-inflater-decoder
config.update("jax_enable_x64", True)


def init_biases(biases, layer_sizes, component_name):
    if biases is None:
        biases = [False] * len(layer_sizes)
    elif len(biases) != len(layer_sizes):
        raise ValueError(f"{component_name}: biases must have the same length as layer_sizes")
    return biases


class DeepMechanisticModel(TwoHeadedDeepAutoencoder):
    data_name: str = eqx.static_field()
    pathway_name: str = eqx.static_field()
    n_features: int = eqx.static_field()
    n_samples: int = eqx.static_field()
    n_model_inputs: int = eqx.static_field()
    n_kin_params: int = eqx.static_field()
    sample_names: List[str] = eqx.static_field()
    x_names: List[str] = eqx.static_field()
    petab_importer: pypesto.petab.PetabImporter = eqx.static_field()
    pypesto_subproblem: pypesto.Problem = eqx.static_field()
    encoder_params_dict: dict = eqx.static_field()
    inflater_params_dict: dict = eqx.static_field()
    decoder_params_dict: dict = eqx.static_field()
    orth_reg_strategy: str = eqx.static_field()
    activation_fn_name: str = eqx.static_field()
    reconstruct: bool = eqx.static_field()

    def __init__(
            self,
            problem: Problem,
            dataset: str,
            encoder_layer_sizes: List[int],  # decoder_layer_sizes = encoder_layer_sizes[::-1]
            inflater_layer_sizes: List[int],
            key: int,
            measurement_table: pd.DataFrame,
            observable_table: pd.DataFrame,
            condition_table: pd.DataFrame,
            samples_list: List[str],
            n_input_features: int,
            n_threads=1,
            # default for all modules: use eqx.nn.Linear layers
            encoder_weight_init_fn: str = "eqx_default",
            encoder_bias_init_fn: str = "eqx_default",
            inflater_weight_init_fn: str = "eqx_default",
            inflater_bias_init_fn: str = "eqx_default",
            decoder_weight_init_fn: str = "eqx_default",
            decoder_bias_init_fn: str = "eqx_default",
            # default: no learnable biases
            encoder_layer_biases: List[bool] = None,
            inflater_layer_biases: List[bool] = None,
            decoder_layer_biases: List[bool] = None,
            orth_reg_strategy: str = "L2",
            activation_fn_name: str = "relu",  # ReLU = Rectified Linear Unit
            reconstruct: bool = False,  # default: single head, no decoder (encoder->inflater)
    ):
        """

        :param dataset:
            name of dataset to use for model.

        :param problem:
            problem.pathway_name contains the name of pathway to use for model.


        -- ENCODER-specific params
        :param encoder_layer_sizes:
            list of layer sizes for encoder component (and decoder component, in reverse).

        :param encoder_weight_init_fn:
            encoder weight initialisation strategy.

        :param encoder_bias_init_fn:
            encoder bias initialisation strategy.

        :param encoder_layer_biases:
            list of bool values indicating whether to add a learnable bias or not for encoder layers.

        -- INFLATER-specific params
        :param inflater_layer_sizes:
            list of layer sizes for inflater component.

        :param inflater_weight_init_fn:
            inflater weight initialisation strategy.

        :param inflater_bias_init_fn:
            inflater bias initialisation strategy.

        :param inflater_layer_biases:
            list of bool values indicating whether to add a learnable bias or not for inflater layers.

        -- DECODER-specific params
        :param decoder_weight_init_fn:
            decoder weight initialisation strategy.

        :param decoder_bias_init_fn:
            decoder bias initialisation strategy.

        :param decoder_layer_biases:
            list of bool values indicating whether to add a learnable bias or not for decoder layers.

        -- OTHER params
        :param key:
            PRNG key.

        :param activation_fn_name:
            choice of activation function.
            Default: ReLU.

        :param reconstruct:
            boolean flag. If set to True, adds a second, autoencoding head to the network
            (encoder->decoder) on top of the first head (encoder->inflater).
            Default: single head (False).

        :param orth_reg_strategy:
            orthogonal regularisation strategy to be used: L1 vs L2 (default).

        :param n_threads:
            number of threads to use for pypesto.

         :param samples_list:
            List of samples (previously features.index).

        :param n_input_features:
            Number of features (not sure if needed).

        """

        self.data_name = dataset
        self.pathway_name = problem.pathway_name

        # TODO @GiacomoFabrini n_params needs to come from petab problem

        # subset samples
        self.petab_importer = load_petab(
            problem=problem,
            dataset=dataset,
            measurement_table=measurement_table,
            condition_table=condition_table,
            observable_table=observable_table,
            samples=samples_list,  # features needed here!
        )
        self.pypesto_subproblem = self.petab_importer.create_problem()

        # extract sample names, ordering of those is important since samples
        # must match when reshaping the inflated matrix
        petab_samples = []
        for name in self.pypesto_subproblem.x_names:
            if not name.startswith(MODEL_FEATURE_PREFIX):
                continue

            sample = name.split("__")[-1]
            if sample not in petab_samples and sample in samples_list:
                petab_samples.append(sample)

        self.n_samples = len(samples_list)
        self.n_features = n_input_features

        # n_model_inputs = number of cell-line-specific parameters (per cell-line = sample)
        # these kinetic parameters are the targets of the inflater module
        self.n_model_inputs = int(
            sum(
                name.startswith(MODEL_FEATURE_PREFIX)
                for name in self.pypesto_subproblem.x_names
            )
            / self.n_samples
        )

        # n_kin_params = number of NON cell-line specific parameters
        self.n_kin_params = (
                self.pypesto_subproblem.dim - self.n_model_inputs * self.n_samples
        )

        # set sample names
        self.sample_names = samples_list

        # set regularisation strategy, activation function and reconstruct flag
        self.orth_reg_strategy = orth_reg_strategy
        self.activation_fn_name = activation_fn_name
        self.reconstruct = reconstruct

        # Initialise module biases to default value if None (i.e. use_bias = False for all)
        # Check for shape mismatches between layer_sizes and layer_biases
        encoder_layer_biases = init_biases(
            biases=encoder_layer_biases,
            layer_sizes=encoder_layer_sizes,
            component_name="encoder"
        )
        inflater_layer_biases = init_biases(
            biases=inflater_layer_biases,
            layer_sizes=inflater_layer_sizes,
            component_name="inflater"
        )
        decoder_layer_biases = init_biases(
            biases=decoder_layer_biases,
            layer_sizes=encoder_layer_sizes,
            component_name="decoder"
        )

        # encoder parameters/properties
        self.encoder_params_dict = {
            "encoder_layer_sizes": encoder_layer_sizes,
            "encoder_layer_biases": encoder_layer_biases,
            "encoder_weight_init_fn": encoder_weight_init_fn,
            "encoder_bias_init_fn": encoder_bias_init_fn,
        }
        # inflater parameters/properties
        self.inflater_params_dict = {
            "inflater_layer_sizes": inflater_layer_sizes,
            "inflater_layer_biases": inflater_layer_biases,
            "inflater_weight_init_fn": inflater_weight_init_fn,
            "inflater_bias_init_fn": inflater_bias_init_fn,
        }
        # decoder parameters/properties
        self.decoder_params_dict = {
            "decoder_layer_biases": decoder_layer_biases,
            "decoder_weight_init_fn": decoder_weight_init_fn,
            "decoder_bias_init_fn": decoder_bias_init_fn,
        }

        # Initialise TwoHeadedDeepAutoencoder
        super().__init__(
            n_input_features=self.n_features,
            n_inflated_specific_kin_params=self.n_model_inputs,
            n_global_kin_params=self.n_kin_params,
            **self.encoder_params_dict,
            **self.inflater_params_dict,
            **self.decoder_params_dict,
            key=key,
            activation_fn_name=activation_fn_name,
            orth_reg_strategy=self.orth_reg_strategy,
            reconstruct=self.reconstruct,
        )

        problem.apply_objective_settings(
            self.pypesto_subproblem.objective, n_threads=n_threads
        )

        # augment TwoHeadedDeepAutoencoder.x_names with ODE x_names
        self.x_names = self.x_names + [
            name
            for ix, name in enumerate(self.pypesto_subproblem.x_names)
            if not name.startswith(MODEL_FEATURE_PREFIX)
            and ix in self.pypesto_subproblem.x_free_indices
        ]

    @property
    def n_latent(self):
        return self.encoder_params_dict["encoder_layer_sizes"][-1]

    @property
    def n_params(self):
        return self.inflater_params_dict["inflater_layer_sizes"][-1]

    # # TODO @GiacomoFabrini ask Fabian about this?!
    # def embedding(self, params: np.ndarray) -> jnp.ndarray:
    #     encode_weights, inflate_weights, kin_params = jnp.split(
    #         params,
    #         np.array(
    #             (
    #                 self.n_encode_weights,
    #                 self.n_inflate_weights + self.n_encode_weights,
    #             )
    #         ),
    #     )
    #     return jnp.concatenate(
    #         [
    #             kin_params,
    #             self.inflate_params(
    #                 self.encode(encode_weights), inflate_weights
    #             ).flatten(),
    #         ]
    #     )

    # TODO @GiacomoFabrini code single loss function incorporating all 4 components + reconstruction loss
    def loss_fn(self, ):
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
            return scale * jnp.mean(jnp.abs(m - jnp.eye(self.n_latent)) ** 2)
        else:
            raise ValueError(f"Invalid orth_reg_strategy: {self.orth_reg_strategy}")

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
            return scale * jnp.mean(jnp.abs(m - jnp.diag(jnp.diag(m))) ** 2)
        else:
            raise ValueError(f"Invalid orth_reg_strategy: {self.orth_reg_strategy}")

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
