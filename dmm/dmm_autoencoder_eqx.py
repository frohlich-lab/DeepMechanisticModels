import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pypesto.petab
from jaxtyping import Array
from sklearn.mixture import GaussianMixture

from . import MEDIAN_FEATURE_PREFIX, MODEL_FEATURE_PREFIX
from .config_options import Conf, ModuleParams
from .deepcomponent_eqx import KinParamsCombiner
from .model_utils import generate_layer_sizes
from .two_headed_deep_autoencoder_eqx import TwoHeadedDeepAutoencoder


def get_reg_exp(orth_reg_strategy):
    reg_exp_dict = {
        "L1": 1,
        "L2": 2,
    }
    if orth_reg_strategy not in reg_exp_dict.keys():
        raise ValueError(f"Invalid orth_reg_strategy: {orth_reg_strategy}")
    return reg_exp_dict[orth_reg_strategy]


def mse(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
):
    """
    Computes the Mean Squared Error (MSE) between predictions and targets.
    """
    return jnp.mean(jnp.square(predictions - targets))


def init_biases(biases, num_layers):
    if (biases is None) or (not biases):  # None or False
        biases = [False] * num_layers
    elif biases:
        biases = [True] * num_layers
    # removed - either all layers have biases or none have biases
    # elif len(biases) != num_layers:
    #     raise ValueError("Biases must have the same length as layer_sizes!")
    return biases


class DeepMechanisticModel(TwoHeadedDeepAutoencoder):
    kin_params_combiner: KinParamsCombiner
    # Sparsity masks are tuples to prevent undesired updates
    input_sparsity_binary_mask: tuple[int, ...]
    output_sparsity_binary_mask: tuple[int, ...]
    enable_inflater_output_reg: bool

    parameter_deviation_names: tuple[str, ...]
    parameter_median_names: tuple[str, ...]
    n_input_features: int
    conf: Conf

    def __init__(
        self,
        pypesto_problem: pypesto.Problem,
        conf: Conf,
        key: jax.random.PRNGKey,
        n_input_features: int,
    ):
        self.conf = conf
        self.enable_inflater_output_reg = True

        # n_deviations = number of cell-line-specific parameters (per cell-line = sample)
        # these kinetic parameters are the targets of the inflater module (previously model_inputs)

        self.parameter_deviation_names = tuple(
            name.replace(MEDIAN_FEATURE_PREFIX, "")
            for name in pypesto_problem.x_names
            if name.startswith(MEDIAN_FEATURE_PREFIX)
        )
        self.parameter_median_names = tuple(
            name.replace(MEDIAN_FEATURE_PREFIX, "")
            for name in pypesto_problem.x_names
            if not name.startswith(MODEL_FEATURE_PREFIX)
        )

        self.n_input_features = n_input_features
        n_deviations = len(self.parameter_deviation_names)
        n_medians = len(self.parameter_median_names)

        # check everything is in the right order:
        par_vector = np.array(pypesto_problem.x_names[n_medians:])
        par_array = par_vector.reshape((-1, n_deviations))

        # check that the all pars follow order in inflated_specific_kin_params
        for ipar, par_name in enumerate(self.parameter_deviation_names):
            assert np.vectorize(
                lambda x: x.startswith(MODEL_FEATURE_PREFIX + par_name)  # noqa: B023
            )(par_array[:, ipar]).all()

        # check roundtrip with flattening
        assert (par_array.flatten() == par_vector).all()

        # check that the median pars follow order in inflated_specific_kin_params
        assert (
            np.array(pypesto_problem.x_names[:n_deviations])
            == np.array(
                np.vectorize(lambda x: MEDIAN_FEATURE_PREFIX + x)(
                    self.parameter_deviation_names
                )
            )
        ).all()

        # Generate layer_sizes for whole modules (input, hidden, output)
        if not conf.multiheaded:
            encoder_layer_sizes = [
                self.n_input_features,
                *generate_layer_sizes(
                    latent_dim=conf.n_hidden,
                    depth=conf.depth,
                    max_width=3*self.n_input_features,
                    multiplier=conf.nn_structure_multiplier,
                    reverse=True,
                ),
                conf.n_hidden,
            ]
        else:
            encoder_layer_sizes = [
                int(self.n_input_features/3),  # one encoder per context (3 contexts)
                *generate_layer_sizes(
                    latent_dim=conf.n_hidden,
                    depth=conf.depth,
                    max_width=3*int(self.n_input_features/3),
                    multiplier=conf.nn_structure_multiplier,
                    reverse=True,
                ),
                conf.n_hidden,
            ]
        inflater_layer_sizes = [
            conf.n_hidden,
            *generate_layer_sizes(
                latent_dim=conf.n_hidden,
                depth=conf.depth,
                max_width=n_deviations,
                multiplier=conf.nn_structure_multiplier,
                reverse=False,
            ),
            n_deviations,
        ]

        # Define encoder, inflater and decoder parameters
        params = {
            f"{module}_params": ModuleParams(
                layer_sizes=layer_sizes,
                # Propagate use_bias to all layers, but do not use biases at inflater output, i.e. parameter deviations
                layer_biases=[conf.use_layer_bias] * len(layer_sizes)
                if module != "inflater"
                else [conf.use_layer_bias] * (len(layer_sizes) - 2) + [False],
                weight_init_fn=conf.nn_init_fn,
                bias_init_fn="zeros",
                last_layer_activation=conf.last_layer_activation,
                # TODO @GiacomoFabrini: discuss with Fabian which modules should have a last layer activation (all?)
                # Only apply dropout (if any) to the encoder module
                dropout_rate=conf.dropout_rate if module == "encoder" else 0.0,
            )
            for module, layer_sizes in zip(
                ["encoder", "inflater", "decoder"],
                [
                    encoder_layer_sizes,
                    inflater_layer_sizes,
                    encoder_layer_sizes[::-1],
                ],
            )
        }

        # Instantiate Kinetic Parameters Combiner module
        self.kin_params_combiner = KinParamsCombiner(
            component_name="kin_params_combiner",
            n_global_kin_params=n_medians,
        )

        # Initialise dummy sparsity binary masks with a tuple of ones the same size as input_features / inflater
        # kinetic param deviations for input / output masks
        self.input_sparsity_binary_mask = tuple(
            [1 for _ in range(self.n_input_features)]
        )
        self.output_sparsity_binary_mask = tuple(
            [1 for _ in range(n_deviations)]
        )

        # Initialise TwoHeadedDeepAutoencoder
        super().__init__(
            **params,
            reconstruct=conf.recon_loss > 0.0,
            key=key,
            activation_fn_name=conf.activation_fn_name,
            multiheaded=conf.multiheaded
        )

    def inflate_params(self, x, key):
        # filter inputs through input_sparsity_binary_mask
        y = jax.vmap(self.inflate, in_axes=(0, None))(
            x * jnp.array(self.input_sparsity_binary_mask), key
        )
        return self.conf.inflater_bound * jnp.tanh(
            y
            * jnp.array(self.output_sparsity_binary_mask)
            / self.conf.inflater_bound
        )

    # TODO - upgrade this function based on inputs needed for the update
    def update_input_sparsity_binary_mask(self, x):
        # Replace with code to update the input sparsity binary mask
        # Left x (input features) as input
        new_input_sparsity_binary_mask = self.input_sparsity_binary_mask
        return eqx.tree_at(
            lambda model: model.input_sparsity_binary_mask,
            self,
            new_input_sparsity_binary_mask,
        )

    def disable_inflater_output_reg(self):
        return eqx.tree_at(
            lambda m: m.enable_inflater_output_reg,
            self,
            False,
        )

    def update_output_sparsity_binary_mask(
        self, x, threshold_perc: str, round_up: bool = False
    ):
        """
        Update the sparsity binary mask based on the median parameter deviation across samples.
        :param x:
            input data.
        :param threshold_perc:
            percentage of the median parameter deviations to retain as cell-line-specific or
            `gmm` for automatic threshold detection via gaussian mixture model
        :param round_up:
            boolean flag to round up or down when computing the threshold (default: False).

        :return:
            new instance of DMM with updated sparsity binary mask.
        """
        # Compute standard deviation of parameter deviation across samples
        param_dev_stds = jnp.std(
            eqx.nn.inference_mode(self).inflate_params(x, jr.PRNGKey(0)),
            axis=0,
        )

        # Sort in descending order
        sorted_deviations = jnp.sort(param_dev_stds)[::-1]

        if threshold_perc == "gmm":
            gmm = GaussianMixture(n_components=2, random_state=42)
            gmm = gmm.fit(jnp.log(sorted_deviations.reshape(-1, 1)))

            threshold = jnp.exp(
                (gmm.means_[0, 0] + gmm.means_[1, 0]) / 2
            )  # take the mean of the two components
        else:
            threshold_perc = float(
                threshold_perc
            )  # convert to int if not already
            # Compute threshold to keep threshold_perc values and ensure within bounds
            # Given the number of cell-line-specific params is odd, we can choose whether to round up or down
            # Considering we want sparsity, I have opted to round down by default - behaviour can be changed via round_up.
            threshold = sorted_deviations[
                jnp.clip(
                    int(
                        jnp.floor(
                            len(sorted_deviations) * threshold_perc / 100
                        )
                    )
                    - 1
                    if not round_up
                    else int(
                        jnp.ceil(len(sorted_deviations) * threshold_perc / 100)
                    )
                    - 1,
                    0,
                    len(sorted_deviations) - 1,
                )
            ]

        # Check kinetic parameter deviation and zero out entries in the sparsity mask if below threshold
        new_output_sparsity_binary_mask = tuple(
            jnp.where(
                param_dev_stds < threshold,
                0.0,
                jnp.array(self.output_sparsity_binary_mask),
            ).tolist()
        )
        return eqx.tree_at(
            lambda model: model.output_sparsity_binary_mask,
            self,
            new_output_sparsity_binary_mask,
        )

    def l1_encode_reg(self):
        """
        L1 regularization of deep encoder weights.
        """
        if not self.multiheaded:
            return l1reg(self.deep_encoder, self.conf.l1reg_encode)
        else:
            return jnp.mean(
                jnp.array(
                    [
                        l1reg(encoder, self.conf.l1reg_encode)
                        for encoder in self.deep_encoder
                    ]
                )
            )

    def orth_encode_reg(self):
        """
        Orthogonal regularization of deep encoder weights.
        """
        if not self.multiheaded:
            return orth_reg(
                self.deep_encoder,
                self.conf.orth_reg_strategy,
                "encoder",
                self.conf.oreg_encode,
            )
        else:
            return jnp.mean(
                jnp.array(
                    [
                        orth_reg(
                            encoder,
                            self.conf.orth_reg_strategy,
                            "encoder",
                            self.conf.oreg_encode,
                        )
                        for encoder in self.deep_encoder
                    ]
                )
            )

    def l1_decode_reg(self):
        """
        L1 regularization of deep decoder weights.
        """
        if self.conf.l1reg_encode == 0.0:
            return 0.0
        if not self.multiheaded:
            return l1reg(self.deep_decoder, self.conf.l1reg_encode)
        else:
            return jnp.mean(
                jnp.array(
                    [
                        l1reg(decoder, self.conf.l1reg_encode)
                        for decoder in self.deep_decoder
                    ]
                )
            )

    def orth_decode_reg(self):
        """
        Orthogonal regularization of deep encoder weights.
        """
        if self.conf.oreg_encode == 0.0:
            return 0.0
        if not self.multiheaded:
            return orth_reg(
                self.deep_decoder,
                self.conf.orth_reg_strategy,
                "decoder",
                self.conf.oreg_encode,
            )
        else:
            return jnp.mean(
                jnp.array(
                    [
                        orth_reg(
                            decoder,
                            self.conf.orth_reg_strategy,
                            "decoder",
                            self.conf.oreg_encode,
                        )
                        for decoder in self.deep_decoder
                    ]
                )
            )

    def l1_inflate_reg(self):
        """
        L1 regularization of deep inflater weights.
        """
        if self.conf.l1reg_inflate == 0.0:
            return 0.0
        return l1reg(self.deep_inflater, self.conf.l1reg_inflate)

    def orth_inflate_reg(self):
        """
        Orthogonal regularization of deep inflater weights.
        """
        if self.conf.oreg_inflate == 0.0:
            return 0.0
        return orth_reg(
            self.deep_inflater,
            self.conf.orth_reg_strategy,
            "inflater",
            self.conf.oreg_inflate,
        )

    def l1reg_inflater_output(self, x: np.ndarray, key):
        """
        L1 regularization of inflater output - number of cell-specific deviations/log fold-changes.
        """
        # Introduced 1e-3 multiplier to investigate lower regularisation strengths without formatting issues
        if (
            not self.enable_inflater_output_reg
            or self.conf.l1reg_inflater_output == 0.0
        ):
            return 0.0
        return self.conf.l1reg_inflater_output * jnp.mean(
            jnp.abs(self.inflate_params(x, key))
        )

    def l2reg_inflater_output(self, x: np.ndarray, key):
        """
        L2 regularization of inflater output - number of cell-specific deviations/log fold-changes.
        """
        if self.conf.l2reg_inflater_output == 0.0:
            return 0.0
        # Introduced 1e-6 multiplier to investigate lower regularisation strengths without formatting issues
        return (
            self.conf.l2reg_inflater_output
            * 1e-6
            * jnp.linalg.norm(self.inflate_params(x, key), 2)
        )

    def reconstruction_loss(
        self,
        x: Array,
        key,
    ):
        """
        Reconstruction loss of the autoencoder (in case `self.reconstruct` == True).
        Simple Mean Squared Error (without the sqrt for now!)
        """
        if self.conf.recon_loss == 0.0:
            return 0.0
        reconstructed_x = jax.vmap(self.decode, in_axes=(0, None))(x, key)
        # fval contains MSE (not RMSE) - using MSE in reconstruction loss
        # TODO @GiacomoFabrini: fval and reconstruction loss use MSEs - need to move to RMSEs?!
        #  Are they on the same scale/order of magnitude as L1 terms if we leave them squared?!
        return self.conf.recon_loss * mse(
            predictions=reconstructed_x, targets=x
        )

    def symmetry_loss(self):
        """
        Symmetry loss for the autoencoder,
        pushes the decoder weights to be the transposed of the encoder weights.
        """
        if self.conf.symm_reg == 0.0:
            return 0.0
        symmetry_reg = 0
        num_layers = len(self.deep_encoder.layers) if not self.multiheaded else len(self.deep_encoder[0].layers)
        encoders = [self.deep_encoder] if not self.multiheaded else self.deep_encoder
        decoders = [self.deep_decoder] if not self.multiheaded else self.deep_decoder
        # Iterate over the encoder and decoder layers
        for encoder, decoder in zip(encoders, decoders):
            for encoder_layer, decoder_layer in zip(
                encoder.layers,
                decoder.layers[::-1],  # zip them in reverse order
            ):
                # Compute the weight difference for each pair of corresponding layers
                diff = encoder_layer.weight - decoder_layer.weight.T
                # Then compute mean squares differences per layer
                symmetry_reg += jnp.mean(jnp.square(diff))
        return (
            self.conf.symm_reg * symmetry_reg / (len(encoders) * num_layers)
        )  # mean across number of encoder/decoder & layers - should be on the same order of magnitude as MSE

    def constrain_median(self, x: Array):
        """
        Constrain median of global parameters to be close to initialisation (avg_model/per_sample), x.
        """
        if self.conf.median_reg == 0.0:
            return 0.0
        return self.conf.median_reg * mse(
            predictions=self.kin_params_combiner.learned_global_kin_params,
            targets=x,
        )

    def get_hyperparams(
        self,
    ) -> dict[str, int | dict]:
        """
        Get the hyperparameters of the model.
        """
        return {
            "conf": self.conf.to_dict(),
            "n_input_features": self.n_input_features,
        }

    def save(self, filename: Path) -> None:
        """
        Save the model to a file.

        :param filename: path of file
        """
        filename.parent.mkdir(exist_ok=True, parents=True)
        with Path.open(filename, "wb") as f:
            # Save model hyperparameters
            hyperparam_str = json.dumps(self.get_hyperparams())
            f.write((hyperparam_str + "\n").encode())
            # Save model parameters (weights, biases)
            eqx.tree_serialise_leaves(f, self)

    @classmethod
    def load(
        cls,
        filename: Path | str,
        pypesto_problem: pypesto.Problem,
    ) -> "DeepMechanisticModel":
        """
        Loads DMM model from a file.

        :param filename: path of file
        :param problem: CytofProblem instance
        :param dataset: dataset name (train/test)
        :param petab_base_files: petab base files (measurement, observable, condition tables)
        :return: Model instance
        """
        # Ensure filename is a Path object
        filename = Path(filename)
        with Path.open(filename, "rb") as f:
            # Load model hyperparameters
            hyperparam_str = f.readline().decode().strip()
            hyperparams = json.loads(hyperparam_str)
            hyperparams["conf"] = Conf(
                **hyperparams["conf"]
            )  # Convert dict to Conf object
            # Make model skeleton
            model = cls(
                **hyperparams,
                key=jax.random.PRNGKey(0),  # dummy key, no effect
                pypesto_problem=pypesto_problem,
            )
            # Apply serialised weights and biases to model skeleton
            model = eqx.tree_deserialise_leaves(f, model)
        return model


def l1reg(module, scale: float = 1.0):
    """
    L1 regularization of generic module weights.
    """
    l1reg_loss = 0
    for layer in module.layers:
        w = layer.weight
        l1reg_loss += scale * jnp.mean(jnp.abs(w))
    return l1reg_loss / len(module.layers)  # mean across all layers


def orth_reg(module, orth_reg_strategy, mode: str, scale: float = 1.0):
    """
    Orthogonal regularization of generic module weights.
    """
    oreg_loss = 0
    reg_exponent = get_reg_exp(orth_reg_strategy)
    if mode == "encoder":
        for layer in module.layers:
            w = layer.weight
            m = jnp.dot(w.T, w)
            oreg_loss += scale * jnp.mean(
                jnp.abs(m - jnp.eye(m.shape[0])) ** reg_exponent
            )
    else:  # decoder, inflater
        for layer in module.layers:
            w = layer.weight
            m = jnp.dot(w, w.T)
            oreg_loss += scale * jnp.mean(
                jnp.abs(m - jnp.diag(jnp.diag(m))) ** reg_exponent
            )
    return oreg_loss / len(module.layers)  # mean across all layers
