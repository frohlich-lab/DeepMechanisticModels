import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import joblib
import numpy as np
import os
import pandas as pd
import pypesto
import scipy.linalg as la

from common import (
    Conf,
    ModuleParams,
    FEATURES_OUTFILE,
    FEATURES_PIPELINE,
    MODEL_FEATURE_PREFIX,
    PER_SAMPLE_OUTFILE_PARS
)
from cytof.problem import CytofProblem
from dmm.dmm_autoencoder_eqx import DeepMechanisticModel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from typing import Dict, List, Tuple, Union


def make_dmm(*, dmm_params, features, key):
    return DeepMechanisticModel(
        **dmm_params,
        sample_name_list=list(features.index),
        n_input_features=features.shape[1],
        key=key,
    )


def get_features(
        conf: Conf,
        datasets: List[str]
) -> Dict[str, pd.DataFrame]:
    features = {
        dataset: pd.read_csv(
                    FEATURES_OUTFILE.format_map(
                        dict(**conf.__dict__, dataset=dataset)
                    ),
                    index_col=0
                )
        for dataset in datasets
    }
    return features


def pca_transform_features(
        features: Dict[str, pd.DataFrame],
        conf: Conf,
        pipeline=None
) -> Dict[str, pd.DataFrame]:
    """
    :param features: dictionary of feature pd.DataFrames
    :param conf: configuration object
    :param pipeline: trained pipeline (optional)

    :return: dictionary of transformed features pd.DataFrames
    """
    if pipeline is None:
        # Construct the pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.95))
        ])
        # Fit the pipeline on the training data
        try:
            pipeline.fit(features["train"])
        except KeyError:
            # "train" key not found in the features dictionary
            raise ValueError("Training features not found in features dictionary - PCA cannot be fitted!")
        except Exception as e:
            # any other exceptions that might occur during fitting
            raise RuntimeError(f"An error occurred while fitting the pipeline: {e}")

        # Serialise the scaling+PCA pipeline
        joblib.dump(pipeline, FEATURES_PIPELINE.format_map(conf.__dict__))

    # Transform features and return them ensuring same pd.DataFrame format as pristine input
    transformed_features = {
        dataset: pd.DataFrame(
            pipeline.transform(features[dataset]),
            index=features[dataset].index,
            columns=[f'pca_{i}' for i in range(pipeline.named_steps['pca'].n_components_)]
        )
        for dataset in features.keys()
    }
    return transformed_features


def process_model_layers(conf: dict) -> dict:
    """
    Convert encoder_layer_sizes and inflater_layer_sizes from string/integer format
    to a list of integers.

    params:
        conf: dictionary version of Conf.

    returns:
        module_layer_sizes: dictionary with keys 'encoder_layer_sizes' and 'inflater_layer_sizes'.
    """
    module_layer_sizes = {}
    attributes = ['encoder_layer_sizes', 'inflater_layer_sizes']

    for attr in attributes:
        if isinstance(conf[attr], str):
            # Handle special case of no hidden layers
            if conf[attr] == "":
                module_layer_sizes[attr] = []
            else:
                # Map string format "size1_._size2_._...._._sizeN" to list [size1, size2, ..., sizeN]
                module_layer_sizes[attr] = list(map(int, conf[attr] .split('_._')))
        elif isinstance(conf[attr], int):
            # Handle case of single hidden layer -> convert int to list
            module_layer_sizes[attr] = [conf[attr]]
        else:
            raise TypeError(f"Invalid type for {attr} - must be either str or int. Found {type(conf[attr])}!")
    return module_layer_sizes


def setup_models(
        conf: Conf,
        petab_base_files,
        dataset: str = "train",
        return_features: bool = False,
) -> Union[
    Tuple[
        Union[
            Tuple[DeepMechanisticModel, DeepMechanisticModel],
            DeepMechanisticModel,
        ],
        CytofProblem,
    ],
    Tuple[
        Union[
            Tuple[DeepMechanisticModel, DeepMechanisticModel],
            DeepMechanisticModel,
        ],
        CytofProblem,
        Dict[str, pd.DataFrame],
    ]
]:
    problem = CytofProblem(conf.model)

    # TODO @GiacomoFabrini issues with test/val nomenclature here!
    settings = {
        "train": ["train"],
        "test": ["val"],
        "train+test": ["train", "val"],
    }
    # loads features corresponding to requested dataset settings
    features = get_features(conf, datasets=settings[dataset])

    if conf.features_transform == "pca":
        # Check whether pipeline has already been trained. If so, load it. If not, train it.
        pipeline_file = FEATURES_PIPELINE.format_map(conf.__dict__)
        if os.path.exists(pipeline_file):
            pipeline = joblib.load(FEATURES_PIPELINE.format_map(conf.__dict__))
        else:
            pipeline = None
        features = pca_transform_features(features, conf, pipeline)

    # Process network architecture parameters
    model_layers = process_model_layers(conf.__dict__)

    # Define encoder, inflater and decoder parameters
    encoder_params = ModuleParams(
        layer_sizes=model_layers['encoder_layer_sizes'],
        layer_biases=conf.use_layer_bias,
        weight_init_fn=conf.nn_init_fn,
        bias_init_fn=conf.nn_init_fn,
    )
    inflater_params = ModuleParams(
        layer_sizes=model_layers['inflater_layer_sizes'],
        layer_biases=conf.use_layer_bias,
        weight_init_fn=conf.nn_init_fn,
        bias_init_fn=conf.nn_init_fn,
    )
    decoder_params = ModuleParams(
        layer_sizes=model_layers['encoder_layer_sizes'][::-1],  # decoder layer sizes mirror encoder layer sizes
        layer_biases=conf.use_layer_bias,
        weight_init_fn=conf.nn_init_fn,
        bias_init_fn=conf.nn_init_fn,
    )

    dmm_params = {
        'problem': problem,
        'dataset': conf.data,
        'n_latent': conf.n_hidden,
        'encoder_params': encoder_params,
        'inflater_params': inflater_params,
        'decoder_params': decoder_params,
        'orth_reg_strategy': conf.orth_reg_strategy,
        'activation_fn_name': conf.activation_fn_name,
        'reconstruct': conf.reconstruct,
        'n_threads': conf.threads,
        **petab_base_files
    }

    key = jr.PRNGKey(conf.job)
    # Split keys for train/validation (otherwise identical weights, etc.)
    if len(settings[dataset]) > 1:
        keys = jr.split(key, num=len(settings[dataset]))
    else:
        keys = [key]

    dmms = (
        make_dmm(
            dmm_params=dmm_params,
            features=features,
            key=subkey,
        )
        for features, subkey in zip(
            [features[feature_dataset] for feature_dataset in settings[dataset]],
            keys
        )
    )

    result = (tuple(dmms), problem) if dataset == "train+test" else (*dmms, problem)
    return (*result, features) if return_features else result


def get_kin_params_median_deviation(
        conf: Conf,
        model: DeepMechanisticModel,
        return_full_combo: bool = False,
):
    pretrained_samples = {}

    for sample in model.sample_name_list:
        df = pd.read_csv(
            PER_SAMPLE_OUTFILE_PARS.format(
                **{**conf.__dict__, **dict(sample=sample)}
            ),
            index_col=[0],
        )
        pretrained_samples[sample] = df[
            [
                col
                for col in df.columns
                if not col.startswith(MODEL_FEATURE_PREFIX)
            ]
        ]
    # Set random seed for poisson sampling
    # this means all 0 jobs have the same matrix
    # of kinetic parameters vs cell-lines.
    # Same applies for all 1 jobs, all 2 jobs, etc.
    # Each job samples from the sets of pre-trained
    # parameters for each cell-line with a bias towards the
    # better performing multi-starts.
    # However, as cell-lines are not-paired,
    # we can combine different multistart parameter sets
    # across cell-lines.
    np.random.seed(conf.job)
    # key = jr.PRNGKey(conf.job)
    # poisson_sampling_keys = jr.split(key, num=len(pretrained_samples.values()))

    # Multi-starts of per-sample training are sorted
    # by loss function (ascending order, lower is better,
    # i.e. towards index 0).
    # Parameters for initialisation are chosen
    # from the multi-starts using Poisson sampling,
    # with Poisson(lambda=2).
    # Lambda is chosen so that the mode is small,
    # but slightly larger than 0, enabling some spread.
    # Lower index values will be more easily sampled,
    # leading to higher chance of sampling lower loss multi-starts.
    # TODO @GiacomoFabrini - discuss with Fabian - cannot get this to work - kept old version for now
    # par_combo = pd.concat(
    #     [
    #         pretraining.iloc[
    #             pretraining.index
    #             == jnp.min(jnp.array([jr.poisson(key=sampling_key, lam=2, shape=(1,))[0], len(pretraining) - 1]))
    #             ]
    #         for pretraining, sampling_key in zip(pretrained_samples.values(), poisson_sampling_keys)
    #     ]
    # )

    par_combo = pd.concat(
        [
            pretraining[
                pretraining.index
                == np.min([np.random.poisson(2, 1)[0], len(pretraining) - 1])
                ]
            for pretraining in pretrained_samples.values()
        ]
    )
    par_combo.index = list(pretrained_samples.keys())
    par_combo = par_combo.reindex(model.sample_name_list)

    if return_full_combo:
        return par_combo
    else:
        # Compute the median across samples
        par_medians = par_combo.median(skipna=True)
        # Subtract the median from the parameters:
        # par_combo now represents variation around the median
        par_deviations = par_combo - par_medians
        return par_medians, par_deviations


# def load_and_subset_input_features(
#         conf: Conf,
#         model: DeepMechanisticModel,
#         dataset: str,
#         pypesto_subproblem: pypesto.Problem = None,
# ):
#     features = pd.read_csv(
#         FEATURES_OUTFILE.format_map(dict(**conf.__dict__, dataset=dataset)),
#         index_col=0,
#     )
#     # extract sample names, ordering of those is important since samples
#     # must match when reshaping the inflated matrix
#     petab_samples = []
#     if pypesto_subproblem is None:
#         pypesto_subproblem = model.pypesto_subproblem
#     for name in pypesto_subproblem.x_names:
#         if not name.startswith(MODEL_FEATURE_PREFIX):
#             continue
#
#         sample = name.split("__")[-1]
#         if sample not in petab_samples and sample in features.index:
#             petab_samples.append(sample)
#
#     input_features = features.loc[petab_samples, :].values
#     return input_features


def subset_features(
        features: pd.DataFrame,
        model: DeepMechanisticModel,
        pypesto_subproblem: pypesto.Problem = None,
) -> np.ndarray:
    # extract sample names, ordering of those is important since samples
    # must match when reshaping the inflated matrix
    petab_samples = []
    if pypesto_subproblem is None:
        pypesto_subproblem = model.pypesto_subproblem
    for name in pypesto_subproblem.x_names:
        if not name.startswith(MODEL_FEATURE_PREFIX):
            continue

        sample = name.split("__")[-1]
        if sample not in petab_samples and sample in features.index:
            petab_samples.append(sample)

    input_features = features.loc[petab_samples, :].values
    return input_features


def linear_nn_init(
        conf: Conf,
        model: DeepMechanisticModel,
        features: Dict[str, np.ndarray],
        dataset: str,
):
    # Check that encoder, inflater (and potentially decoder) all have a single layer
    # but decoder layer sizes are simply given by the encoder, so only need to check encoder and inflater.
    if (len(model.deep_encoder.layers) > 1) or (len(model.deep_inflater.layers) > 1):
        raise ValueError("Both encoder and inflater must be single linear layers for linear initialisation!")

    if conf.features_transform == 'pca':
        # Features have already been PCA-transformed, just need to subset
        features_pca = features[dataset][:, :model.n_latent]
        # Initialise with all zeros
        new_encoder_weights = jnp.zeros_like(model.deep_encoder.layers[0].weight)
        # and replace upper model.n_latent * model.n_latent block with identity matrix of size model.n_latent
        new_encoder_weights = new_encoder_weights.at[:model.n_latent, :model.n_latent].set(jnp.eye(model.n_latent))
    else:
        # Features are pristine
        try:
            # fit PCA to training features
            pca = PCA(n_components=model.n_latent).fit(features["train"])
        except KeyError:
            # "train" key not found in the features dictionary
            raise ValueError("Training features not found in features dictionary - PCA cannot be fitted!")
        except Exception as e:
            # any other exceptions that might occur during fitting
            raise RuntimeError(f"An error occurred while fitting the pipeline: {e}")

        features_pca = pca.transform(features[dataset])
        # Compute new encoder weights with PCA components -- PREVIOUS SOLUTION
        # This will NOT produce the first n_hidden/model.n_latent PCA-transformed features as embedding
        # new_encoder_weights = jnp.array(
        #     pca.components_.T.flatten()
        # ).reshape(model.deep_encoder.layers[0].weight.shape)
        # APPROACH 1: last squares solution
        # new_encoder_weights = jnp.array(
        #     la.lstsq(
        #         features[dataset],
        #         features_pca[:, :model.n_latent],
        #     )[0].flatten()
        # ).reshape(model.deep_encoder.layers[0].weight.shape)
        # APPROACH 2: pinv -- lower numerical discrepancy between actual computed embedding, i.e.
        # `jax.vmap(model.deep_encoder)(features[dataset])` and target `features_pca`
        new_encoder_weights = jnp.dot(
            jnp.linalg.pinv(features[dataset]),  # pseudo-inverse
            features_pca
        ).T

    model = eqx.tree_at(
        lambda m: m.deep_encoder.layers[0].weight,  # fetch weights from single layer of encoder
        model,
        new_encoder_weights
    )
    if conf.use_layer_bias:
        model = eqx.tree_at(
            lambda m: m.deep_encoder.layers[0].bias,  # fetch bias from single linear layer
            model,
            jnp.zeros_like(model.deep_encoder.layers[0].bias),  # and set it to zero
        )

    # Compute target for least square initialisation of inflater weights:
    # kinetic parameter deviation around the median
    _, par_deviations = get_kin_params_median_deviation(conf=conf, model=model)

    inputs = [
        "__".join(p.split("__")[:-1]).replace(MODEL_FEATURE_PREFIX, "")
        for p in model.petab_importer.petab_problem.parameter_df.index
        if p.startswith(MODEL_FEATURE_PREFIX) and p.endswith(par_deviations.index[0])
    ]
    # Overwrite inflater weights with least squares solution
    new_inflater_weights = jnp.array(
        la.lstsq(
            features_pca,  # initialisation should ensure correct number of columns (i.e. model.n_latent)
            par_deviations[inputs].values,
        )[0].flatten()
    ).reshape(model.deep_inflater.layers[0].weight.shape)

    model = eqx.tree_at(
        lambda m: m.deep_inflater.layers[0].weight,
        model,
        new_inflater_weights
    )
    if conf.use_layer_bias:
        model = eqx.tree_at(
            lambda m: m.deep_inflater.layers[0].bias,
            model,
            jnp.zeros_like(model.deep_inflater.layers[0].bias),
        )
    # Overwrite decoder weights with inverse of encoder weights
    if conf.reconstruct:
        # initialise the decoder with the transpose of the encoder weights
        new_decoder_weights = new_encoder_weights.T
        if new_decoder_weights.shape != model.deep_decoder.layers[0].weight.shape:
            raise ValueError("Incorrect shape of new decoder weights!")
        model = eqx.tree_at(
            lambda m: m.deep_decoder.layers[0].weight,
            model,
            new_decoder_weights
        )
        if conf.use_layer_bias:
            model = eqx.tree_at(
                lambda m: m.deep_decoder.layers[0].bias,
                model,
                jnp.zeros_like(model.deep_decoder.layers[0].bias),
            )
    return model


def init_global_kin_params_combiner(
        conf: Conf,
        model: DeepMechanisticModel,
        nn_pretrain: bool,
):
    if not nn_pretrain:
        par_medians, _ = get_kin_params_median_deviation(conf=conf, model=model, return_full_combo=False)
        # Initialise global kin parameters combiner with median values of non-cell-line-specific parameter components
        new_global_kin_params = jnp.array(par_medians.values)
        # Check shape match prior to initialisation
        if new_global_kin_params.shape != model.kin_params_combiner.learned_global_kin_params.shape:
            raise ValueError("Incorrect shape of new global kin parameters!")
        # Initialise KinParamsCombiner parameters
        model = eqx.tree_at(
            lambda m: m.kin_params_combiner.learned_global_kin_params,  # fetch weights from single layer of encoder
            model,
            new_global_kin_params
        )
        return model
    else:
        # weights of kin_params_combiner are already initialised to zeros by default
        # just need to freeze them
        filter_spec = jtu.tree_map(lambda _: True, model)  # everything trained by default
        filter_spec = eqx.tree_at(
            lambda tree: (
                tree.kin_params_combiner.learned_global_kin_params,
            ),
            filter_spec,
            replace=(
                False,
            ),
        )
    return model, filter_spec


def get_targets(
        model: DeepMechanisticModel,
        par_combo: pd.DataFrame,
) -> jnp.ndarray:
    inputs = [
        "__".join(p.split("__")[:-1]).replace(MODEL_FEATURE_PREFIX, "")
        for p in model.petab_importer.petab_problem.parameter_df.index
        if p.startswith(MODEL_FEATURE_PREFIX) and p.endswith(par_combo.index[0])
    ]

    return par_combo[inputs].values
