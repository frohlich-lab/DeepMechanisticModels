import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import pandas as pd
import pypesto
import scipy.linalg as la

from common import (
    Conf,
    ModuleParams,
    FEATURES_OUTFILE,
    MODEL_FEATURE_PREFIX,
    PER_SAMPLE_OUTFILE_PARS
)
from cytof.problem import CytofProblem
from dmm.dmm_autoencoder_eqx import DeepMechanisticModel
from sklearn.decomposition import PCA
from typing import Tuple, Union
from util import load_petab_base_files


def make_dmm(*, dmm_params, features, key):
    return DeepMechanisticModel(
        **dmm_params,
        sample_name_list=list(features.index),
        n_input_features=features.shape[1],
        key=key,
    )


def load_models(
        conf: Conf,
        dataset: str = "train",
) -> Tuple[
    Union[
        Tuple[DeepMechanisticModel, DeepMechanisticModel],
        DeepMechanisticModel,
    ],
    CytofProblem,
]:
    problem = CytofProblem(conf.model)

    petab_base_files = load_petab_base_files(conf, reweight=True)

    # NOT NEEDED - moved from full layer list to just n_hidden layer list
    # # Check n_hidden is equal to last encoder layer size as well as first inflater layer size
    # if conf.encoder_layer_sizes[-1] != conf.n_hidden:
    #     raise ValueError("Chosen latent dimension mus match the size of the last encoder layer!")
    # elif conf.inflater_layer_sizes[0] != conf.n_hidden:
    #     raise ValueError("Chosen latent dimension mus match the size of the first inflater layer!")

    # TODO @GiacomoFabrini issues with test/val nomenclature here!
    settings = {
        "train": ["train"],
        "test": ["val"],
        "train+test": ["train", "val"],
    }

    # Define encoder, inflater and decoder parameters
    encoder_params = ModuleParams(
        layer_sizes=conf.encoder_layer_sizes,
        layer_biases=conf.use_layer_bias,
        weight_init_fn=conf.nn_init_fn,
        bias_init_fn=conf.nn_init_fn,
    )
    inflater_params = ModuleParams(
        layer_sizes=conf.inflater_layer_sizes,
        layer_biases=conf.use_layer_bias,
        weight_init_fn=conf.nn_init_fn,
        bias_init_fn=conf.nn_init_fn,
    )
    decoder_params = ModuleParams(
        layer_sizes=conf.encoder_layer_sizes[::-1],  # decoder layer sizes mirror encoder layer sizes
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
            [
                pd.read_csv(
                    FEATURES_OUTFILE.format_map(
                        dict(**conf.__dict__, dataset=setting)
                    ),
                    index_col=0
                )
                for setting in settings[dataset]
            ],
            keys
        )
    )

    # returns (dmm_train, dmm_val), problem | dmm_train, problem | dmm_val, problem depending on `dataset`
    if dataset == "train+test":
        return tuple(dmms), problem
    else:
        return *dmms, problem


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


def load_and_subset_input_features(
        conf: Conf,
        model: DeepMechanisticModel,
        dataset: str,
        pypesto_subproblem: pypesto.Problem = None,
):
    features = pd.read_csv(
        FEATURES_OUTFILE.format_map(dict(**conf.__dict__, dataset=dataset)),
        index_col=0,
    )
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
        dataset: str,  # train or test,
        pypesto_subproblem: pypesto.Problem = None,
):
    # Check that encoder, inflater (and potentially decoder) all have a single layer
    # but decoder layer sizes are simply given by the encoder, so only need to check encoder and inflater.
    if (len(model.deep_encoder.layers) > 1) or (len(model.deep_inflater.layers) > 1):
        raise ValueError("Both encoder and inflater must be single linear layers for linear initialisation!")

    # Always load training features
    if dataset == 'train':
        input_features_train = load_and_subset_input_features(conf=conf, model=model, dataset='train')
    elif (dataset == 'val') and (pypesto_subproblem is not None):
        input_features_train = load_and_subset_input_features(
            conf=conf,
            model=model,
            dataset='train',
            pypesto_subproblem=pypesto_subproblem,  # needed to get model.pypesto_subproblem.x_names
        )
    elif (dataset == 'val') and (pypesto_subproblem is None):
        raise ValueError("Need to pass pypesto_subproblem from model_train for validation dataset!")

    # fit PCA to training features
    pca = PCA(n_components=model.n_latent).fit(input_features_train)

    if dataset == 'train':
        # simply transform the already loaded training input features
        features_pca = pca.transform(input_features_train)
    elif dataset == 'val':
        # Load val features
        input_features_val = load_and_subset_input_features(
            conf=conf,
            model=model,
            dataset=dataset,
        )
        # and transform them with PCA fitted to training features
        features_pca = pca.transform(input_features_val)
    else:
        raise ValueError("Unknown dataset type: must be train/val.")

    # Overwrite encoder weights with PCA components
    new_encoder_weights = jnp.array(
        pca.components_.T.flatten()
    ).reshape(model.deep_encoder.layers[0].weight.shape)

    model = eqx.tree_at(
        lambda m: m.deep_encoder.layers[0].weight,  # fetch weights from single layer of encoder
        model,
        new_encoder_weights
    )
    if conf.encoder_layer_biases is not None:
        if conf.encoder_layer_biases[0]:  # if use_bias=True for single layer
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
            features_pca[:, : model.n_latent],
            par_deviations[inputs].values,
        )[0].flatten()
    ).reshape(model.deep_inflater.layers[0].weight.shape)

    model = eqx.tree_at(
        lambda m: m.deep_inflater.layers[0].weight,
        model,
        new_inflater_weights
    )
    if conf.inflater_layer_biases is not None:
        # same as done for encoder in case of use_bias=True
        if conf.inflater_layer_biases[0]:
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
        if conf.decoder_layer_biases is not None:
            # same as done for encoder/inflater in case of use_bias=True
            if conf.decoder_layer_biases[0]:
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
) -> jnp.ndarray():
    inputs = [
        "__".join(p.split("__")[:-1]).replace(MODEL_FEATURE_PREFIX, "")
        for p in model.petab_importer.petab_problem.parameter_df.index
        if p.startswith(MODEL_FEATURE_PREFIX) and p.endswith(par_combo.index[0])
    ]

    return par_combo[inputs].values
