import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pypesto
import scipy.linalg as la

from common import (
    Conf,
    FEATURES_OUTFILE,
    MODEL_FEATURE_PREFIX,
    PER_SAMPLE_OUTFILE_PARS
)
from cytof.problem import CytofProblem
from dmm.dmm_autoencoder_eqx import DeepMechanisticModel
from jax import tree_util
from sklearn.decomposition import PCA
from typing import Tuple, Union
from util import load_petab_base_files


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

    # Check n_hidden is equal to last encoder layer size as well as first inflater layer size
    if conf.encoder_layer_sizes[-1] != conf.n_hidden:
        raise ValueError("Chosen latent dimension mus match the size of the last encoder layer!")
    elif conf.inflater_layer_sizes[0] != conf.n_hidden:
        raise ValueError("Chosen latent dimension mus match the size of the first inflater layer!")

    # TODO @GiacomoFabrini issues with test/val nomenclature here!
    settings = {
        "train": ["train"],
        "test": ["val"],
        "train+test": ["train", "val"],
    }

    dmm_params = {
        'problem': problem,
        'dataset': conf.data,
        'encoder_layer_sizes': conf.encoder_layer_sizes,
        'encoder_layer_biases': conf.encoder_layer_biases,
        'inflater_layer_sizes': conf.inflater_layer_sizes,
        'inflater_layer_biases': conf.inflater_layer_biases,
        'decoder_layer_biases': conf.decoder_layer_biases,
        'orth_reg_strategy': conf.orth_reg_strategy,
        'activation_fn_name': conf.activation_fn_name,
        'reconstruct': conf.reconstruct,
        'n_threads': conf.threads,
        **petab_base_files
    }

    dmms = [
        DeepMechanisticModel(
            **dmm_params,
            samples_list=list(features.index),
            n_input_features=features.shape[1]
        )
        for features in [
            pd.read_csv(
                FEATURES_OUTFILE.format_map(
                    dict(**conf.__dict__, dataset=setting)
                ),
                index_col=0
            )
            for setting in settings[dataset]
        ]
    ]

    # returns (dmm_train, dmm_val), problem | dmm_train, problem | dmm_val, problem depending on `dataset`
    return (*dmms), problem if len(settings[dataset]) > 1 else dmms[0], problem


def init_linear_weights(model, new_weights):
    is_linear = lambda x: isinstance(x, eqx.nn.Linear)
    get_weights = lambda m: [x.weight
                             for x in tree_util.tree_leaves(m, is_leaf=is_linear)
                             if is_linear(x)]
    new_model = eqx.tree_at(get_weights, model, new_weights)
    return new_model


def linear_nn_init(
    conf: Conf,
    model: DeepMechanisticModel,
    dataset: str,  # train or test
    problem: CytofProblem,
    pypesto_problem: pypesto.Problem,
):
    pretrained_samples = {}

    for sample in model.sample_names:
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
    par_combo = par_combo.reindex(model.sample_names)
    # Compute the median across samples
    means = par_combo.median(skipna=True)
    # Subtract the median from the parameters:
    # par_combo now represents variation around the median
    par_combo -= means

    inputs = [
        "__".join(p.split("__")[:-1]).replace(MODEL_FEATURE_PREFIX, "")
        for p in model.petab_importer.petab_problem.parameter_df.index
        if p.startswith(MODEL_FEATURE_PREFIX) and p.endswith(par_combo.index[0])
    ]

    # Load train/val features
    input_features_train = pd.read_csv(
        FEATURES_OUTFILE.format_map(dict(**conf.__dict__, dataset="train")),
        index_col=0,
    )
    input_features_val = pd.read_csv(
        FEATURES_OUTFILE.format_map(dict(**conf.__dict__, dataset="val")),
        index_col=0,
    )
    # Fit PCA on input features - train
    pca = PCA(n_components=model.n_latent).fit(input_features_train)

    # transform input features - both train and val
    features_train_pca = pca.transform(input_features_train)
    features_val_pca = pca.transform(input_features_val)

    # Overwrite encoder weights with PCA components
    new_encoder_weights = jnp.array(pca.components_.T.flatten())
    model.deep_encoder = init_linear_weights(
        model.deep_encoder,
        new_encoder_weights,
    )
    # Overwrite inflater weights with least squares solution
    # select features_pca depending on `dataset`
    features_pca = features_train_pca if dataset == 'train' else features_val_pca
    new_inflater_weights = jnp.array(la.lstsq(
        features_pca[:, : model.n_latent],
        par_combo[inputs].values,
    )[0].flatten())
    model.deep_inflater = init_linear_weights(
        model.deep_inflater,
        new_inflater_weights,
    )
    # Overwrite decoder weights with inverse of encoder weights
    if conf.reconstruct:
        # initialise the decoder with the transpose of the encoder weights
        new_decoder_weights = new_encoder_weights.T
        model.deep_decoder = init_linear_weights(
            model.deep_decoder,
            new_decoder_weights,
        )
    return model


# TODO @GiacomoFabrini: code initialisation for additive component following the inflater
def init_global_kin_params_adder(conf: Conf,
    model: DeepMechanisticModel,
    # dataset: str,  # train or test
    problem: CytofProblem,
    pypesto_problem: pypesto.Problem,
):

    return
