import pandas as pd

from common import (
    Conf,
    FEATURES_OUTFILE,
)
from cytof.problem import CytofProblem
from dmm.dmm_autoencoder_eqx import DeepMechanisticModel
from typing import Tuple, Union
from util import load_petab_base_files


def load_models(
    conf: Conf,
    dataset: str = "train",
) -> Tuple[
    Union[
        DeepMechanisticModel,
        Tuple[DeepMechanisticModel, DeepMechanisticModel],
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
            n_features=features.shape[1]
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