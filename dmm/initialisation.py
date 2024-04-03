import dataclasses
from typing import Tuple, Union
from common import (
    FEATURES_OUTFILE,
)

import pandas as pd
from cytof.problem import CytofProblem
from util import load_petab_base_files
from dmm.autoencoder_eqx import DeepMechanisticModel

@dataclasses.dataclass
class Conf(dict):
    model: str
    data: str
    context: str = None
    features: str = None
    samples: str = None
    sample: str = None
    n_hidden: int = None
    encoder_layer_sizes: List[int]
    encoder_layer_biases: List[bool]
    inflater_layer_sizes: List[int]
    inflater_layer_biases: List[bool]
    decoder_layer_biases: List[bool]
    activation_fn_name: str
    reconstruct: bool
    orth_reg_strategy: str = None  # values: "L1" / "L2"
    l1reg_inflate: float = 0.0
    oreg_inflate: float = 0.0
    l1reg_encode: float = 0.0
    oreg_encode: float = 0.0
    job: int = None
    threads: int = 1
    n_starts: int = None


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

    # Check n_hidden is equal to last encoder layer size and first inflater layer size
    if conf.encoder_layer_sizes[-1] != conf.n_hidden
        raise ValueError("Chosen latent dimension mus match the size of the last encoder layer!")
    elif conf.inflater_layer_sizes[0] != conf.n_hidden:
        raise ValueError("Chosen latent dimension mus match the size of the first inflater layer!")

    # features_train = pd.read_csv(
    #     FEATURES_OUTFILE.format_map(dict(**conf.__dict__, dataset="train")),
    #     index_col=0,
    # )

    dmm_train = DeepMechanisticModel(
        problem=problem,
        dataset=conf.data,
        encoder_layer_sizes=conf.encoder_layer_sizes,
        encoder_layer_biases=conf.encoder_layer_biases,
        inflater_layer_sizes=conf.inflater_layer_sizes,
        inflater_layer_biases=conf.inflater_layer_biases,
        decoder_layer_biases=conf.decoder_layer_biases,
        orth_reg_strategy=conf.orth_reg_strategy,
        activation_fn_name=conf.activation_fn_name,
        reconstruct=conf.reconstruct,
        **petab_base_files,
        # features=features_train,
        n_threads=conf.threads,
    )

    if dataset == "train":
        return dmm_train, problem

    # features_test = pd.read_csv(
    #     FEATURES_OUTFILE.format_map(dict(**conf.__dict__, dataset="val")),
    #     index_col=0,
    # )

    dmm_test = DeepMechanisticModel(
        problem=problem,
        dataset=conf.data,
        encoder_layer_sizes=conf.encoder_layer_sizes,
        encoder_layer_biases=conf.encoder_layer_biases,
        inflater_layer_sizes=conf.inflater_layer_sizes,
        inflater_layer_biases=conf.inflater_layer_biases,
        decoder_layer_biases=conf.decoder_layer_biases,
        orth_reg_strategy=conf.orth_reg_strategy,
        activation_fn_name=conf.activation_fn_name,
        reconstruct=conf.reconstruct,
        **petab_base_files,
        # features=features_test,
        n_threads=conf.threads,
        # pca=dmm_train.pca,  # ISSUE HERE!
        # TODO @GiacomoFabrini issue with PCA here - what shall we do?!
    )
    if dataset == "train+test":
        return (dmm_train, dmm_test), problem

    return dmm_test, problem