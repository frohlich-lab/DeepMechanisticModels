import dataclasses
from typing import Dict, Tuple, Union

import pandas as pd

from common import (
    CONDITIONS_FILE,
    MEASUREMENTS_FILE,
    MEASUREMENTS_FILE_RW,
    OBSERVABLES_FILE,
    Wildcards,
    test_samples,
    training_samples,
)
from cytof.problem import CytofProblem
from dmm.autoencoder import DeepMechanisticModel


@dataclasses.dataclass
class Conf(dict):
    model: str
    data: str
    context: str = None
    samples: str = None
    sample: str = None
    n_hidden: int = None
    alpha: float = None
    job: int = None
    n_threads: int = 1
    n_starts: int = None
    pretrain: bool = True


def load_petab_base_files(
    conf: Conf, reweight=False
) -> Dict[str, pd.DataFrame]:
    return {
        label: pd.read_csv(
            file.format(**conf.__dict__),
            index_col=0,
            sep="\t",
        )
        for label, file in (
            (
                "measurement_table",
                MEASUREMENTS_FILE_RW if reweight else MEASUREMENTS_FILE,
            ),
            ("condition_table", CONDITIONS_FILE),
            ("observable_table", OBSERVABLES_FILE),
        )
    }


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

    samples = training_samples(Wildcards(conf.data, conf.samples))

    dmm_train = DeepMechanisticModel(
        problem,
        conf.data,
        conf.n_hidden,
        **petab_base_files,
        samples=samples,
        l2reg=conf.alpha,
        contextualization=conf.context,
        n_threads=conf.n_threads,
    )

    if dataset == "train":
        return dmm_train, problem

    dmm_test = DeepMechanisticModel(
        problem,
        conf.data,
        conf.n_hidden,
        **petab_base_files,
        samples=test_samples(Wildcards(conf.data, conf.samples)),
        l2reg=conf.alpha,
        contextualization=conf.context,
        features=dmm_train.features,
        imputer=dmm_train.imputer,
        scaler=dmm_train.scaler,
        pca=dmm_train.pca,
        n_threads=conf.n_threads,
    )
    if dataset == "train+test":
        return (dmm_train, dmm_test), problem

    return dmm_test, problem
