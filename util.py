import dataclasses
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
import petab
from amici.petab_objective import rdatas_to_simulation_df
from pypesto.C import MODE_RES

from common import (
    CONDITIONS_FILE,
    FEATURES_OUTFILFE,
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
    features: str = None
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

    features_train = pd.read_csv(
        FEATURES_OUTFILFE.format_map(dict(**conf.__dict__, dataset="train")),
        index_col=0,
    )

    dmm_train = DeepMechanisticModel(
        problem,
        conf.data,
        conf.n_hidden,
        **petab_base_files,
        features=features_train,
        l1reg=conf.alpha,
        n_threads=conf.n_threads,
    )

    if dataset == "train":
        return dmm_train, problem

    features_test = pd.read_csv(
        FEATURES_OUTFILFE.format_map(dict(**conf.__dict__, dataset="val")),
        index_col=0,
    )

    dmm_test = DeepMechanisticModel(
        problem,
        conf.data,
        conf.n_hidden,
        **petab_base_files,
        features=features_test,
        l1reg=conf.alpha,
        n_threads=conf.n_threads,
        pca=dmm_train.pca,
    )
    if dataset == "train+test":
        return (dmm_train, dmm_test), problem

    return dmm_test, problem


def rmse(pp, xx):
    try:
        x = pp.objective.jax_fun(xx)
        obj = pp.objective.base_objective._objectives[0]
        amici_model = obj.amici_model
        petab_problem = obj.amici_object_builder.petab_problem
        res = obj(x, mode=MODE_RES, return_dict=True)
        simulation_df = rdatas_to_simulation_df(
            res["rdatas"],
            model=amici_model,
            measurement_df=petab_problem.measurement_df,
        )
        return np.sqrt(
            np.mean(
                np.square(
                    simulation_df[petab.SIMULATION]
                    - petab_problem.measurement_df[petab.MEASUREMENT]
                )
            )
        )
    except Exception as e:
        print(e)
        return np.NaN
