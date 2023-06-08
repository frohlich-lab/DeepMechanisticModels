import dataclasses
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import petab
from amici.petab_objective import rdatas_to_simulation_df
from pypesto.store import OptimizationResultHDF5Reader

from common import (
    CONDITIONS_FILE,
    MEASUREMENTS_FILE,
    OBSERVABLES_FILE,
    PER_SAMPLE_OUTFILE_RESULTS,
    Wildcards,
    test_samples,
    training_samples,
)
from cytof.problem import CytofProblem
from dmm.autoencoder import DeepMechanisticModel
from dmm.petab_subproblem import load_petab
from dmm.pretraining import generate_per_sample_pretraining_problems


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


def load_petab_base_files(conf: Conf) -> Dict[str, pd.DataFrame]:
    return {
        label: pd.read_csv(
            file.format(data=conf.data, model=conf.model),
            index_col=0,
            sep="\t",
        )
        for label, file in (
            ("measurement_table", MEASUREMENTS_FILE),
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

    petab_base_files = load_petab_base_files(conf)

    samples = training_samples(Wildcards(conf.data, conf.samples))

    petab_base_importer = load_petab(
        problem,
        conf.data,
        0.0,
        **petab_base_files,
    )

    sigmas = {}
    for sample in samples:
        importer = generate_per_sample_pretraining_problems(
            petab_base_importer,
            problem,
            conf.data,
            sample,
        )
        pypesto_problem = importer.create_problem()
        rfile = PER_SAMPLE_OUTFILE_RESULTS.format(
            model=conf.model, data=conf.data, sample=sample
        )
        result = OptimizationResultHDF5Reader(rfile).read()

        problem.apply_objective_settings(pypesto_problem.objective)
        x = pypesto_problem.get_reduced_vector(
            result.optimize_result.list[0].x
        )
        res = pypesto_problem.objective(x, return_dict=True)

        simulation_df = rdatas_to_simulation_df(
            res["rdatas"],
            model=pypesto_problem.objective.amici_model,
            measurement_df=importer.petab_problem.measurement_df,
        )

        residuals_df = importer.petab_problem.measurement_df.copy()
        residuals_df["residual"] = (
            importer.petab_problem.measurement_df[petab.MEASUREMENT]
            - simulation_df[petab.SIMULATION]
        )

        sigmas.update(
            {
                (sample, observable, condition): np.sqrt(
                    np.mean(np.power(group_df["residual"], 2))
                )
                for (observable, condition), group_df in residuals_df.groupby(
                    [petab.OBSERVABLE_ID, petab.SIMULATION_CONDITION_ID]
                )
            }
        )

    measurement_df = petab_base_files["measurement_table"].copy()
    for (sample, observable, condition), sigma in sigmas.items():
        measurement_df.loc[
            (measurement_df[petab.OBSERVABLE_ID] == observable)
            & (measurement_df[petab.SIMULATION_CONDITION_ID] == condition)
            & (measurement_df[petab.PREEQUILIBRATION_CONDITION_ID] == sample),
            petab.NOISE_PARAMETERS,
        ] = sigma

    petab_base_files["measurement_table"] = measurement_df

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
        pca=dmm_train.pca,
        n_threads=conf.n_threads,
    )
    if dataset == "train+test":
        return (dmm_train, dmm_test), problem

    return dmm_test, problem
