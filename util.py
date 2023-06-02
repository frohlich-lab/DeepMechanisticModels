import pandas as pd
import numpy as np
import petab
import dataclasses

from typing import Dict, Tuple, List, Union

from common import (
    MEASUREMENTS_FILE, CONDITIONS_FILE, OBSERVABLES_FILE, Wildcards, training_samples, test_samples,
    PER_SAMPLE_OUTFILE_RESULTS
)
from cytof.problem import CytofProblem
from amici.petab_objective import rdatas_to_simulation_df
from pypesto.store import OptimizationResultHDF5Reader
from mEncoder.autoencoder import MechanisticAutoEncoder
from mEncoder.petab_subproblem import load_petab
from mEncoder.pretraining import generate_per_sample_pretraining_problems


def load_petab_base_files(model: str, dataset: str) -> Dict[str, pd.DataFrame]:
    return {
        'measurement_table':
            pd.read_csv(MEASUREMENTS_FILE.format(data=dataset, model=model), index_col=0, sep="\t"),
        'condition_table':
            pd.read_csv(CONDITIONS_FILE.format(data=dataset, model=model), index_col=0, sep="\t"),
        'observable_table':
            pd.read_csv(OBSERVABLES_FILE.format(data=dataset, model=model), index_col=0, sep="\t"),
    }


@dataclasses.dataclass
class Conf(dict):
    model: str
    data: str
    context: str
    samples: str
    n_hidden: int = 4
    alpha: float = 0.0
    job: int = 0


def load_mae(
    model: str,
    data: str,
    context: str = 'baseline',
    samples: str = '0_5',
    n_hidden: int = 4,
    alpha: float = 0.0,
    job: int = -1,
    dataset: str = 'train',
    n_threads: int = 1,
) -> Tuple[Conf, Union[MechanisticAutoEncoder, Tuple[MechanisticAutoEncoder,MechanisticAutoEncoder]], CytofProblem]:
    conf = Conf(
        model=model,
        data=data,
        context=context,
        samples=samples,
        n_hidden=n_hidden,
        alpha=alpha,
        job=job,
    )
    problem = CytofProblem(conf.model)

    petab_base_files = load_petab_base_files(conf.model, conf.data)

    samples = training_samples(Wildcards(conf.data, conf.samples))

    petab_base_importer = load_petab(
        problem,
        data,
        0.0,
        **load_petab_base_files(model, data),
    )

    sigmas = {}
    for sample in samples:
        importer = generate_per_sample_pretraining_problems(
            petab_base_importer,
            problem,
            data,
            sample,
        )
        pypesto_problem = importer.create_problem()
        rfile = PER_SAMPLE_OUTFILE_RESULTS.format(model=model, data=data, sample=sample)
        result = OptimizationResultHDF5Reader(rfile).read()

        problem.apply_objective_settings(pypesto_problem.objective)
        x = pypesto_problem.get_reduced_vector(result.optimize_result.list[0].x)
        res = pypesto_problem.objective(x, return_dict=True)

        simulation_df = rdatas_to_simulation_df(
            res["rdatas"],
            model=pypesto_problem.objective.amici_model,
            measurement_df=importer.petab_problem.measurement_df,
        )

        residuals_df = importer.petab_problem.measurement_df.copy()
        residuals_df['residual'] = \
            importer.petab_problem.measurement_df[petab.MEASUREMENT] - simulation_df[petab.SIMULATION]

        sigmas.update({
            (sample, observable, condition): np.sqrt(np.mean(np.power(group_df['residual'], 2)))
            for (observable, condition), group_df in
            residuals_df.groupby([petab.OBSERVABLE_ID, petab.SIMULATION_CONDITION_ID])
        })

    measurement_df = petab_base_files['measurement_table'].copy()
    for (sample, observable, condition), sigma in sigmas.items():
        measurement_df.loc[
            (measurement_df[petab.OBSERVABLE_ID] == observable) &
            (measurement_df[petab.SIMULATION_CONDITION_ID] == condition) &
            (measurement_df[petab.PREEQUILIBRATION_CONDITION_ID] == sample),
            petab.NOISE_PARAMETERS
        ] = sigma

    petab_base_files['measurement_table'] = measurement_df

    mae_train = MechanisticAutoEncoder(
        problem,
        conf.data,
        conf.n_hidden,
        **petab_base_files,
        samples=samples,
        l1reg=conf.alpha,
        contextualization=conf.context,
        n_threads=n_threads,
    )

    if dataset == "train":
        return conf, mae_train, problem

    mae_test = MechanisticAutoEncoder(
        problem,
        conf.data,
        conf.n_hidden,
        **petab_base_files,
        samples=test_samples(Wildcards(conf.data, conf.samples)),
        l1reg=conf.alpha,
        contextualization=conf.context,
        features=mae_train.features,
        imputer=mae_train.imputer,
        pca=mae_train.pca,
        n_threads=n_threads,
    )
    if dataset == "train+test":
        return conf, (mae_train, mae_test), problem

    return conf, mae_test, problem


def load_from_argv(
    argv: List[str],
    n_threads=1,
    dataset='train'
) -> Tuple[Conf, MechanisticAutoEncoder, CytofProblem]:
    argv.pop(0)  # remove script name
    return load_mae(
        model=argv.pop(0),
        data=argv.pop(0),
        context=argv.pop(0),
        samples=argv.pop(0),
        n_hidden=int(argv.pop(0)),
        alpha=float(argv.pop(0)),
        job=int(argv.pop(0) if argv else -1),
        dataset=dataset,
        n_threads=n_threads,
    )
