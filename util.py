import pandas as pd
import dataclasses

from typing import Dict, Tuple, List

from common import (
    MEASUREMENTS_FILE, CONDITIONS_FILE, OBSERVABLES_FILE, Wildcards, training_samples
)
from cytof.problem import CytofProblem
from mEncoder.autoencoder import MechanisticAutoEncoder


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


def load_from_conf(conf):
    samples = training_samples(Wildcards(conf.data, conf.samples))
    problem = CytofProblem(conf.model)
    mae = MechanisticAutoEncoder(
        problem,
        conf.data,
        conf.n_hidden,
        **load_petab_base_files(conf.model, conf.data),
        samples=samples,
        l1reg=conf.alpha,
        contextualization=conf.context,
        n_threads=4,
    )
    return conf, mae, problem


def load_from_argv(argv: List) -> Tuple[Conf, MechanisticAutoEncoder, CytofProblem]:
    argv.pop(0)  # remove script name
    conf = Conf(
        model=argv.pop(0),
        data=argv.pop(0),
        context=argv.pop(0),
        samples=argv.pop(0),
        n_hidden=int(argv.pop(0)),
        alpha=float(argv.pop(0)),
        job=int(argv.pop(0) if argv else -1)
    )
    return load_from_conf(conf)


def load_from_kwargs(
    model: str,
    data: str,
    context: str = 'baseline',
    samples: str = '0_5',
    n_hidden: int = 4,
    alpha: float = 0.0,
    job: int = -1,
) -> Tuple[Conf, MechanisticAutoEncoder, CytofProblem]:
    conf = Conf(
        model=model,
        data=data,
        context=context,
        samples=samples,
        n_hidden=n_hidden,
        alpha=alpha,
        job=job,
    )
    return load_from_conf(conf)
