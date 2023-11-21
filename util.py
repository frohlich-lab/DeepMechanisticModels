import dataclasses
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
import pypesto
import scipy.linalg as la

from common import (
    CONDITIONS_FILE,
    FEATURES_OUTFILFE,
    MEASUREMENTS_FILE,
    MEASUREMENTS_FILE_RW,
    MODEL_FEATURE_PREFIX,
    OBSERVABLES_FILE,
    PER_SAMPLE_OUTFILE_PARS,
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
    l1reg_inflate: float = 0.0
    oreg_inflate: float = 0.0
    l1reg_encode: float = 0.0
    oreg_encode: float = 0.0
    job: int = None
    threads: int = 1
    n_starts: int = None


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
        n_threads=conf.threads,
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
        n_threads=conf.threads,
        pca=dmm_train.pca,
    )
    if dataset == "train+test":
        return (dmm_train, dmm_test), problem

    return dmm_test, problem


def generate_startpoint(
    conf: Conf,
    model: DeepMechanisticModel,
    problem: CytofProblem,
    pypesto_problem: pypesto.Problem,
) -> np.ndarray:
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

    np.random.seed(conf.job)

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
    means = par_combo.median(skipna=True)
    par_combo -= means

    inputs = [
        "__".join(p.split("__")[:-1]).replace(MODEL_FEATURE_PREFIX, "")
        for p in model.petab_importer.petab_problem.parameter_df.index
        if p.startswith(MODEL_FEATURE_PREFIX)
        and p.endswith(par_combo.index[0])
    ]

    w_inflate = la.lstsq(
        model.features_pca[:, : model.n_latent],
        par_combo[inputs].values,
    )[0].flatten()

    w_encode = model.pca.components_.T.flatten()

    xs = np.empty((pypesto_problem.dim,))

    # compute INPUT parameters as difference to mean
    for ix, xname in enumerate(pypesto_problem.x_names):
        if xname.startswith("inflate") and xname.endswith("weight"):
            xi = w_inflate[int(xname.split("_")[1])]
        elif xname.startswith("encode") and xname.endswith("weight"):
            xi = w_encode[int(xname.split("_")[1])]
        else:
            xi = means[xname]

        if np.isnan(xi):
            lb, ub, _ = problem.bounds[xname.split("_")[-1]]
            xi = np.random.random() * (ub - lb) + lb

        xs[ix] = xi

    return xs
