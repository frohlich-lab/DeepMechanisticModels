from typing import Dict

import numpy as np
import pandas as pd
import pypesto
import scipy.linalg as la

from common import (
    CONDITIONS_FILE,
    MEASUREMENTS_FILE,
    MODEL_FEATURE_PREFIX,
    OBSERVABLES_FILE,
    PER_SAMPLE_OUTFILE_PARS,
)
from cytof.problem import CytofProblem
from dmm.autoencoder import DeepMechanisticModel
from dmm.config_options import Conf


def load_petab_base_files(conf: Conf) -> Dict[str, pd.DataFrame]:
    return {
        label: pd.read_csv(
            file.format(**conf.__dict__),
            index_col=0,
            sep="\t",
        )
        for label, file in (
            ("measurement_table", MEASUREMENTS_FILE),
            ("condition_table", CONDITIONS_FILE),
            ("observable_table", OBSERVABLES_FILE),
        )
    }


# def load_models(
#     conf: Conf,
#     dataset: str = "train",
# ) -> Tuple[
#     Union[
#         DeepMechanisticModel,
#         Tuple[DeepMechanisticModel, DeepMechanisticModel],
#     ],
#     CytofProblem,
# ]:
#     problem = CytofProblem(conf.model)
#
#     petab_base_files = load_petab_base_files(conf)  # this used reweight=True, but we dropped reweighing
#
#     features_train = pd.read_csv(
#         FEATURES_OUTFILE.format_map(dict(**conf.__dict__, dataset="train")),
#         index_col=0,
#     )
#
#     dmm_train = DeepMechanisticModel(
#         problem,
#         conf.data,
#         conf.n_hidden,
#         conf.orth_reg_strategy,
#         **petab_base_files,
#         features=features_train,
#         n_threads=conf.threads,
#     )
#
#     if dataset == "train":
#         return dmm_train, problem
#
#     features_test = pd.read_csv(
#         FEATURES_OUTFILE.format_map(dict(**conf.__dict__, dataset="val")),
#         index_col=0,
#     )
#
#     dmm_test = DeepMechanisticModel(
#         problem,
#         conf.data,
#         conf.n_hidden,
#         conf.orth_reg_strategy,
#         **petab_base_files,
#         features=features_test,
#         n_threads=conf.threads,
#         pca=dmm_train.pca,
#     )
#     if dataset == "train+test":
#         return (dmm_train, dmm_test), problem
#
#     return dmm_test, problem


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
                **{**conf.__dict__, "sample": sample}
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
        if p.startswith(MODEL_FEATURE_PREFIX)
        and p.endswith(par_combo.index[0])
    ]

    # Background: Kunin et al. 2019, arXiv:1901.08168 [cs.LG]
    # showed that a linear autoencoder (LAE)
    # can be regularised to learn PCA by imposing an L2 penalty.
    # Here, we are using a simple linear encoder,
    # the weights of which (w_encode) are initialised with PCA
    w_encode = model.pca.components_.T.flatten()

    # Encoder weights (w_encode) are initialised with PCA.
    # Mechanistic model parameters are initialised with the median
    # across samples (means -- NEED TO FIX NAME).
    # For consistency, since the latent variables are inflated
    # into the mechanistic model parameters,
    # the inflate weights (w_inflate) are initialised as the
    # least squares solution of predicting the kinetic
    # parameters from the PCA encoder weights using a linear model.
    # In particular, for now they are initialised as the
    # least squares solution of regressing from PCA (w_encode)
    # the variation around the median of the kinetic
    # parameters (par_combo[input].values).
    w_inflate = la.lstsq(
        model.features_pca[:, : model.n_latent],
        par_combo[inputs].values,
    )[0].flatten()

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
