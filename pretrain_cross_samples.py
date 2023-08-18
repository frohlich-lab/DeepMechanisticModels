"""
Pretraining of population + individual parameters based on per sample
pretraining
"""

from pathlib import Path

import fire
import numpy as np
import pandas as pd
import scipy.linalg as la
from pypesto.startpoint import FunctionStartpoints, UniformStartpoints

from common import (
    CROSS_SAMPLE_OUTFILE_PARS,
    CROSS_SAMPLE_OUTFILE_RESULTS,
    CROSS_SAMPLE_OUTFILE_TRACE,
    PER_SAMPLE_OUTFILE_PARS,
)
from dmm import MODEL_FEATURE_PREFIX
from dmm.pretraining import generate_cross_sample_pretraining_problem
from dmm.training import train
from util import Conf, load_models

conf = fire.Fire(Conf)

(model_train, model_test), problem = load_models(conf, dataset="train+test")

pypesto_problem_train, pypesto_problem_test = (
    generate_cross_sample_pretraining_problem(mae, problem)
    for mae in (model_train, model_test)
)
pretrained_samples = {}

if conf.pretrain:
    for sample in model_train.sample_names:
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


def startpoints(**kwargs):
    """
    Custom startpoint routine for cross sample pretraining. This function
    uses the results computed for the completely unconstrained problem
    where the model is just fitted to each individual sample. For each
    sample, a random local optimization result is picked. Then
    shared population parameters are computed as mean over all samples and
    sample specific input parameter are computed by substracting this mean
    from the local solution.
    """
    n_starts = kwargs["n_starts"]
    lb = kwargs["lb"]

    dim = lb.size
    xs = np.empty((n_starts, dim))

    for istart in range(n_starts):
        # use parameter values from random start for each sample
        par_combo = pd.concat(
            [
                pretraining[
                    pretraining.index
                    == np.min(
                        [np.random.poisson(2, 1)[0], len(pretraining) - 1]
                    )
                ]
                for pretraining in pretrained_samples.values()
            ]
        )
        par_combo.index = list(pretrained_samples.keys())
        par_combo = par_combo.reindex(model_train.sample_names)
        means = par_combo.mean(skipna=True)
        par_combo -= means

        inputs = [
            "__".join(p.split("__")[:-1]).replace(MODEL_FEATURE_PREFIX, "")
            for p in model_train.petab_importer.petab_problem.parameter_df.index
            if p.startswith(MODEL_FEATURE_PREFIX)
            and p.endswith(par_combo.index[0])
        ]

        w = la.lstsq(
            model_train.features_pca[:, : model_train.n_latent],
            par_combo[inputs].values,
        )[0].flatten()
        assert f"inflate_{len(w)-1}_weight" in pypesto_problem_train.x_names
        # compute INPUT parameters as difference to mean
        for ix, xname in enumerate(pypesto_problem_train.x_names):
            if xname.startswith("inflate") and xname.endswith("weight"):
                xi = w[int(xname.split("_")[1])]
            else:
                xi = means[xname]
            lb, ub, _ = problem.bounds[xname.split("_")[-1]]
            xs[istart, ix] = (
                xi if not np.isnan(xi) else np.random.random() * (ub - lb) + lb
            )

    return xs


np.random.seed(conf.job)
hfile = Path(CROSS_SAMPLE_OUTFILE_TRACE.format(**conf.__dict__))
if conf.pretrain:
    startpoint_method = FunctionStartpoints(startpoints)
else:
    startpoint_method = UniformStartpoints

rfile = Path(CROSS_SAMPLE_OUTFILE_RESULTS.format(**conf.__dict__))
pfile = Path(CROSS_SAMPLE_OUTFILE_PARS.format(**conf.__dict__))

sps = startpoint_method(
    n_starts=1,
    problem=pypesto_problem_train,
)

schedule_config = dict(
    init_value=1e-1,
    transition_steps=100,
    end_value=1e-2,
)

result = train(
    model_train,
    problem_train=pypesto_problem_train,
    problem_test=pypesto_problem_test,
    rfile=rfile,
    mode="pretrain",
    conf=conf.__dict__,
    schedule_config=schedule_config,
    n_epoch=250,
    x0=sps[0, :],
    par_dims=(("inflate", "kinetic"), (model_train.n_inflate_weights,)),
)

parameter_df = pd.Series(
    result.optimize_result.list[0]["x"],
    index=pypesto_problem_train.x_names,
)
parameter_df.to_csv(pfile)
