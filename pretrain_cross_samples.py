"""
Pretraining of population + individual parameters based on per sample
pretraining
"""

import sys
from pathlib import Path

import fides
import numpy as np
import pandas as pd
import scipy.linalg as la
from pypesto.optimize import FidesOptimizer

import wandb
from common import (
    CROSS_SAMPLE_OUTFILE_PARS,
    CROSS_SAMPLE_OUTFILE_RESULTS,
    CROSS_SAMPLE_OUTFILE_TRACE,
    PER_SAMPLE_OUTFILE_PARS,
)
from mEncoder import MODEL_FEATURE_PREFIX
from mEncoder.pretraining import (
    generate_cross_sample_pretraining_problem,
    pretrain,
    store_and_plot_pretraining,
)
from util import load_from_argv

conf, (mae_train, mae_test), problem = load_from_argv(
    sys.argv, dataset="train+test"
)

pypesto_problem_train, pypesto_problem_test = (
    generate_cross_sample_pretraining_problem(mae, problem)
    for mae in (mae_train, mae_test)
)
pretrained_samples = {}

for sample in mae_train.sample_names:
    df = pd.read_csv(
        PER_SAMPLE_OUTFILE_PARS.format(**conf.__dict__, sample=sample),
        index_col=[0],
    )
    pretrained_samples[sample] = df[
        [col for col in df.columns if not col.startswith(MODEL_FEATURE_PREFIX)]
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
        par_combo = par_combo.reindex(mae_train.sample_names)
        means = par_combo.mean(skipna=True)
        par_combo -= means

        inputs = [
            "__".join(p.split("__")[:-1]).replace(MODEL_FEATURE_PREFIX, "")
            for p in mae_train.petab_importer.petab_problem.parameter_df.index
            if p.startswith(MODEL_FEATURE_PREFIX)
            and p.endswith(par_combo.index[0])
        ]

        # use this code to use reference inputs for initialization
        # if DATA.startswith("synthetic"):
        #     reference_inputs = pd.read_csv(
        #         data_dir / f"{DATA}__{MODEL}__reference_inputs.csv", index_col=[0]
        #     )
        #     for col in par_combo.columns:
        #         means[col] = reference_inputs.loc[col.replace("_obs", "")].values[0]
        #         if col not in inputs:
        #             continue
        #         for sample in par_combo.index:
        #             par_combo.loc[sample, col] = reference_inputs.loc[
        #                 f"{MODEL_FEATURE_PREFIX}{col}_{sample}"
        #             ].values[0, 0]
        w = la.lstsq(
            mae_train.data_pca[:, : mae_train.n_latent],
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


'''
elif INIT == 'sampling':
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
        n_starts = kwargs['n_starts']
        lb = kwargs['lb']

        dim = lb.size
        xs = np.empty((n_starts, dim))

        for istart in range(n_starts):
            for ix, xname in enumerate(
                    problem.get_reduced_vector(np.asarray(problem.x_names),
                                               problem.x_free_indices)
            ):
                xs[istart, ix] = np.random.random()*(ub-lb) + lb

        return xs
'''

fides_options = {
    fides.Options.FATOL: 0,
    fides.Options.FRTOL: 0,
    fides.Options.XTOL: 1e-8,
    fides.Options.MAXTIME: 3600 * 10,
    fides.Options.MAXITER: 100,
}

optimizer = FidesOptimizer(
    hessian_update=fides.HybridFixed(),
    options=fides_options,
)
np.random.seed(conf.job)
hfile = Path(CROSS_SAMPLE_OUTFILE_TRACE.format(**conf.__dict__))
result = pretrain(pypesto_problem_train, startpoints, 1, optimizer, hfile)

rfile = Path(CROSS_SAMPLE_OUTFILE_RESULTS.format(**conf.__dict__))
pfile = Path(CROSS_SAMPLE_OUTFILE_PARS.format(**conf.__dict__))
store_and_plot_pretraining(
    result, pfile=pfile, rfile=rfile, plot_waterfall=False
)

wandb.init(
    project=f"DeepMechanisticModels.{conf.data}.{conf.model}",
    group="pretraining",
    config={
        **conf.__dict__,
        "fides": fides_options,
    },
    name="pretraining__" + rfile.stem,
)

wandb.define_metric("loss_train", summary="min", step_metric="iter")
wandb.define_metric("loss_loss", summary="min", step_metric="iter")
wandb.define_metric("iter", summary="last", hidden=True)

for iter, (fval_train, x, grad) in enumerate(
    zip(
        result.optimize_result.list[0].history.get_fval_trace(trim=True),
        result.optimize_result.list[0].history.get_x_trace(trim=True),
        result.optimize_result.list[0].history.get_grad_trace(trim=True),
    )
):
    loss_train, loss_val = (
        pp.objective.base_objective._objectives[0](pp.objective.jax_fun(x))
        for pp in (pypesto_problem_train, pypesto_problem_test)
    )
    x_inflate, x_kinetic = np.split(x, (mae_train.n_inflate_weights,))
    grad_inflate, grad_kinetic = np.split(grad, (mae_train.n_inflate_weights,))
    wandb.log(
        {
            "loss_train": loss_train,
            "loss_val": loss_val,
            "iter": iter,
            "x_inflate": wandb.Histogram(x_inflate),
            "x_kinetic": wandb.Histogram(x_kinetic),
            "grad_inflate": wandb.Histogram(np.log(np.abs(grad_inflate))),
            "grad_kinetic": wandb.Histogram(np.log(np.abs(grad_kinetic))),
        }
    )

wandb.finish()
