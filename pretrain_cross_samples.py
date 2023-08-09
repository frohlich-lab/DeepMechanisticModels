"""
Pretraining of population + individual parameters based on per sample
pretraining
"""

import itertools as itt
from pathlib import Path

import fire
import numpy as np
import pandas as pd
import scipy.linalg as la
from optax import adam, apply_updates, exponential_decay
from pypesto import Result
from pypesto.result.optimize import OptimizationResult, OptimizeResult
from pypesto.startpoint import FunctionStartpoints

import wandb
from common import (
    CROSS_SAMPLE_OUTFILE_PARS,
    CROSS_SAMPLE_OUTFILE_RESULTS,
    CROSS_SAMPLE_OUTFILE_TRACE,
    PER_SAMPLE_OUTFILE_PARS,
)
from dmm import MODEL_FEATURE_PREFIX
from dmm.pretraining import generate_cross_sample_pretraining_problem
from util import Conf, load_models, rmse

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
    startpoint_method = startpoints
else:
    startpoint_method = None

rfile = Path(CROSS_SAMPLE_OUTFILE_RESULTS.format(**conf.__dict__))
pfile = Path(CROSS_SAMPLE_OUTFILE_PARS.format(**conf.__dict__))

sps = FunctionStartpoints(startpoint_method)(
    n_starts=1,
    problem=pypesto_problem_train,
)

schedule_config = dict(
    init_value=1e-1,
    transition_steps=10,
    decay_rate=0.8,
    end_value=1e-3,
)

wandb.init(
    project=f"DeepMechanisticModels.{conf.data}.{conf.model}",
    group=f"pretraining_{conf.context}_{conf.features}_{conf.n_hidden}",
    config={
        **conf.__dict__,
        "adam_schedule": schedule_config,
        "mode": "pretrain",
    },
    name="pretraining__" + rfile.stem,
    settings=wandb.Settings(start_method="fork"),
)

wandb.define_metric("rmse_train", summary="min", step_metric="iter")
wandb.define_metric("rmse_val", summary="min", step_metric="iter")
for val_type, xname in itt.product(("x", "g"), ("inflate", "kinetic")):
    wandb.define_metric(f"{val_type}_{xname}", step_metric="iter")
wandb.define_metric("x_inflate", step_metric="iter")
wandb.define_metric("iter", summary="last", hidden=True)

N_ITER = 1000

x = sps[0, :]
x0 = x.copy()
fval = np.inf
grads = np.NaN * np.ones_like(x)

schedule = exponential_decay(**schedule_config)
opt = adam(schedule)
opt_state = opt.init(x)

opt_x = x.copy()
opt_fval = fval
opt_grads = grads.copy()
rmse_test_min = np.inf

for iteration in range(N_ITER + 1):
    fval, grads = pypesto_problem_train.objective(x, sensi_orders=(0, 1))

    updates, opt_state = opt.update(grads, opt_state)
    x = apply_updates(x, updates)
    print(
        f"iter {iteration:4d} (lr={schedule(opt_state[1].count):.2e}): {fval:.2f}"
    )
    if iteration % 10 == 0:
        rmses = dict()
        for dataset, pp in zip(
            ("train", "test"), (pypesto_problem_train, pypesto_problem_test)
        ):
            rmses[dataset] = rmse(pp, x)

        if rmses["test"] < rmse_test_min:
            rmse_test_min = rmses["test"]
            opt_x = x.copy()
            opt_fval = fval.copy()
            opt_grads = grads.copy()

        wandb.log(
            {
                "rmse_train": rmses["train"],
                "rmse_val": rmses["test"],
                "iter": iteration,
                **{
                    f"{val_type}_{xname}": None
                    if not np.all(np.isfinite(value))
                    else wandb.Histogram(value)
                    if val_type == "x"
                    else wandb.Histogram(np.log10(np.abs(value)))
                    for val_type, values in (
                        ("x", x),
                        ("g", grads),
                    )
                    for xname, value in zip(
                        ("inflate", "kinetic"),
                        np.split(values, (model_train.n_inflate_weights,)),
                    )
                },
            }
        )

wandb.finish()

OResult = OptimizeResult()
OResult.append(
    OptimizationResult(
        fval=opt_fval,
        x=opt_x,
        grad=opt_grads,
        x0=x0,
    )
)
result = Result(
    problem=pypesto_problem_train,
    optimize_result=OResult,
)


parameter_df = pd.Series(
    opt_x,
    index=pypesto_problem_train.x_names,
)
parameter_df.to_csv(pfile)
