import itertools as itt
from pathlib import Path

import fire
import git
import numpy as np
import pandas as pd
from optax import adam, apply_updates, linear_schedule
from pypesto import Result
from pypesto.result.optimize import OptimizeResult, OptimizerResult
from pypesto.store import OptimizationResultHDF5Writer

import wandb
from common import CROSS_SAMPLE_OUTFILE_PARS, TRAINING_OUTFILE_RESULTS
from dmm.training import create_pypesto_problem
from util import Conf, load_models, rmse

conf = fire.Fire(Conf)

(model_train, model_test), problem = load_models(conf, dataset="train+test")

pretraining_file = (
    Path(CROSS_SAMPLE_OUTFILE_PARS.format(**conf.__dict__))
    if conf.pretrain
    else None
)

pypesto_problem_train, pypesto_problem_test = (
    create_pypesto_problem(mae, problem) for mae in (model_train, model_test)
)

rfile = Path(TRAINING_OUTFILE_RESULTS.format(**conf.__dict__))

schedule_config = dict(
    init_value=1e-2,
    transition_steps=100,
    end_value=1e-3,
)

repo = git.Repo(search_parent_directories=True)

wandb.init(
    project=f"DeepMechanisticModels.{conf.data}.{conf.model}",
    group=f"training_{conf.context}_{conf.features}_{conf.n_hidden}",
    config={
        **conf.__dict__,
        "schedule_config": schedule_config,
        "optimizer": "adam",
        "scheduler": "linear",
        "mode": "train",
    },
    name="training__" + rfile.stem,
    settings=wandb.Settings(
        start_method="fork",
        git_commit=repo.head.object.hexsha,
        git_url=repo.remotes.origin.url,
    ),
)

wandb.define_metric("rmse_train", summary="min")
wandb.define_metric("rmse_val", summary="min")
for val_type, xname in itt.product(
    ("x", "g"), ("encode", "inflate", "kinetic")
):
    wandb.define_metric(f"{val_type}_{xname}")


schedule = linear_schedule(**schedule_config)
opt = adam(schedule)

N_ITER = 100

if pretraining_file is not None and pretraining_file.exists():
    pretraining = pd.read_csv(pretraining_file, index_col=0)
    xi = pretraining.loc[
        pypesto_problem_train.x_names[model_train.n_encode_weights :]
    ].values[:, 0]
else:
    xi = []
    for xname in pypesto_problem_train.x_names:
        lb, ub, _ = problem.bounds[xname.split("_")[-1]]
        xi.append(np.random.random() * (ub - lb) + lb)
    xi = np.array(xi)

# use first couple of PCA components
w = (
    model_train.pca.components_.T
    / np.sqrt(model_train.pca.explained_variance_)
).flatten()

x = np.hstack([w, xi])
x0 = x.copy()
opt_state = opt.init(x)
fval = np.inf
grads = np.NaN * np.ones_like(x)

opt_x = x.copy()
opt_fval = fval
opt_grads = grads.copy()
rmse_test_min = np.inf

for epoch in range(N_ITER + 1):
    fval, grads = pypesto_problem_train.objective(x, sensi_orders=(0, 1))
    updates, opt_state = opt.update(grads, opt_state)
    x = apply_updates(x, updates)
    rmses = dict()
    wandb.log({"fval": fval}, step=epoch)
    for dataset, pp in zip(
        ("train", "test"), (pypesto_problem_train, pypesto_problem_test)
    ):
        rmses[dataset] = rmse(pp, x)

    if rmses["test"] < rmse_test_min:
        rmse_test_min = rmses["test"]
        opt_x = x.copy()
        opt_fval = fval
        opt_grads = grads.copy()

    wandb.log(
        {
            "rmse_train": rmses["train"],
            "rmse_val": rmses["test"],
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
                    ("encode", "inflate", "kinetic"),
                    np.split(
                        values,
                        (
                            model_train.n_encode_weights,
                            model_train.n_encoder_pars,
                        ),
                    ),
                )
            },
        },
        step=epoch,
    )

wandb.finish()

OResult = OptimizeResult()
OResult.append(
    OptimizerResult(
        fval=opt_fval,
        x=opt_x,
        grad=opt_grads,
        x0=x0,
        id=str(conf.job),
    )
)
result = Result(
    problem=pypesto_problem_train,
    optimize_result=OResult,
)

rfile.parent.mkdir(exist_ok=True, parents=True)
writer = OptimizationResultHDF5Writer(str(rfile))
writer.write(result, overwrite=True)
