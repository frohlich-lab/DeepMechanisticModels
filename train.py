import sys
from pathlib import Path

import numpy as np
from pypesto.store import OptimizationResultHDF5Writer

import wandb
from common import (
    CROSS_SAMPLE_OUTFILE_PARS,
    TRAINING_OUTFILE_RESULTS,
    TRAINING_OUTFILE_TRACE,
)
from mEncoder.training import create_pypesto_problem, train
from util import load_from_argv

conf, (mae_train, mae_test), problem = load_from_argv(
    sys.argv, dataset="train+test"
)

pretraining_file = Path(CROSS_SAMPLE_OUTFILE_PARS.format(**conf.__dict__))

pypesto_problem_train, pypesto_problem_test = (
    create_pypesto_problem(mae, problem) for mae in (mae_train, mae_test)
)

hfile = Path(TRAINING_OUTFILE_TRACE.format(**conf.__dict__))
result, fides_options = train(
    mae_train,
    pypesto_problem_train,
    problem,
    pretraining_file,
    hfile,
    n_starts=1,
    seed=conf.job,
)

rfile = Path(TRAINING_OUTFILE_RESULTS.format(**conf.__dict__))
rfile.parent.mkdir(exist_ok=True, parents=True)
writer = OptimizationResultHDF5Writer(str(rfile))
writer.write(result, overwrite=True)

wandb.init(
    project=f"DeepMechanisticModels.{conf.data}.{conf.model}",
    group=f"training_{conf.context}_{conf.n_hidden}",
    config={
        **conf.__dict__,
        "fides": fides_options,
    },
    name="training__" + rfile.stem,
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
    x_encode, x_inflate, x_kinetic = np.split(
        x,
        (
            mae_train.n_encode_weights,
            mae_train.n_encoder_pars,
        ),
    )
    grad_encode, grad_inflate, grad_kinetic = np.split(
        grad,
        (
            mae_train.n_encode_weights,
            mae_train.n_encoder_pars,
        ),
    )
    wandb.log(
        {
            "loss_train": loss_train,
            "loss_val": loss_val,
            "iter": iter,
            "x_encode": wandb.Histogram(x_encode),
            "x_inflate": wandb.Histogram(x_inflate),
            "x_kinetic": wandb.Histogram(x_kinetic),
            "labs_g_encode": wandb.Histogram(np.log(np.abs(grad_encode))),
            "labs_g_inflate": wandb.Histogram(np.log(np.abs(grad_inflate))),
            "labs_g_kinetic": wandb.Histogram(np.log(np.abs(grad_kinetic))),
        }
    )

wandb.finish()
