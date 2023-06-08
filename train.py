import sys
from pathlib import Path

import fire
import numpy as np
from pypesto.store import OptimizationResultHDF5Writer

import wandb
from common import (
    CROSS_SAMPLE_OUTFILE_PARS,
    TRAINING_OUTFILE_RESULTS,
    TRAINING_OUTFILE_TRACE,
    select_values,
)
from dmm.training import create_pypesto_problem, train
from util import Conf, load_models

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
hfile = Path(TRAINING_OUTFILE_TRACE.format(**conf.__dict__))

wandb.init(
    project=f"DeepMechanisticModels.{conf.data}.{conf.model}",
    group=f"training_{conf.context}_{conf.n_hidden}",
    config={
        **conf.__dict__,
        "mode": "train",
    },
    name="training__" + rfile.stem,
)

result, fides_options = train(
    model_train,
    pypesto_problem_train,
    problem,
    pretraining_file,
    hfile,
    n_starts=1,
    seed=conf.job,
)

wandb.config.update({"fides": fides_options})

rfile.parent.mkdir(exist_ok=True, parents=True)
writer = OptimizationResultHDF5Writer(str(rfile))
writer.write(result, overwrite=True)

wandb.define_metric("loss_train", summary="min", step_metric="iter")
wandb.define_metric("loss_val", summary="min", step_metric="iter")
wandb.define_metric("iter", summary="last", hidden=True)

for iter, (fval_train, x, grad) in select_values(
    enumerate(
        zip(
            result.optimize_result.list[0].history.get_fval_trace(trim=True),
            result.optimize_result.list[0].history.get_x_trace(trim=True),
            result.optimize_result.list[0].history.get_grad_trace(trim=True),
        )
    ),
    0.1,
):
    loss_train, loss_val = (
        pp.objective.base_objective._objectives[0](pp.objective.jax_fun(x))
        for pp in (pypesto_problem_train, pypesto_problem_test)
    )
    wandb.log(
        {
            "loss_train": loss_train,
            "loss_val": loss_val,
            "iter": iter,
            **{
                f"{val_type}_{xname}": None
                if not np.all(np.isfinite(value))
                else wandb.Histogram(value)
                if val_type == "x"
                else wandb.Histogram(np.log(np.abs(value)))
                for val_type, values in (
                    ("x", x),
                    ("g", grad),
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
        }
    )

wandb.finish()
