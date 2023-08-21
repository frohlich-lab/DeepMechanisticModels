from pathlib import Path

import fire
import numpy as np
import pandas as pd

from common import CROSS_SAMPLE_OUTFILE_PARS, TRAINING_OUTFILE_RESULTS
from dmm.training import create_pypesto_problem, train
from util import Conf, load_models

conf = fire.Fire(Conf)

rfile = Path(TRAINING_OUTFILE_RESULTS.format(**conf.__dict__))
pfile = Path(CROSS_SAMPLE_OUTFILE_PARS.format(**conf.__dict__))

(model_train, model_test), problem = load_models(
    conf,
    dataset="train+test",
)

pretraining_file = pfile if conf.pretrain else None

pypesto_problem_train, pypesto_problem_test = (
    create_pypesto_problem(mae, problem) for mae in (model_train, model_test)
)

if pretraining_file is not None and pretraining_file.exists():
    print(f'Loading pretraining from "{pretraining_file}"')
    pretraining = pd.read_csv(pretraining_file, index_col=0)
    xi = pretraining.loc[
        pypesto_problem_train.x_names[model_train.n_encode_weights :]
    ].values[:, 0]
    # use first couple PCA components
    w = model_train.pca.components_.T.flatten()
    x0 = np.hstack([w, xi])
else:
    print(f"randomly initializing training")
    xi = []
    for xname in pypesto_problem_train.x_names:
        lb, ub, _ = problem.bounds[xname.split("_")[-1]]
        xi.append(np.random.random() * (ub - lb) + lb)
    x0 = np.array(xi)

# inner_x = np.asarray(pypesto_problem_train.objective.jax_fun(x0))
#
# inner_pars = pd.DataFrame(
#     inner_x[model_train.n_kin_params:].reshape(
#         (len(model_train.sample_names), model_train.n_params)
#     ),
#     columns=model_train.x_names[
#         -model_train.n_kin_params:-(model_train.n_kin_params - model_train.n_params)
#     ],
#     index=model_train.sample_names
# )

schedule_config = dict(
    init_value=1e-2,
    transition_steps=100,
    end_value=1e-3,
)

train(
    model=model_train,
    problem_train=pypesto_problem_train,
    problem_test=pypesto_problem_test,
    conf=conf.__dict__,
    rfile=rfile,
    mode="train",
    schedule_config=schedule_config,
    n_epoch=500,
    x0=x0,
    par_dims=(
        ("encode", "inflate", "kinetic"),
        (
            model_train.n_encode_weights,
            model_train.n_encoder_pars,
        ),
    ),
)
