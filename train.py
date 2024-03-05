from pathlib import Path

import fire

from common import TRAINING_OUTFILE_RESULTS
from dmm.training import create_pypesto_problem, train
from util import Conf, generate_startpoint, load_models

conf = fire.Fire(Conf)

rfile = Path(TRAINING_OUTFILE_RESULTS.format(**conf.__dict__))

(model_train, model_test), problem = load_models(
    conf,
    dataset="train+test",
)

pypesto_problem_train, pypesto_problem_test = (
    create_pypesto_problem(mae, problem) for mae in (model_train, model_test)
)

x0 = generate_startpoint(
    conf=conf,
    model=model_train,
    problem=problem,
    pypesto_problem=pypesto_problem_train,
)

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
    schedule_config=schedule_config,
    n_epoch=1000,
    x0=x0,
    use_early_stopping=True,  # enables flax.training.early_stopping
    patience=9,  # number of consecutive epochs where we tolerate rmse_val not improving by at least min_improvement
    # flax evaluates early_stop.should_stop before updating early_stop.patience_count, so it actually stops
    # when early_stop.patience_count=patience+1, hence setting it to 9 for a desired max early_stop.patience_count=10
    min_improvement=5e-3,  # min_delta for flax.training.early_stopping: absolute improvement
    # 1% relative improvement on rmse_val around 0.5 corresponds to 5e-3 absolute improvement
)
