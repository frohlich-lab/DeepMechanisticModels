from pathlib import Path
from time import sleep

import fire
import numpy as np
from jax import config

from common import (  # TRAINING_OUTFILE_RESULTS,
    FEATURES_OUTFILE,
    PER_SAMPLE_OUTFILE_PARS,
    PRETRAINED_BEST_MODELS,
    TRAINED_BEST_MODELS,
    debug_mode,
)

# from cytof.problem import CytofProblem
from dmm.config_options import Conf, EarlyStoppingParams
from dmm.initialisation import (
    get_features_filepath,
    init_global_kin_params_combiner,
    process_features_and_setup_models,
    subset_features,
)
from dmm.training import train
from dmm.training_helper_funcs import create_pypesto_problem
from dmm.wandb_init_log import init_wandb
from training_configuration import (
    MIN_IMPROVEMENT,
    N_ENSEMBLE_MEMBERS,
    N_EPOCHS,
    PATIENCE,
)
from util import load_petab_base_files

# wait up to 1 minute to avoid filestream limits with wandb
sleep(np.random.randint(0, 60))

conf = fire.Fire(Conf)

per_sample_parameter_file = PER_SAMPLE_OUTFILE_PARS.format(
    **{**conf.__dict__, "sample": "{sample}"}
)
avg_model_parameter_file = PER_SAMPLE_OUTFILE_PARS.format(
    **{**conf.__dict__, "sample": f"model_average_{conf.samples}"}
)
# results_file = Path(TRAINING_OUTFILE_RESULTS.format(**conf.__dict__))
model_file = TRAINED_BEST_MODELS.format(
    **{**conf.__dict__, "ensemble_id": "{ensemble_id}"}
)
pretrained_model_file = Path(PRETRAINED_BEST_MODELS.format(**conf.__dict__))

# Get filepaths for features and feature transformation pipeline
features_filepath = get_features_filepath(
    conf, FEATURES_OUTFILE
)

# Set JAX configuration
config.update("jax_enable_x64", True)

# Get petab_base_files
petab_base_files = load_petab_base_files(conf=conf)
# Setup models + load and (potentially) transform input features (e.g. PCA)
(
    (model_train, model_test),
    problem,
    features,
) = process_features_and_setup_models(
    conf=conf,
    features_filepath=features_filepath,
    petab_base_files=petab_base_files,
    dataset="train+val",
    return_features=True,
)

# Subset input features to cell-lines in train/val sets
input_features_train, input_features_test = (
    subset_features(
        features=features[dataset],
        model=model,
    )
    for model, dataset in zip([model_train, model_test], ["train", "val"])
)

early_stopping_params = EarlyStoppingParams(
    patience=PATIENCE,  # (n_epoch-1) where we tolerate `rmse_val` not improving by at least min_improvement
    min_improvement=MIN_IMPROVEMENT,  # min absolute improvement not to lose patience (i.e. increase patience counter)
)

# Initialise the params of the KinParamsCombiner
model_train = init_global_kin_params_combiner(
    model=model_train,
    per_sample_parameter_file=per_sample_parameter_file,
    avg_model_parameter_file=avg_model_parameter_file,
    random_seed=conf.job,
    median_params_method=conf.median_init,
)

# Setup pypesto problems for train/validation
pypesto_problem_train, pypesto_problem_test = (
    create_pypesto_problem(mae) for mae in (model_train, model_test)
)

# Initialise W&B run and train
init_wandb(model_train, conf, early_stopping_params)
samples_name_list_dict = {
    dataset: model.sample_name_list
    for dataset, model in zip(["train", "test"], [model_train, model_test])
}
best_models = train(
    model=model_train,  # can be pretrained or not (in case of linear benchmark)
    problem_train=pypesto_problem_train,
    problem_test=pypesto_problem_test,
    input_features_train=input_features_train,
    input_features_test=input_features_test,
    conf=conf.__dict__,
    # rfile=results_file,
    model_file=model_file,
    samples_name_list_dict=samples_name_list_dict,
    n_epoch=N_EPOCHS,
    early_stopping_params=early_stopping_params,
    debug_mode=debug_mode,
    ensemble_members=N_ENSEMBLE_MEMBERS,
)

# TODO @GiacomoFabrini -- if this is still useful, it needs to handle the new structure of best_models, i.e.
#  a list where each item is (rmse_val, model)
# # Check whether the saved best_model indeed produces the best recorded RMSE on validation
# check_best_model(
#     best_model_filename=model_file,
#     cytof_problem=CytofProblem(conf.model),
#     petab_base_files=petab_base_files,
#     input_data=input_features_test,
#     pp=pypesto_problem_test,
#     best_rmse_val=rmse_test_min,
# )
