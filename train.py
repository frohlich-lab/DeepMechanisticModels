from logging import ERROR

import amici.logging
import fire
from jax import config

from common import (  # TRAINING_OUTFILE_RESULTS,
    FEATURES_OUTFILE,
    PER_SAMPLE_OUTFILE_PARS,
    TRAINED_MODEL,
    debug_mode,
)
from dmm.config_options import Conf, EarlyStoppingParams
from dmm.initialisation import (
    get_features_filepath,
    init_global_kin_params_combiner,
    process_features_and_setup_models,
    sort_features,
)
from dmm.training import train
from dmm.training_helper_funcs import create_pypesto_problem
from dmm.wandb_init_log import init_wandb
from training_configuration import (
    MIN_IMPROVEMENT,
    PATIENCE,
)
from util import load_petab_base_files

conf = fire.Fire(Conf)

avg_model_parameter_file = PER_SAMPLE_OUTFILE_PARS.format(
    **{**conf.to_dict(), "sample": f"model_average_{conf.samples}"}
)
# results_file = Path(TRAINING_OUTFILE_RESULTS.format(**conf)
model_file = TRAINED_MODEL.format(**conf.to_dict())

# Get filepaths for features and feature transformation pipeline
features_filepath = get_features_filepath(conf, FEATURES_OUTFILE)

# Set JAX configuration
config.update("jax_enable_x64", True)

# Get petab_base_files
petab_base_files = load_petab_base_files(conf=conf)
# Setup models + load and (potentially) transform input features (e.g. PCA)
(
    model,
    problem,
    pypesto_subproblems,
    features,
) = process_features_and_setup_models(
    conf=conf,
    features_filepath=features_filepath,
    petab_base_files=petab_base_files,
    dataset="train+val",
)

# Subset input features to cell-lines in train/val sets

input_features_train, input_features_test = (
    sort_features(
        features=features[dataset],
        pypesto_problem=pypesto_subproblems[dataset],
    )
    for dataset in ["train", "val"]
)

early_stopping_params = EarlyStoppingParams(
    patience=PATIENCE,  # (n_epoch-1) where we tolerate `rmse_val` not improving by at least min_improvement
    min_improvement=MIN_IMPROVEMENT,  # min absolute improvement not to lose patience (i.e. increase patience counter)
)

# Initialise the params of the KinParamsCombiner
model_train = init_global_kin_params_combiner(
    model=model,
    avg_model_parameter_file=avg_model_parameter_file,
    random_seed=conf.job,
)

# Setup pypesto problems for train/validation
pypesto_problem_train, pypesto_problem_test = (
    create_pypesto_problem(pypesto_subproblems[dataset])
    for dataset in ["train", "val"]
)

# Initialise W&B run and train
init_wandb(model_train, conf, early_stopping_params)
amici.logging.get_logger("amici.swig_wrappers").setLevel(ERROR)
train(
    model=model_train,
    problem_train=pypesto_problem_train,
    problem_test=pypesto_problem_test,
    input_features_train=input_features_train,
    input_features_test=input_features_test,
    conf=conf,
    # rfile=results_file,
    model_file=model_file,
    early_stopping_params=early_stopping_params,
    debug_mode=debug_mode,
)
