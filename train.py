import fire
import jax.tree_util as jtu

from common import (FEATURES_OUTFILE, FEATURES_PIPELINE,  # TRAINING_OUTFILE_RESULTS,
                    TRAINED_BEST_MODELS, PRETRAINED_BEST_MODELS, PER_SAMPLE_OUTFILE_PARS, debug_mode)
from cytof.problem import CytofProblem
from dmm.config_options import Conf, EarlyStoppingParams
from dmm.initialisation import (linear_nn_init,
                                get_kin_params_median_deviation,
                                init_global_kin_params_combiner,
                                setup_models,
                                subset_features,
                                get_targets)
from dmm.network_pretraining import pretrain_network
from dmm.training import train
from dmm.training_helper_funcs import check_best_model, create_pypesto_problem, map_params_to_array, sparsify_model
from dmm.wandb_init_log import init_wandb
from jax import config
from pathlib import Path
from sklearn.model_selection import train_test_split
from training_configuration import PATIENCE, MIN_IMPROVEMENT, N_EPOCHS, PRETRAIN_N_EPOCHS
from util import load_petab_base_files


conf = fire.Fire(Conf)

per_sample_parameter_file = PER_SAMPLE_OUTFILE_PARS.format(
    **{**conf.__dict__, **dict(sample="{sample}")}
)
avg_model_parameter_file = PER_SAMPLE_OUTFILE_PARS.format(
    **{**conf.__dict__, **dict(sample=f"model_average_{conf.samples}")}
)
# results_file = Path(TRAINING_OUTFILE_RESULTS.format(**conf.__dict__))
model_file = Path(TRAINED_BEST_MODELS.format(**conf.__dict__))
pretrained_model_file = Path(PRETRAINED_BEST_MODELS.format(**conf.__dict__))
features_filepath = FEATURES_OUTFILE.format(
    **{**conf.__dict__, **dict(dataset='{dataset}')}
)
feature_transform_pipeline_filepath = Path(FEATURES_PIPELINE.format(**conf.__dict__))
# Set JAX configuration
config.update("jax_enable_x64", True)

# Get petab_base_files
petab_base_files = load_petab_base_files(conf=conf)
# Setup models + load and (potentially) transform input features (e.g. PCA)
(model_train, model_test), problem, features = setup_models(
    conf=conf,
    features_filepath=features_filepath,
    pipeline_filepath=feature_transform_pipeline_filepath,
    petab_base_files=petab_base_files,
    dataset="train+test",
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

# TODO @GiacomoFabrini - differentiate schedule and early-stop between network pretraining and whole DMM training?
early_stopping_params = EarlyStoppingParams(
    patience=PATIENCE,  # (n_epoch-1) where we tolerate `rmse_val` not improving by at least min_improvement
    min_improvement=MIN_IMPROVEMENT,  # min absolute improvement not to lose patience (i.e. increase patience counter)
)

# There are three scenarios:
# 1. No hidden layers, linear_benchmark enabled - PCA/least squares initialisation for encoder/inflater (old approach)
# 2. No hidden layers, linear_benchmark disabled - pretraining
# 3. Hidden layers, linear_benchmark enabled - linear benchmark ignored -> pretraining
if (conf.depth == 0) and conf.linear_benchmark:
    input_features = {
        "train": input_features_train,
        "val": input_features_test
    }
    # First perform initialisation of encoder/inflater/decoder weights according to
    # previous linear benchmark strategy and then initialise KinParamsCombiner module
    # TODO @GiacomoFabrini: could also just process model_train, as model_test is only needed for its
    #  model_test.pypesto_subproblem
    model_train, model_test = (
        init_global_kin_params_combiner(
            model=linear_nn_init(
                conf=conf,
                model=model,
                per_sample_parameter_file=per_sample_parameter_file,
                avg_model_parameter_file=avg_model_parameter_file,
                features=input_features,
                dataset=dataset,
            ),
            per_sample_parameter_file=per_sample_parameter_file,
            avg_model_parameter_file=avg_model_parameter_file,
            random_seed=conf.job,
            nn_pretrain=False,
        )
        for model, dataset in zip((model_train, model_test), ["train", "val"])
    )

    # Setup filter_spec for model training (all True - learnable params)
    filter_spec_per_param = jtu.tree_map(lambda _: True, model_train)
else:
    # Get training targets as parameter deviations (second component, while first contains medians)
    _, par_deviation_train = get_kin_params_median_deviation(
        model=model_train,
        parameter_filepath=per_sample_parameter_file,
        avg_model_parameter_file=avg_model_parameter_file,
        random_seed=conf.job,
        return_full_combo=False
    )
    # Get test targets
    _, par_deviation_test = get_kin_params_median_deviation(
        model=model_test,
        parameter_filepath=per_sample_parameter_file,
        avg_model_parameter_file=avg_model_parameter_file,
        random_seed=conf.job,
        return_full_combo=False
    )
    targets_train, targets_test = (
        get_targets(model, par_deviations)
        for model, par_deviations in zip([model_train, model_test], [par_deviation_train, par_deviation_test])
    )
    # Split training data and targets into pretrain train and val data and targets not to leak true validation
    # Disabled after GM - using all train for pretraining
    # data_pretrain_train, data_pretrain_val, targets_pretrain_train, targets_pretrain_val = train_test_split(
    #     input_features_train,
    #     targets_train,
    #     test_size=0.2,
    #     random_state=42
    # )
    # Define filter_spec_per_param to freeze the KinParamsCombiner in the model
    model_train, filter_spec = init_global_kin_params_combiner(
        model_train,
        per_sample_parameter_file=per_sample_parameter_file,
        avg_model_parameter_file=avg_model_parameter_file,
        random_seed=conf.job,
        nn_pretrain=True,
    )
    # Initialise W&B run
    init_wandb(model_train, conf, early_stopping_params, pretrain=True)
    # Get pretrained model
    pretrained_model = pretrain_network(
        model=model_train,
        filter_spec=filter_spec,
        training_data=input_features_train,  # (batch_size, input_size)
        training_targets=targets_train,  # (batch_size, output_size)
        validation_data=input_features_test,
        validation_targets=targets_test,
        conf=conf.__dict__,
        # rfile=rfile,
        pretrained_model_file=pretrained_model_file,
        n_epoch=PRETRAIN_N_EPOCHS,
        early_stopping_params=early_stopping_params,
        debug_mode=debug_mode,
        return_best="train",
    )
    # Initialise the params of the KinParamsCombiner (No need for filter_spec_per_param?)
    model_train = init_global_kin_params_combiner(
        model=pretrained_model,
        per_sample_parameter_file=per_sample_parameter_file,
        avg_model_parameter_file=avg_model_parameter_file,
        random_seed=conf.job,
        nn_pretrain=False,
    )
    # For pretrained models: it might be desirable to keep the learnt sparsity pattern, but drop regularisation.
    # We do this by zeroing-out parameters below a given threshold + filtering their updates (effectively freezing
    # them) via a filter_spec.
    model_train, filter_spec_per_param = sparsify_model(
        model_train,
        conf.drop_reg_after_pretrain,
        conf.sparsity_threshold,
    )

# Whole DMM Training
# Setup pypesto problems for train/validation
pypesto_problem_train, pypesto_problem_test = (
    create_pypesto_problem(mae) for mae in (model_train, model_test)
)

# Initialise W&B run and train
init_wandb(model_train, conf, early_stopping_params, pretrain=False)
samples_name_list_dict = {
    dataset: model.sample_name_list
    for dataset, model in zip(["train", "test"], [model_train, model_test])
}
best_model, rmse_test_min = train(
    model=model_train,  # can be pretrained or not (in case of linear benchmark)
    filter_spec_per_param=filter_spec_per_param,
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
)

# Check whether the saved best_model indeed produces the best recorded RMSE on validation
check_best_model(
    best_model_filename=model_file,
    cytof_problem=CytofProblem(conf.model),
    petab_base_files=petab_base_files,
    input_data=input_features_test,
    pp=pypesto_problem_test,
    best_rmse_val=rmse_test_min,
)
