import fire

from common import Conf, EarlyStoppingParams, TRAINING_OUTFILE_RESULTS, TRAINED_BEST_MODELS
from dmm.initialisation import (linear_nn_init,
                                get_kin_params_median_deviation,
                                init_global_kin_params_combiner,
                                load_models,
                                load_and_subset_input_features,
                                get_targets)
from model_training.network_pretraining import pretrain_network
from model_training.training import train
from model_training.training_helper_funcs import create_pypesto_problem, map_params_to_array, sparsify_model
from model_training.wandb_init_log import init_wandb
from jax import config
from pathlib import Path
from sklearn.model_selection import train_test_split
from training_configuration import PATIENCE, MIN_IMPROVEMENT, N_EPOCHS


conf = fire.Fire(Conf)

# Remove blank spaces introduced by encoder/inflater_layer_sizes
results_file = Path(TRAINING_OUTFILE_RESULTS.format(**conf.__dict__).replace(" ", ""))
model_file = Path(TRAINED_BEST_MODELS.format(**conf.__dict__).replace(" ", ""))

# Set JAX configuration
config.update("jax_enable_x64", True)

# Setup models
(model_train, model_test), problem = load_models(
    conf,
    dataset="train+test",
)

# Load model_training and validation features
input_features_train, input_features_test = (
    load_and_subset_input_features(
        conf=conf,
        model=model,
        dataset=dataset,
    )
    for model, dataset in zip([model_train, model_test], ["train", "val"])
)

# TODO @GiacomoFabrini - differentiate schedule and early-stop between network pretraining and whole DMM model_training?
early_stopping_params = EarlyStoppingParams(
    patience=PATIENCE,  # (n_epoch-1) where we tolerate `rmse_val` not improving by at least min_improvement
    min_improvement=MIN_IMPROVEMENT,  # min absolute improvement not to lose patience (i.e. increase patience counter)
)

# There are three scenarios:
# 1. No hidden layers, linear_benchmark enabled - PCA/least squares initialisation for encoder/inflater (old approach)
# 2. No hidden layers, linear_benchmark disabled - pretraining
# 3. Hidden layers, linear_benchmark enabled - linear benchmark ignored -> pretraining
if (len(conf.encoder_layer_sizes) == 0) and (len(conf.inflater_layer_sizes) == 0) and conf.linear_benchmark:
    model_train = linear_nn_init(
        conf=conf,
        model=model_train,
        dataset="train",
    )
    model_test = linear_nn_init(
        conf=conf,
        model=model_test,
        dataset="val",
        pypesto_subproblem=model_train.pypesto_subproblem,  # needed to get PCA
    )
    model_train = init_global_kin_params_combiner(
        conf,
        model_train,
        nn_pretrain=False,
    )
    model_test = init_global_kin_params_combiner(
        conf,
        model_test,
        nn_pretrain=False,
    )
# elif (len(conf.encoder_layer_sizes) > 0) and (len(conf.inflater_layer_sizes) > 0) and conf.linear_benchmark:
#     raise ValueError("Linear benchmark is not possible with non-zero hidden layers!")
else:
    # Get training targets as parameter deviations (second component, while first contains medians)
    _, par_deviation_train = get_kin_params_median_deviation(conf, model_train)
    targets_train = get_targets(model_train, par_deviation_train)
    # Split model_training data and targets into pretrain train and val data and targets not to leak true validation
    data_pretrain_train, data_pretrain_val, targets_pretrain_train, targets_pretrain_val = train_test_split(
        input_features_train,
        targets_train,
        test_size=0.2,
        random_state=42
    )
    # Define filter_spec_per_param to freeze the KinParamsCombiner in the model
    model_train, filter_spec = init_global_kin_params_combiner(
        conf,
        model_train,
        nn_pretrain=True,
    )
    # Initialise W&B run
    init_wandb(model_train, conf, early_stopping_params, pretrain=True)
    # Get pretrained model
    pretrained_model = pretrain_network(
        model=model_train,
        filter_spec=filter_spec,
        training_data=data_pretrain_train,  # (batch_size, input_size)
        training_targets=targets_pretrain_train,  # (batch_size, output_size)
        validation_data=data_pretrain_val,
        validation_targets=targets_pretrain_val,
        conf=conf.__dict__,
        # rfile=rfile,
        n_epoch=1000,
        early_stopping_params=early_stopping_params,
    )
    # Now initialise the params of the KinParamsCombiner (No need for filter_spec_per_param?)
    model_train = init_global_kin_params_combiner(
        conf,
        pretrained_model,
        nn_pretrain=False,
    )

# Whole DMM Training
# Setup pypesto problems for train/validation
pypesto_problem_train, pypesto_problem_test = (
    create_pypesto_problem(mae) for mae in (model_train, model_test)
)

# Keep sparsity pattern learnt during regularisation, but drop regularisation, i.e.
# zero-out parameters below a given threshold and prepare filter_spec_per_param to mask updates (freeze).
model_train, filter_spec_per_param = sparsify_model(
    model_train,
    conf.drop_reg_after_pretrain,
    conf.sparsity_threshold,
)

# Get PEtab-compatible embedding of model parameters (i.e. global kin params concatenated with cell-line specific
# parameters, flattened for all model_training set samples/cell-lines).
x0 = map_params_to_array(model_train)

# Initialise W&B run
init_wandb(model_train, conf, early_stopping_params, pretrain=False)
train(
    model=model_train,  # can be pretrained or not (in case of linear benchmark)
    filter_spec_per_param=filter_spec_per_param,
    problem_train=pypesto_problem_train,
    input_features_train=input_features_train,
    input_features_test=input_features_test,
    problem_test=pypesto_problem_test,
    conf=conf.__dict__,
    rfile=results_file,
    model_file=model_file,
    n_epoch=N_EPOCHS,
    x0=x0,
    early_stopping_params=early_stopping_params,
)
