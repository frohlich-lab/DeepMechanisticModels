import fire
import jax.numpy as jnp

from common import Conf, EarlyStoppingParams, TRAINING_OUTFILE_RESULTS
from dmm.initialisation import (linear_nn_init,
                                get_kin_params_median_deviation,
                                init_global_kin_params_combiner,
                                load_models,
                                load_and_subset_input_features,
                                get_targets)
from dmm.network_pretraining import pretrain_network
from dmm.training import create_pypesto_problem, map_params_to_array, train
from pathlib import Path
from sklearn.model_selection import train_test_split

conf = fire.Fire(Conf)

rfile = Path(TRAINING_OUTFILE_RESULTS.format(**conf.__dict__))

# Setup models
(model_train, model_test), problem = load_models(
    conf,
    dataset="train+test",
)

# Load training and validation features
input_features_train, input_features_test = (
    load_and_subset_input_features(
        conf=conf,
        model=model,
        dataset=dataset,
    )
    for model, dataset in zip([model_train, model_test], ["train", "val"])
)

# Setup training configuration: schedule / early-stopping (same for pretraining and training) - FOR NOW
# TODO @GiacomoFabrini - differentiate schedule and early-stop between network pretraining and whole DMM training?
schedule_config = dict(
    init_value=1e-2,
    transition_steps=100,
    end_value=1e-3,
)

early_stopping_params = EarlyStoppingParams(
    use_early_stopping=True,  # enables flax.training.early_stopping
    patience=9,  # number of consecutive epochs where we tolerate rmse_val not improving by at least min_improvement
    # flax evaluates early_stop.should_stop before updating early_stop.patience_count, so it actually stops
    # when early_stop.patience_count=patience+1, hence setting it to 9 for a desired max early_stop.patience_count=10
    min_improvement=0,  # min_delta for flax.training.early_stopping: absolute improvement
    # 1% relative improvement on rmse_val around 0.5 corresponds to 5e-3 absolute improvement
    # reducing this to 0 to prolong training - 04.04.2024 (Fabian's suggestion)
)

# TODO - KinParamsCombiner init?
# Linear benchmark: initialise via linear_nn_init. 0 items in layer_sizes = no hidden layers.
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
elif (len(conf.encoder_layer_sizes) > 0) and (len(conf.inflater_layer_sizes) > 0) and conf.linear_benchmark:
    raise ValueError("Linear benchmark is not possible with non-zero hidden layers!")
else:
    # Get training targets as parameter deviations (second component, while first contains medians)
    _, par_deviation_train = get_kin_params_median_deviation(conf, model_train)
    targets_train = get_targets(model_train, par_deviation_train)
    # Split training data and targets into pretrain train and val data and targets not to leak true validation
    data_pretrain_train, data_pretrain_val, targets_pretrain_train, targets_pretrain_val = train_test_split(
        input_features_train,
        targets_train,
        test_size=0.2,
        random_state=42
    )
    # Define filter_spec to freeze the KinParamsCombiner in the model
    model_train, filter_spec = init_global_kin_params_combiner(
        conf,
        model_train,
        nn_pretrain=True,
    )
    # Get pretrained model
    pretrained_model = pretrain_network(
        model=model_train,
        filter_spec=filter_spec,
        training_data=data_pretrain_train.T,
        training_targets=targets_pretrain_train.T,
        validation_data=data_pretrain_val.T,
        validation_targets=targets_pretrain_val.T,
        conf=conf.__dict__,
        # rfile=rfile,
        schedule_config=schedule_config,
        n_epoch=1000,
        early_stopping_params=early_stopping_params,
    )
    # TODO @GiacomoFabrini - is this enough to pass the pretrained model?
    # Now initialise the params of the KinParamsCombiner (No need for filter_spec?)
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

# TODO @GiacomoFabrini -- need to link global params in KinParamsCombiner to global params with their names?!
x0 = map_params_to_array(model_train)

train(
    model=model_train,  # can be pretrained or not (in case of linear benchmark)
    problem_train=pypesto_problem_train,
    input_features_train=input_features_train.T,
    input_features_test=input_features_test.T,
    problem_test=pypesto_problem_test,
    conf=conf.__dict__,
    rfile=rfile,
    schedule_config=schedule_config,
    n_epoch=1000,
    x0=x0,
    early_stopping_params=early_stopping_params,
)
