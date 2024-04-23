import fire

from common import Conf, EarlyStoppingParams, TRAINING_OUTFILE_RESULTS
from dmm.initialisation import linear_nn_init, load_models, load_and_subset_input_features
from dmm.training import create_pypesto_problem, map_params_to_array, train
from pathlib import Path

conf = fire.Fire(Conf)

rfile = Path(TRAINING_OUTFILE_RESULTS.format(**conf.__dict__))

# TODO @GiacomoFabrini - this needs to load PRETRAINED NN modules!!!
#  Need to write the relevant scripts/rules!
(model_train, model_test), problem = load_models(
    conf,
    dataset="train+test",
)

# Linear benchmark: initialise via linear_nn_init
# 0 items in layer_sizes = no hidden layers
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
elif (len(conf.encoder_layer_sizes) > 0) and (len(conf.inflater_layer_sizes) > 0) and conf.linear_benchmark:
    print("Linear benchmark is not possible with non-zero hidden layers! conf.linear_benchmark will be ignored.")

# BROKEN FROM HERE ONWARDS
pypesto_problem_train, pypesto_problem_test = (
    create_pypesto_problem(mae) for mae in (model_train, model_test)
)

input_features_train, input_features_test = (
    load_and_subset_input_features(
        conf=conf,
        model=model,
        dataset=dataset,
    )
    for model, dataset in zip([model_train, model_test], ["train", "val"])
)

# To get the startpoint (x0) for kinetic parameters (x),
# simply pass the input_features_train into the pre-trained model_train
# (transpose them to shape = (n_features, n_samples))
# and extract the first component (output = augmented_inflated, decoded)
#x0 = model_train(input_features_train.T)[0]
x0 = map_params_to_array(model_train)

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

train(
    model=model_train,
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
