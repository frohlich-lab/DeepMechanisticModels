import itertools as itt
import pandas as pd

from common import EVALUATION_TRAINING, evaluations_dir, SafeDict
from training_configuration import (
    SPLITS, LAST_LAYER_ACTIVATION, USE_EARLY_STOP
)


def generate_hp_config(n_starts: int):
    STARTS = [str(i) for i in range(n_starts)]

    df = pd.read_csv(
        evaluations_dir / "EGFR_MAPK" / "dream_cytof" / "EGFR_MAPK.dream_cytof.top_10_best_dmm.csv"
    )
    # Subset to test/val and drop unnecessary columns + early-stopping (to override it)
    df_sub = df[df["dataset"] == "test"].drop(
        columns=[
            'Unnamed: 0', 'dataset', 'use_early_stopping', 'rmse mean', 'rmse std'
        ]
    )
    for column in ["n_hidden", "nn_structure_multiplier", "depth", "opt_steps", "opt_mult"]:
        df_sub[column] = df_sub[column].astype(int)
    # Transform to dictionary
    config_dict = df_sub.to_dict(orient='records')
    # Initialise list of hyperparameter configurations
    hyperparam_configurations = []
    for config, use_early_stopping, last_layer_activation, job, split in itt.product(
            config_dict, USE_EARLY_STOP, LAST_LAYER_ACTIVATION, STARTS, SPLITS,
    ):
        # Combine best performing configs with possible choices of early-stopping and last layer activation
        hyperparam_configurations.append(
            {
                **config,
                "samples": split,
                "last_layer_activation": last_layer_activation,
                "use_early_stopping": use_early_stopping,
                "job": job,
            }
        )
    training_evaluations = [
        EVALUATION_TRAINING.format_map(SafeDict(**hyperparam_configuration, dataset=dataset))
        for hyperparam_configuration in hyperparam_configurations
        for dataset in ['train', 'test']
    ]
    return hyperparam_configurations

configs = generate_hp_config(10)
