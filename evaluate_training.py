from pathlib import Path
from typing import Dict

import fire
import equinox as eqx
import jax
import pandas as pd

from common import (
    EVALUATION_EMBEDDING,
    EVALUATION_FULL_PARAMETERS,
    EVALUATION_PARAMETER_DEVIATIONS,
    EVALUATION_PLOT_FILE,
    EVALUATION_TRAINING,
    FEATURES_OUTFILE,
    Wildcards,
    fig_dir,
    results_dir,
    training_samples,
    val_samples,
)
from dmm.analysis import evaluate_simulations
from dmm.config_options import Conf
from dmm.initialisation import (
    get_features_filepath,
    process_features,
    subset_features,
)
from evaluation_utils import get_embedding_and_params_df, load_model_and_obj
from util import load_petab_base_files

conf = fire.Fire(Conf)

outdir = fig_dir / conf.model / conf.data
indir = results_dir / conf.model / conf.data

samples = {
    "train": training_samples(Wildcards(conf.data, conf.samples)),
    "val": val_samples(Wildcards(conf.data, conf.samples)),
}


def evaluate_training(
    dataset: str,
    features: Dict[str, pd.DataFrame],
    conf: Conf,
    samples: dict,
    petab_base_files: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    # Initialise list to store evaluations
    evaluations = []

    # Load model and objective
    model, obj = load_model_and_obj(conf, petab_base_files, dataset)

    # Set model to inference mode (essential for evaluation if dropout is applied to the encoder)
    model = eqx.nn.inference_mode(model)

    # Extract needed features from input dictionary
    input_features = subset_features(
        features=features[dataset],
        model=model,  # all models have the same input features
    )

    # Get latent embeddings and parameter dataframes
    le_df, params_dev_df, params_df = get_embedding_and_params_df(
        dmm_model=model,
        input_features=input_features,
        context=conf.context,
        split=conf.samples,
        dataset=dataset,
        job=conf.job,
    )

    evaluate_simulations(
        model=model,
        input_features=input_features,
        obj=obj,
        conf=conf,
        samples=samples[dataset],
        petab_problem=obj.amici_object_builder.petab_problem,
        dataset=dataset,
        outdir=outdir / "simulation",
        evaluations=evaluations,
        plot_file_prefix=EVALUATION_PLOT_FILE.format(
            dataset=dataset, **conf.__dict__
        ),
    )

    return pd.DataFrame(evaluations), le_df, params_dev_df, params_df


# Load petab_base_files (once only)
petab_base_files = load_petab_base_files(conf)

# Get filepaths for features and feature transformation pipeline
features_filepath = get_features_filepath(conf, FEATURES_OUTFILE)

features = process_features(
    conf=conf,
    features_filepath=features_filepath,
    datasets=["train", "val"],
)


for dataset in [
    "val",
    "train",
    # "test"  # TODO still don't have test data!
]:
    # clear jax cache to avoid error where jitted function uses input with shape of train
    # which differs from test
    jax.clear_caches()
    df, le_df, params_dev_df, params_df = evaluate_training(
        dataset=dataset,
        features=features,
        conf=conf,
        samples=samples,
        petab_base_files=petab_base_files,
    )
    df.to_csv(EVALUATION_TRAINING.format(dataset=dataset, **conf.__dict__))
    for results, path_format in zip(
        [le_df, params_dev_df, params_df],
        [
            EVALUATION_EMBEDDING,
            EVALUATION_PARAMETER_DEVIATIONS,
            EVALUATION_FULL_PARAMETERS,
        ],
    ):
        filepath = Path(path_format.format(dataset=dataset, **conf.__dict__))
        filepath.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(filepath)
