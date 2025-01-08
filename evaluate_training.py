import fire
import jax
import pandas as pd

from common import (
    EVALUATION_TRAINING,
    EVALUATION_EMBEDDING,
    EVALUATION_FULL_PARAMETERS,
    EVALUATION_PARAMETER_DEVIATIONS,
    EVALUATION_PLOT_FILE,
    FEATURES_OUTFILE,
    FEATURES_PIPELINE,
    Wildcards,
    fig_dir,
    results_dir,
    test_samples,
    training_samples,
)
from dmm.analysis import evaluate_simulations
from dmm.config_options import Conf
from dmm.initialisation import (get_features, get_features_filepaths, process_features,
                                pca_transform_features, subset_features)
from evaluation_utils import load_model_and_obj, get_embedding_and_params_df
from training_configuration import N_ENSEMBLE_EVALUATION, N_ENSEMBLE_MEMBERS
from pathlib import Path
from typing import Dict
from util import load_petab_base_files


conf = fire.Fire(Conf)

outdir = fig_dir / conf.model / conf.data
indir = results_dir / conf.model / conf.data

# TODO @GiacomoFabrini: check here "val" vs "test"
samples = {
    "train": training_samples(Wildcards(conf.data, conf.samples)),
    "test": test_samples(Wildcards(conf.data, conf.samples)),
}


def evaluate_training(
        dataset: str,
        features: Dict[str, pd.DataFrame],
        conf: Conf,
        samples: dict,
        petab_base_files: Dict[str, pd.DataFrame],
        num_ensemble_members: int,
) -> pd.DataFrame:
    # Initialise list to store evaluations
    evaluations = []

    # Load ensemble models and objectives
    ensemble_models, obj = load_model_and_obj(conf, petab_base_files, dataset, num_ensemble_members)

    # TODO @GiacomoFabrini need to fix this inconsistency in naming!
    # Extract needed features from input dictionary
    if dataset == 'train':
        features_dataset = 'train'
    elif dataset == 'test':
        features_dataset = 'val'
    input_features = subset_features(
            features=features[features_dataset],
            model=ensemble_models[0],  # all models have the same input features
    )

    # Get latent embeddings and parameter dataframes
    le_df, params_dev_df, params_df = get_embedding_and_params_df(
        dmm_model=ensemble_models[0],
        input_features=input_features,
        context=conf.context,
        split=conf.samples,
        dataset=dataset,
        job=conf.job,
    )

    evaluate_simulations(
        models=ensemble_models,
        input_features=input_features,
        obj=obj,
        conf=conf,
        samples=samples[dataset],
        petab_problem=obj.amici_object_builder.petab_problem,
        dataset=dataset,
        outdir=outdir / "simulation",
        evaluations=evaluations,
        plot_file_prefix=EVALUATION_PLOT_FILE.format(dataset=dataset, **conf.__dict__),
    )

    return pd.DataFrame(evaluations), le_df, params_dev_df, params_df


# Load petab_base_files (once only)
petab_base_files = load_petab_base_files(conf)

# Get filepaths for features and feature transformation pipeline
features_filepath, feature_transform_pipeline_filepath = get_features_filepaths(
    conf, FEATURES_OUTFILE, FEATURES_PIPELINE
)

features = process_features(
    conf=conf,
    features_filepath=features_filepath,
    pipeline_filepath=feature_transform_pipeline_filepath,
    datasets=["train", "val"],
)


# TODO @GiacomoFabrini: check here "val" vs "test"
for dataset in [
        "train",
        "test"
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
        num_ensemble_members=N_ENSEMBLE_EVALUATION,
    )
    df.to_csv(
        EVALUATION_TRAINING.format(dataset=dataset, **conf.__dict__)
    )
    for results, path_format in zip(
        [le_df, params_dev_df, params_df],
        [EVALUATION_EMBEDDING, EVALUATION_PARAMETER_DEVIATIONS, EVALUATION_FULL_PARAMETERS],
    ):
        filepath = Path(path_format.format(dataset=dataset, **conf.__dict__))
        filepath.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(
                filepath
        )
