import fire
import jax
import joblib
import os
import pandas as pd

from common import (
    EVALUATION_TRAINING,
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
from dmm.initialisation import get_features, pca_transform_features, subset_features
from evaluation_utils import load_model_and_obj
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
) -> pd.DataFrame:
    # Initialise list to store evaluations
    evaluations = []

    # Load model and objective
    model, obj = load_model_and_obj(conf, petab_base_files, dataset)

    # TODO @GiacomoFabrini need to fix this inconsistency in naming!
    # Extract needed features from input dictionary
    if dataset == 'train':
        features_dataset = 'train'
    elif dataset == 'test':
        features_dataset = 'val'
    input_features = subset_features(
            features=features[features_dataset],
            model=model,
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
        plot_file_prefix=EVALUATION_PLOT_FILE.format(dataset=dataset, **conf.__dict__),
    )

    return pd.DataFrame(evaluations)


# Load petab_base_files (once only)
petab_base_files = load_petab_base_files(conf, reweight=True)

# Load and transform features
features_filepath = FEATURES_OUTFILE.format(
    **{**conf.__dict__, **dict(dataset='{dataset}')}
)
feature_transform_pipeline_filepath = Path(
    FEATURES_PIPELINE.format_map(conf.__dict__)
)
features = get_features(features_filepath=features_filepath, datasets=['train', 'val'])
if conf.features_transform == "pca":
    # Load pre-trained pipeline if it exists
    if os.path.exists(feature_transform_pipeline_filepath):
        pipeline = joblib.load(feature_transform_pipeline_filepath)
    else:
        pipeline = None
    features = pca_transform_features(
        features=features,
        pipeline_filepath=feature_transform_pipeline_filepath,
        pipeline=pipeline,
    )


# TODO @GiacomoFabrini: check here "val" vs "test"
for dataset in [
        "train",
        "test"
]:
    # clear jax cache to avoid error where jitted function uses input with shape of train
    # which differs from test
    jax.clear_caches()
    df = evaluate_training(
        dataset=dataset,
        features=features,
        conf=conf,
        samples=samples,
        petab_base_files=petab_base_files
    )
    df.to_csv(
        EVALUATION_TRAINING.format(dataset=dataset, **conf.__dict__)
    )
