from pathlib import Path
from typing import Dict

import fire
import jax
import pandas as pd
from tqdm import tqdm

from common import (
    EVALUATION_EMBEDDING,
    EVALUATION_FULL_PARAMETERS,
    EVALUATION_PARAMETER_DEVIATIONS,
    EVALUATION_PLOT_FILE,
    EVALUATION_SENSITIVITY_LATENT,
    EVALUATION_SENSITIVITY_PARAMS,
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
    process_features_and_setup_models,
)
from dmm.training_helper_funcs import create_pypesto_problem, rmse
from evaluation_utils import get_embedding_and_params_df, load_model
from sensitivity import (
    activate_single_param,
    classify_param,
    set_param_to_zero,
    zero_latent_direction,
)
from util import load_petab_base_files

conf = fire.Fire(Conf)

outdir = fig_dir / conf.model / conf.data
indir = results_dir / conf.model / conf.data

samples = {
    "train": training_samples(Wildcards(conf.data, conf.samples)),
    "val": val_samples(Wildcards(conf.data, conf.samples)),
}


def evaluate_training(
    model,
    dataset: str,
    features: Dict[str, pd.DataFrame],
    conf: Conf,
    pypesto_problem,
) -> tuple[pd.DataFrame, ...]:
    # Initialise list to store evaluations
    evaluations = []

    # Get latent embeddings and parameter dataframes
    le_df, params_dev_df, params_df = get_embedding_and_params_df(
        dmm_model=model,
        input_features=features[dataset].values,
        context=conf.context,
        split=conf.samples,
        dataset=dataset,
        job=conf.job,
        samples=list(features[dataset].index),
    )

    if pypesto_problem is not None:
        evaluate_simulations(
            model=model,
            input_features=features[dataset].values,
            obj=pypesto_problem.objective.base_objective.base_objective,
            conf=conf,
            samples=list(features[dataset].index),
            petab_problem=pypesto_problem.objective.base_objective.amici_object_builder.petab_problem,
            dataset=dataset,
            outdir=outdir / "simulation",
            evaluations=evaluations,
            plot_file_prefix=EVALUATION_PLOT_FILE.format(
                dataset=dataset, **conf.to_dict()
            ),
        )

    return pd.DataFrame(evaluations), le_df, params_dev_df, params_df


# Load petab_base_files (once only)
petab_base_files = load_petab_base_files(conf)

# Get filepaths for features and feature transformation pipeline
features_filepath = get_features_filepath(conf, FEATURES_OUTFILE)

(
    _,
    problem,
    pypesto_subproblems,
    features,
) = process_features_and_setup_models(
    conf=conf,
    features_filepath=features_filepath,
    petab_base_files=petab_base_files,
    dataset="train+val",
)

model = load_model(conf, pypesto_subproblems["train"])

pypesto_problems = {
    dataset: create_pypesto_problem(pypesto_subproblems[dataset])
    for dataset in ["train", "val"]
}

for dataset in [
    "val",
    "train",
]:
    # clear jax cache to avoid error where jitted function uses input with shape of train
    # which differs from test
    jax.clear_caches()
    df, le_df, params_dev_df, params_df = evaluate_training(
        model=model,
        dataset=dataset,
        features=features,
        conf=conf,
        pypesto_problem=pypesto_problems[dataset],
    )
    for results, path_format in zip(
        [df, le_df, params_dev_df, params_df],
        [
            EVALUATION_TRAINING,
            EVALUATION_EMBEDDING,
            EVALUATION_PARAMETER_DEVIATIONS,
            EVALUATION_FULL_PARAMETERS,
        ],
    ):
        filepath = Path(path_format.format(dataset=dataset, **conf.to_dict()))
        filepath.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(filepath)

# ########################################################################### #
# ######################### Sensitivity Analysis ############################ #
# ########################################################################### #

parameters_interest = model.parameter_deviation_names

# Precompute pristine RMSEs once per dataset
rmse_pristine = {
    dataset: float(
        rmse(pypesto_problems[dataset], model, features[dataset].values)
    )
    for dataset in ["train", "val"]
}

# Parameter sensitivities
res_params = []
for param in tqdm(
    parameters_interest, desc=f"Parameter sensitivities (job={conf.job})"
):
    zeroed_model = set_param_to_zero(model, param)
    activated_model = activate_single_param(model, param)
    for dataset in ["train", "val"]:
        rmse_zeroed = rmse(
            pypesto_problems[dataset], zeroed_model, features[dataset].values
        )
        if activated_model is not None:
            rmse_activated = rmse(
                pypesto_problems[dataset],
                activated_model,
                features[dataset].values,
            )
        else:
            rmse_activated = rmse_pristine[dataset]
        res_params.append(
            {
                "param": param,
                "dataset": dataset,
                "rmse_pristine": rmse_pristine[dataset],
                "rmse_zeroed": rmse_zeroed,
                "rmse_activated": rmse_activated,
                "rmse_diff_zeroed": rmse_zeroed - rmse_pristine[dataset],
                "rmse_diff_activated": rmse_activated - rmse_pristine[dataset],
            }
        )

sens_params_df = pd.DataFrame(res_params)
sens_params_df["receptor_group"] = sens_params_df["param"].apply(
    classify_param
)

# Latent dimension sensitivities
res_latent = []
for latent_dimension in range(conf.n_hidden):
    latent_zeroed_model = zero_latent_direction(
        model, latent_dimension, latent_dim=conf.n_hidden
    )
    for dataset in ["train", "val"]:
        rmse_zeroed = rmse(
            pypesto_problems[dataset],
            latent_zeroed_model,
            features[dataset].values,
        )
        res_latent.append(
            {
                "latent_dimension": latent_dimension,
                "dataset": dataset,
                "rmse_pristine": rmse_pristine[dataset],
                "rmse_zeroed": rmse_zeroed,
                "rmse_diff": rmse_zeroed - rmse_pristine[dataset],
            }
        )

sens_latent_df = pd.DataFrame(res_latent)

# Save sensitivity results
for results, path_format in zip(
    [sens_params_df, sens_latent_df],
    [EVALUATION_SENSITIVITY_PARAMS, EVALUATION_SENSITIVITY_LATENT],
):
    for dataset in ["train", "val"]:
        filepath = Path(path_format.format(dataset=dataset, **conf.to_dict()))
        filepath.parent.mkdir(parents=True, exist_ok=True)
        results[results.dataset == dataset].to_csv(filepath)

print("Finished sensitivity analysis.")
