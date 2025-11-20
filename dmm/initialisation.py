from typing import Dict, List, Union

import equinox as eqx
import jax.random as jr
import numpy as np
import pandas as pd
import pypesto
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

from cytof.problem import CytofProblem

from . import MEDIAN_FEATURE_PREFIX, MODEL_FEATURE_PREFIX
from .config_options import Conf
from .dmm_autoencoder_eqx import DeepMechanisticModel
from .petab_subproblem import load_petab


def get_features(
    features_filepath: str, datasets: List[str]
) -> Dict[str, pd.DataFrame]:
    features = {
        dataset: pd.read_csv(
            features_filepath.format(dataset=dataset), index_col=0
        ).sort_index()  # once again, ensure cell-lines appear in alphabetical order
        for dataset in datasets
    }
    return features


def process_features(conf: Conf, features_filepath: str, datasets: List[str]):
    # loads features corresponding to requested dataset settings
    features = get_features(
        features_filepath=features_filepath, datasets=datasets
    )
    if conf.standardise_features:
        if "train" in features:
            scaler = StandardScaler()
            scaler.fit(features["train"])
            features = {
                dataset: pd.DataFrame(
                    scaler.transform(features[dataset]),
                    index=features[dataset].index,
                    columns=features[dataset].columns,
                )
                for dataset in features.keys()
            }
        else:
            raise ValueError(
                "Standard scaling is only supported when 'train' dataset is provided!"
            )
    return features


def process_features_and_setup_models(
    conf: Conf,
    features_filepath: Union[str, List[str]],
    petab_base_files: Dict[str, pd.DataFrame],
    dataset: str = "train",
) -> tuple[
    DeepMechanisticModel,
    CytofProblem,
    dict[str, pypesto.Problem | None],
    dict[str, pd.DataFrame],
]:
    problem = CytofProblem(conf.model)

    datasets = dataset.split("+")

    features = process_features(
        conf=conf,
        features_filepath=features_filepath,
        datasets=datasets,
    )

    # Check features arrays are two-dimensional
    for key in features.keys():
        if features[key].values.ndim != 2:
            raise ValueError(
                f"features for `{key}` were expected to be two-dimensional, "
                f"but were {features[key].values.ndim}-dimensional!"
            )

    pypesto_problems = {}
    for dataset in datasets:
        features_dataset = features[dataset]
        samples = list(features_dataset.index)
        if not samples:
            pypesto_problems[dataset] = None
            continue
        petab_importer = load_petab(
            problem=problem,
            dataset=conf.data,
            **petab_base_files,
            samples=samples,
        )
        factory = petab_importer.create_objective_creator()
        objective = factory.create_objective()
        problem.apply_objective_settings(objective, n_threads=conf.threads)
        pypesto_problems[dataset] = petab_importer.create_problem(
            objective=objective,
        )
    dmm = DeepMechanisticModel(
        pypesto_problem=pypesto_problems["train"],
        n_input_features=features["train"].shape[1],
        conf=conf,
        key=jr.PRNGKey(conf.job),
    )
    return dmm, problem, pypesto_problems, features


def get_kin_params_median_deviation(
    avg_model_parameter_file: str,
    random_seed: int,
):
    # Set random seed for poisson sampling, allow 10 different seeds
    np.random.seed(random_seed % 10)
    # Fetch avg_model params (for all multi-starts)
    avg_model_params = pd.read_csv(
        avg_model_parameter_file,
        index_col=[0],
    )
    # Subset to columns not starting with MODEL_FEATURE_PREFIX
    avg_model_params = avg_model_params[
        [
            col
            for col in avg_model_params.columns
            if not col.startswith(MODEL_FEATURE_PREFIX)
        ]
    ]
    avg_model_params.rename(
        columns=lambda col: col[len(MEDIAN_FEATURE_PREFIX) :]
        if col.startswith(MEDIAN_FEATURE_PREFIX)
        else col,
        inplace=True,
    )

    # Poisson sample one among the multi-starts avg_model parameters and use as medians
    avg_param_combo = avg_model_params[
        avg_model_params.index
        == np.min([np.random.poisson(2, 1)[0], len(avg_model_params) - 1])
    ].iloc[0]

    # Use the best initialisation from avg_model (here second best)
    # avg_param_combo = avg_model_params[avg_model_params.index == 1].iloc[0]
    return avg_param_combo


# def load_and_subset_input_features(
#         conf: Conf,
#         model: DeepMechanisticModel,
#         dataset: str,
#         pypesto_subproblem: pypesto.Problem = None,
# ):
#     features = pd.read_csv(
#         FEATURES_OUTFILE.format_map(dict(**conf.__dict__, dataset=dataset)),
#         index_col=0,
#     )
#     # extract sample names, ordering of those is important since samples
#     # must match when reshaping the inflated matrix
#     petab_samples = []
#     if pypesto_subproblem is None:
#         pypesto_subproblem = model.pypesto_subproblem
#     for name in pypesto_subproblem.x_names:
#         if not name.startswith(MODEL_FEATURE_PREFIX):
#             continue
#
#         sample = name.split("__")[-1]
#         if sample not in petab_samples and sample in features.index:
#             petab_samples.append(sample)
#
#     input_features = features.loc[petab_samples, :].values
#     return input_features


def sort_features(
    features: pd.DataFrame,
    pypesto_problem: pypesto.Problem | None,
) -> np.ndarray:
    # extract sample names, ordering of those is important since samples
    # must match when reshaping the inflated matrix
    if pypesto_problem is None:
        return features.values
    ref_par = pypesto_problem.x_names[0].replace(MEDIAN_FEATURE_PREFIX, "")
    samples = [
        par.split("__")[-1]
        for par in pypesto_problem.x_names
        if par.startswith(MODEL_FEATURE_PREFIX + ref_par)
    ]
    input_features = features.loc[samples, :].values
    return input_features


def init_global_kin_params_combiner(
    model: DeepMechanisticModel,
    avg_model_parameter_file: str,
    random_seed: int,
) -> DeepMechanisticModel:
    """
    Setup KinParamsCombiner module (model.kin_params_combiner).
    The parameters of KinParamsCombiner are initialised with the median of non-cell-line specific parameters (learnable).

    :param model:
        DeepMechanisticModel instance.
    :param avg_model_parameter_file:
        filepath (str) to load avg_model kinetic parameters in case median_params_method == "avg_model"
    :param random_seed:
        int, used to seed sampling of kinetic parameters in case median_params_method == "per_sample"

    :returns: updated model with initialised model.kin_params_combiner
    """

    par_medians = get_kin_params_median_deviation(
        avg_model_parameter_file=avg_model_parameter_file,
        random_seed=random_seed,
    )
    model = eqx.tree_at(
        lambda m: m.kin_params_combiner.learned_global_kin_params,  # fetch weights from single layer of encoder
        model,
        par_medians.loc[list(model.parameter_median_names)].values,
    )
    return model


def get_features_filepath(
    conf: Conf,
    features_file_template: str,
) -> str:
    return features_file_template.format(
        **{**conf.to_dict(), **{"dataset": "{dataset}"}}
    )


def impute_features(features: dict) -> dict:
    # Simply impute missing values (no scaling, no PCA)
    imputer = KNNImputer()
    imputer.fit(features["train"])
    features = {
        dataset: pd.DataFrame(
            imputer.transform(features[dataset]),
            index=features[dataset].index,
            columns=features[dataset].columns,
        )
        for dataset in features.keys()
    }
    return features
