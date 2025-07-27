from typing import Dict, List, Tuple, Union

import equinox as eqx
import jax.numpy as jnp
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


def make_dmm(*, dmm_params, features, key):
    return DeepMechanisticModel(
        **dmm_params,
        sample_name_list=list(features.index),
        n_input_features=features.shape[1],
        key=key,
    )


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


def get_median_param_names(model: DeepMechanisticModel):
    return [
        name[4:] if "MED" in name else name
        for name in model.pypesto_subproblem.x_names
        if "DEV" not in name
    ]


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
    features = impute_features(features)
    return features


def process_features_and_setup_models(
    conf: Conf,
    features_filepath: Union[str, List[str]],
    petab_base_files: Dict[str, pd.DataFrame],
    dataset: str = "train",
    return_features: bool = False,
) -> Union[
    Tuple[
        Union[
            Tuple[DeepMechanisticModel, DeepMechanisticModel],
            DeepMechanisticModel,
        ],
        CytofProblem,
    ],
    Tuple[
        Union[
            Tuple[DeepMechanisticModel, DeepMechanisticModel],
            DeepMechanisticModel,
        ],
        CytofProblem,
        Dict[str, pd.DataFrame],
    ],
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

    dmm_params = {
        "problem": problem,
        "dataset": conf.data,
        "n_latent": conf.n_hidden,
        "module_depth": conf.depth,
        "module_structure_multiplier": conf.nn_structure_multiplier,
        "use_layer_bias": conf.use_layer_bias,
        "last_layer_activation": conf.last_layer_activation,
        "weight_init_fn": conf.nn_init_fn,
        "bias_init_fn": "zeros",
        "orth_reg_strategy": conf.orth_reg_strategy,
        "activation_fn_name": conf.activation_fn_name,
        "reconstruct": conf.reconstruct,
        "n_threads": conf.threads,
        **petab_base_files,
    }

    key = jr.PRNGKey(conf.job)
    # Split keys for train/validation (otherwise identical weights, etc.)
    if len(datasets) > 1:
        keys = jr.split(key, num=len(datasets))
    else:
        keys = [key]

    dmms = (
        make_dmm(
            dmm_params=dmm_params,
            features=features,
            key=subkey,
        )
        for features, subkey in zip(
            [features[feature_dataset] for feature_dataset in datasets],
            keys,
        )
    )
    # TODO @GiacomoFabrini: add support for multiple datasets?
    result = (tuple(dmms), problem) if len(datasets) > 1 else (*dmms, problem)
    return (*result, features) if return_features else result


def get_kin_params_median_deviation(
    avg_model_parameter_file: str,
    random_seed: int,
):
    # Set random seed for poisson sampling
    np.random.seed(random_seed)
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

    # Poisson sample one among the multi-starts avg_model parameters and use as medians -- DISABLED
    # avg_param_combo = avg_model_params[
    #     avg_model_params.index == np.min([np.random.poisson(2, 1)[0], len(avg_model_params) - 1])
    #     ].iloc[0]

    # Use the best initialisation from avg_model (here second best)
    avg_param_combo = avg_model_params[avg_model_params.index == 1].iloc[0]
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


def subset_features(
    features: pd.DataFrame,
    model: DeepMechanisticModel,
    pypesto_subproblem: pypesto.Problem = None,
) -> np.ndarray:
    # extract sample names, ordering of those is important since samples
    # must match when reshaping the inflated matrix
    petab_samples = []
    if pypesto_subproblem is None:
        pypesto_subproblem = model.pypesto_subproblem
    # This filtering can be seen as redundant, as model.sample_name_list is set to petab_samples,
    # computed in the same way from the original features.index, but here we also make sure that such
    # samples appear in the features dataframe. That being said, if they are not, DMMs won't work because
    # they expect the full sample set
    # TODO discuss with Fabian - can be removed and a check for same length can be included instead (with ValueError)
    for name in pypesto_subproblem.x_names:
        if not name.startswith(MODEL_FEATURE_PREFIX):
            continue

        sample = name.split("__")[-1]
        if sample not in petab_samples and sample in features.index:
            petab_samples.append(sample)

    input_features = features.loc[petab_samples, :].values
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

    # Check order of initialised median parameters matches petab median parameter order
    assert list(par_medians.index.values) == get_median_param_names(model)

    # Initialise global kin parameters combiner with median values of non-cell-line-specific parameter components
    new_global_kin_params = jnp.array(par_medians.values)
    # Check shape match prior to initialisation
    if (
        new_global_kin_params.shape
        != model.kin_params_combiner.learned_global_kin_params.shape
    ):
        raise ValueError("Incorrect shape of new global kin parameters!")
    # Initialise KinParamsCombiner parameters
    model = eqx.tree_at(
        lambda m: m.kin_params_combiner.learned_global_kin_params,  # fetch weights from single layer of encoder
        model,
        new_global_kin_params,
    )
    return model


def get_features_filepath(
    conf: Conf,
    features_file_template: str,
) -> str:
    return features_file_template.format(
        **{**conf.__dict__, **{"dataset": "{dataset}"}}
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
