from dataclasses import dataclass, replace
from pathlib import Path

import fire
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import FEATURES_OUTFILE, Wildcards, test_samples, training_samples
from dmm.feature_selection import (
    build_preprocessor,
    load_data,
    preprocess_mosa_latent,
)
from training_configuration import SPLITS
from util import load_petab_base_files


@dataclass(init=True)
class MinimalConf(dict):
    model: str
    data: str
    context: str
    features: str
    samples: str


def get_feature_importances(model, X, y, method="auto"):
    """
    Get feature importances for RandomForest or ElasticNet/LogisticRegression models.

    Parameters:
    - model: fitted sklearn model
    - method: 'auto', 'coef', 'tree', 'permutation'
    - X: input features (optional, needed for permutation importance)
    - y: target variable (optional, needed for permutation importance)

    Returns:
    - feature_importances: np.ndarray of shape (n_features,)
    """
    if method == "tree":
        return model.named_steps["regressor"].feature_importances_

    elif method == "permute":
        if X is None or y is None:
            raise ValueError(
                "X and y must be provided for permutation importance."
            )
        result = permutation_importance(
            model, X, y, n_repeats=10, random_state=42
        )
        return result.importances_mean

    else:
        raise ValueError(f"Unknown method: {method}")


def get_selected_features(
    input_data,
    output_data,
    context: str,
    features: str,
    features_all: list,
    cv=None,
):
    if features == "all":
        return features_all

    elif features.startswith("RFE_") or features.startswith("HVGRFE_"):
        reduce_factor = 0.80
        # drop nans
        input_data = input_data.dropna(axis=1, how="any")
        if features.startswith("HVG") and context in [
            "proteomics",
            "transcriptomics",
        ]:
            # remove 20% of features with lowest mean:
            means = np.mean(input_data, axis=0)
            threshold = np.percentile(means, 20)
            input_data = input_data.loc[:, means >= threshold]
            var_threshold = sorted(
                np.nanvar(input_data, axis=0), reverse=True
            )[500]
            input_data = input_data.loc[
                :, np.nanvar(input_data, axis=0) >= var_threshold
            ]

        n_features = int(features.split("_")[1])
        method = features.split("_")[2]
        random_state = 42  # For reproducibility
        estimator = RandomForestRegressor(
            random_state=random_state,
            max_features=reduce_factor,
        )
        pipeline = Pipeline(
            [("scaler", StandardScaler()), ("regressor", estimator)]
        )
        while input_data.shape[1] * reduce_factor > n_features:
            pipeline = pipeline.fit(input_data, output_data)
            y_pred = pipeline.predict(input_data)
            rmse = np.sqrt(np.mean(np.square(output_data.values - y_pred)))
            importances = get_feature_importances(
                pipeline, input_data, output_data, method=method
            )

            n_features_target = int(np.ceil(len(importances) * reduce_factor))
            if n_features_target == input_data.shape[1]:
                n_features_target -= 1  # reduce by at least one feature
            indices = np.argsort(importances)[::-1][:n_features_target]
            input_data = input_data.iloc[:, indices]
            print(
                f"Reduced features to: {input_data.shape[1]:>5} ({rmse:.2f})",
            )
        # Fit the final model with the selected features
        pipeline = pipeline.fit(input_data, output_data)
        importances = get_feature_importances(
            pipeline, input_data, output_data, method=method
        )
        indices = np.argsort(importances)[::-1][:n_features]
        return input_data.columns[indices]

    preprocessor = build_preprocessor(features, input_data, output_data, cv=cv)
    preprocessor = preprocessor.fit(input_data, output_data)

    return preprocessor.feature_names_in_[
        preprocessor.steps[-1][1].get_support()
    ]


conf = fire.Fire(MinimalConf)
petab_base_files = load_petab_base_files(conf)
del petab_base_files["condition_table"]

if (conf.context == "MOSA") and ("4of5" == conf.samples):
    raise ValueError(f"{conf.context} not available for CV split")

samples_train = {
    split: sorted(training_samples(Wildcards(conf.data, split)))
    for split in sorted(SPLITS)
}
samples_val = {
    split: sorted(test_samples(Wildcards(conf.data, split)))
    for split in sorted(SPLITS)
}

for context in conf.context.split("+"):
    subconf = replace(conf, context=context)

    input_parts = []
    output_parts = []
    features_all = None
    all_indices = []
    split_indices = []

    if subconf.context == "MOSA":
        input_train, input_val, features_all = preprocess_mosa_latent(
            subconf, samples_train[conf.samples], samples_val[conf.samples]
        )
    else:
        input_train, features_all = load_data(
            contextualization=context,
            samples=samples_train[conf.samples],
            features=None,
            **petab_base_files,
        )
        input_val, _ = load_data(
            contextualization=context,
            samples=samples_val[conf.samples],
            features=features_all,
            **petab_base_files,
        )

    output_train, features_output_train = load_data(
        contextualization="cytof_dynamic",
        samples=samples_train[conf.samples],
        features=None,
        **petab_base_files,
    )
    imputer = KNNImputer()
    filled = imputer.fit_transform(output_train)
    output_train = pd.DataFrame(
        filled,
        index=output_train.index,
        columns=output_train.columns,
    )

    selected_features = get_selected_features(
        input_data=input_train,
        output_data=output_train,
        context=subconf.context,
        features=conf.features,
        features_all=features_all,
        cv=None,
    )
    print(
        f"Selected {len(selected_features)} features for split {conf.samples} for {subconf.context}: {selected_features}"
    )

    # Transform and save per split
    if subconf.context == "MOSA":
        input_train, input_val, _ = preprocess_mosa_latent(
            subconf, samples_train, samples_val
        )

    for dataset, inputs in zip(("train", "val"), (input_train, input_val)):
        outfile = FEATURES_OUTFILE.format_map(
            dict(**subconf.__dict__, dataset=dataset)
        )
        Path(outfile).parent.mkdir(exist_ok=True, parents=True)
        print(
            f"Preprocessing {dataset} data for split {conf.samples} to {outfile}"
        )
        df_inputs = pd.DataFrame(
            inputs[selected_features].values,
            index=inputs.index,
            columns=selected_features,
        )
        df_inputs.to_csv(outfile)
