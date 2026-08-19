from typing import List

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.linear_model import (
    LinearRegression,
    MultiTaskElasticNetCV,
    MultiTaskLassoCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_pipeline(
    steps_list: List[str],
    cv=5,
) -> Pipeline:
    """
    Builds a sklearn.pipeline.Pipeline consisting of:
    - StandardScaler(),
    - KNNImputer(),
    - additional steps in steps_list

    :param steps_list:
        list of additional Pipeline steps

    :param cv:
        cross-validation strategy handed to the alpha-selecting regressors.
        Anything accepted by scikit-learn; defaults to 5-fold.
    """
    # standard steps: scaling, imputation via KNN
    steps = [
        ("scaler", StandardScaler()),
        ("imputer", KNNImputer()),
    ]

    # regressor steps
    regressor_steps = {
        # LinearRegression automatically supports MultiOutput/MultiTask
        "linreg": LinearRegression(),
        # `alphas` takes the count directly; `n_alphas` is deprecated in
        # scikit-learn 1.7 and removed in 1.9
        "lasso": MultiTaskLassoCV(cv=cv, alphas=20),
        "elasticnet": MultiTaskElasticNetCV(cv=cv, alphas=20),
    }

    # PCA + one among linear regression/lasso/elasticnet
    if (steps_list is not None) and (len(steps_list) > 0):
        for step in steps_list:
            if step == "pca":
                steps.append(
                    ("pca", PCA(n_components=0.95, whiten=True))
                )  # added whitening
            elif step in regressor_steps.keys():
                steps.append((step, regressor_steps[step]))
            else:
                raise ValueError(f"Unknown step {step}")
    else:
        if not steps_list:
            if steps_list is None:
                raise TypeError(
                    "Expected type list for steps_list, got None type"
                )
            else:
                raise ValueError("List of pipeline steps is empty")
    return Pipeline(steps)


def train_pipeline(
    input_data_train: pd.DataFrame,
    output_data_train: pd.DataFrame,
    pipeline_steps: List[str],
    impute_missing_output: bool = True,
    cv=5,
):
    """
    Trains a sklearn.pipeline.Pipeline built via build_pipeline()

    :param input_data_train:
        input data to train the regressor Pipeline on.

    :param output_data_train:
        output data to train the regressor Pipeline on.

    :param pipeline_steps:
        list of Pipeline steps to be passed to build_pipeline()

    :param impute_missing_output:
        whether to impute missing data in output_data during pipeline training

    :param cv:
        cross-validation strategy for the alpha-selecting regressors. Callers
        pass the DMM's own CV folds, so alpha is chosen the way the DMM's
        configuration is; the fit itself stays on the training data.
    """
    if impute_missing_output:
        # Impute missing data in output_data during pipeline training
        output_data_train = KNNImputer().fit_transform(output_data_train)

    pipeline = build_pipeline(steps_list=pipeline_steps, cv=cv)

    return pipeline.fit(
        input_data_train, output_data_train
    ), input_data_train.columns
