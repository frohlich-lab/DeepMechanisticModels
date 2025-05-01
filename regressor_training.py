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
from typing import List


def build_pipeline(
    steps_list: List[str],
) -> Pipeline:
    """
    Builds a sklearn.pipeline.Pipeline consisting of:
    - StandardScaler(),
    - KNNImputer(),
    - additional steps in steps_list

    :param steps_list:
        list of additional Pipeline steps
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
        "lasso": MultiTaskLassoCV(cv=5, n_alphas=20),
        "elasticnet": MultiTaskElasticNetCV(cv=5, n_alphas=20),
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
                raise TypeError("Expected type list for steps_list, got None type")
            else:
                raise ValueError("List of pipeline steps is empty")
    return Pipeline(steps)


def train_pipeline(
    input_data_train: pd.DataFrame,
    output_data_train: pd.DataFrame,
    pipeline_steps: List[str],
    impute_missing_output: bool = True,
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
    """

    if impute_missing_output:
        # Impute missing data in output_data during pipeline training
        output_data_train = KNNImputer().fit_transform(output_data_train)

    # Build pipeline and return trained_pipeline, features_train
    pipeline = build_pipeline(
        steps_list=pipeline_steps,
        # input_data=input_data
    )

    return pipeline.fit(input_data_train, output_data_train), input_data_train.columns
