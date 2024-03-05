import numpy as np
import pandas as pd
import petab
from sklearn.linear_model import (LinearRegression,
                                      MultiTaskLassoCV,
                                      MultiTaskElasticNetCV)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from util import load_petab_base_files
from dmm.feature_selection import load_data


def build_pipeline(
        steps_list: str,
        input_data: np.ndarray,
):
    """
    builds a sklearn.pipeline.Pipeline consisting of:
    - StandardScaler(),
    - KNNImputer(),
    - additional steps in steps_list

    :param steps_list:
        list of additional Pipeline steps

    :param input_data:
        input_data used to fit PCA step in Pipeline
    """

    # standard steps: scaling, imputation via KNN
    steps = [
        ("scaler", StandardScaler()),
        ("imputer", KNNImputer()),
    ]

    # regressor steps
    regressor_steps = {
        # seems like LinearRegression automatically supports MultiOutput/MultiTask
        "linreg": LinearRegression(),
        "lasso": MultiTaskLassoCV(cv=5, n_alphas=20),
        "elasticnet": MultiTaskElasticNetCV(cv=5, n_alphas=20),
    }

    # PCA + one among linear regression/lasso/elasticnet
    if (steps_list is not None) and (len(steps_list )>0):
        for step in steps_list:
            if step == "pca":
                inputs = Pipeline(steps).fit_transform(input_data)
                var_expl = (
                    PCA(n_components=input_data.shape[0])
                    .fit(inputs)
                    .explained_variance_ratio_
                )
                n_pca = np.nonzero(np.cumsum(var_expl) > 0.95)[0][0] + 1
                steps.append(("pca", PCA(n_components=n_pca)))
            elif step in regressor_steps.keys():
                steps.append((step, regressor_steps[step]))
            else:
                raise ValueError(f"Unknown step {step}")
    else:
        if steps_list is None:
            raise TypeError("Expected type list for steps_list, got None type")
        elif len(steps_list) == 0:
            raise ValueError("List of pipeline steps is empty")

    return Pipeline(steps)

def train_pipeline(
        pipeline_steps,
        conf,
        context,
        samples_train,
):
    """
    trains a sklearn.pipeline.Pipeline built via build_pipeline()

    :param pipeline_steps:
        list of Pipeline steps to be passed to build_pipeline()

    :param conf:
        configuration

    :param context:
        contextualisation - can be cytof_init/proteomics/transcriptomics

    :param samples_train:
        data to train the regressor Pipeline on
    """

    # Load petab files
    petab_base_files = load_petab_base_files(conf, reweight=False)
    del petab_base_files["condition_table"]

    # Load input and output data
    input_data, features_train = load_data(
        contextualization=context,
        samples=samples_train,
        features=None,
        **petab_base_files,
    )
    output_data, _, _ = load_data(
        contextualization="cytof_dynamic",
        samples=samples_train,
        features=None,
        **petab_base_files,
    )
    # Build pipeline and return trained_pipeline, features_train
    pipeline = build_pipeline(
                steps_list=pipeline_steps,
                input_data=input_data
            )

    return pipeline.fit(input_data, output_data), features_train