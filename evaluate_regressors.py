import itertools as itt
import numpy as np
import os
from typing import Dict, List

import fire
import pandas as pd
import petab
import warnings

from joblib import dump, load
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.linear_model import (
    LinearRegression,
    MultiTaskElasticNetCV,
    MultiTaskLassoCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import (
    CONTEXT_SET,
    EVALUATION_REGRESSOR,
    REGR_FEATURES_TRAIN,
    REGR_TRAINED_PIPELINE,
    Wildcards,
    fig_dir,
    pretrain_dir,
    test_samples,
    training_samples,
)
from dmm.analysis import process_simulation
from dmm.config_options import Conf
from dmm.feature_selection import load_data
from dmm.plotting import plot_cross_samples
from evaluation_utils import get_measurements_and_obervables
from util import load_petab_base_files

conf = fire.Fire(Conf)

outdir = fig_dir / conf.model / conf.data
indir = pretrain_dir / conf.model / conf.data

# cross_sample_dir = outdir / "pretrain_cross_sample"
# cross_sample_dir.mkdir(exist_ok=True, parents=True)

# TODO @GiacomoFabrini: NEED TO CHANGE "train" to encompass "train" and "validation" (currently called
#  "test") from the splits. Change "test" to be the untouched "test" set. This is to ensure
#  that MultiTaskLassoCV and MultiTaskElasticNetCV have the same learning opportunities in
#  CV than the full DMM (i.e. their CV should be performed on train+val, not on train only)
samples = {
    "train": training_samples(Wildcards(conf.data, conf.samples)),
    "test": test_samples(Wildcards(conf.data, conf.samples)),
}

# Suppress all DeprecationWarning warnings (coming from petab)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def build_pipeline(
    steps_list: List[str],
    # input_data: np.ndarray,  # not needed if using PCA(n_components=0.95)
) -> Pipeline:
    """Builds a sklearn.pipeline.Pipeline consisting of:
    - StandardScaler(),
    - KNNImputer(),
    - additional steps in steps_list

    :param steps_list:
        list of additional Pipeline steps

    # :param input_data:
    #     input_data used to fit PCA step in Pipeline
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
    if (steps_list is not None) and (len(steps_list) > 0):
        for step in steps_list:
            if step == "pca":
                # inputs = Pipeline(steps).fit_transform(input_data)
                # var_expl = (
                #     PCA(n_components=input_data.shape[0])
                #     .fit(inputs)
                #     .explained_variance_ratio_
                # )
                # n_pca = np.nonzero(np.cumsum(var_expl) > 0.95)[0][0] + 1
                # steps.append(("pca", PCA(n_components=n_pca)))
                steps.append(
                    ("pca", PCA(n_components=0.95, whiten=True))
                )  # added whitening
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
    pipeline_steps: List[str],
    petab_base_files: Dict[str, pd.DataFrame],
    context: str,
    samples_train,
    impute_missing_output: bool = True,
):
    """Trains a sklearn.pipeline.Pipeline built via build_pipeline()

    :param pipeline_steps:
        list of Pipeline steps to be passed to build_pipeline()

    :param conf:
        configuration

    :param context:
        contextualisation - can be cytof_init/proteomics/transcriptomics

    :param samples_train:
        data to train the regressor Pipeline on

    :param impute_missing_output:
        whether to impute missing data in output_data during pipeline training
    """
    # Load input and output data
    input_data, features_train = load_data(
        contextualization=context,
        samples=samples_train,
        features=None,
        **petab_base_files,
    )
    output_data, _ = load_data(
        contextualization="cytof_dynamic",
        samples=samples_train,
        features=None,
        **petab_base_files,
    )
    if impute_missing_output:
        # Impute missing data in output_data during pipeline training
        output_data = KNNImputer().fit_transform(output_data)
    # Build pipeline and return trained_pipeline, features_train
    pipeline = build_pipeline(
        steps_list=pipeline_steps,
        # input_data=input_data
    )

    return pipeline.fit(input_data, output_data), features_train


def evaluate_standard_regression(
    dataset: str,
    conf: Conf,
    samples,
    context: str,
    mode: str,  # 'linreg', 'lasso', 'elasticnet'
    trained_pipeline: Pipeline,
    features_train,
    features_test,
    petab_base_files: Dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, list]:
    # Check the regressors have been trained
    if trained_pipeline is None:
        raise ValueError("No trained_pipeline provided for this regressor!")
    elif (dataset == "test") and (features_train is None):
        raise ValueError(
            f"No features_train provided for {dataset} evaluation!"
        )

    # Subset to "train"/"test"
    samples_eval = samples[dataset]

    # TODO @GiacomoFabrini - need to fix this! Always consistent features between train and test!
    # Load input and output data
    input_data, _ = load_data(
        contextualization=context,
        samples=samples_eval,
        features=features_train if dataset == "test" else None,
        **petab_base_files,
    )
    output_data, test_columns = load_data(
        contextualization="cytof_dynamic",
        samples=samples_eval,
        features=features_test if dataset == "test" else None,
        **petab_base_files,
    )

    # Process regression output/predictions (reg_pred) and output data before plotting and evaluating simulations
    # Convert into pandas dataframe with same index and column headers as output_test
    # Then process to use with plot_cross_samples() and process_simulation()
    # Finally drop index and rename column from 0 to 'simulation' to use in process_simulation()
    reg_pred = (
        pd.DataFrame(
            trained_pipeline.predict(input_data),
            index=output_data.index,
            columns=output_data.columns,
        )
        .T.stack()
        .reset_index()
        .sort_values(
            by=[
                petab.PREEQUILIBRATION_CONDITION_ID,
                petab.OBSERVABLE_ID,
                petab.SIMULATION_CONDITION_ID,
                petab.TIME,
            ]
        )
        .reset_index()
        .drop(columns="index")
        .rename(columns={0: "simulation"})
    )

    # output_data
    # Column needs renaming from 0 to "measurement" for use in process_simulation()
    output_data = (
        output_data.T.stack()
        .reset_index()
        .sort_values(
            by=[
                petab.PREEQUILIBRATION_CONDITION_ID,
                petab.OBSERVABLE_ID,
                petab.SIMULATION_CONDITION_ID,
                petab.TIME,
            ]
        )
        .reset_index()
        .drop(columns="index")
        .rename(columns={0: "measurement"})
    )

    # Produce plots to analyse performance
    # import original output data as in avg/avg_model
    df_meas, df_obs = get_measurements_and_obervables(conf)
    # Groupby to average replicates as done for regression output
    df_meas = (
        df_meas.groupby(
            [
                petab.OBSERVABLE_ID,
                petab.PREEQUILIBRATION_CONDITION_ID,
                petab.TIME,
                petab.SIMULATION_CONDITION_ID,
            ]
        )
        .agg({"measurement": "mean", "noiseParameters": "mean"})
        .reset_index()
    )
    # Sort to make comparable with output_train
    df_meas = df_meas.sort_values(
        by=[
            petab.OBSERVABLE_ID,
            petab.PREEQUILIBRATION_CONDITION_ID,
            petab.SIMULATION_CONDITION_ID,
            petab.TIME,
        ]
    )

    # Subset to cell lines that are in output_data (i.e. output_train/output_test)
    df_meas = (
        df_meas[
            df_meas.preequilibrationConditionId.isin(
                output_data.preequilibrationConditionId
            )
        ]
        .reset_index()
        .drop(columns="index")
    )

    # process simulation condition id
    df_meas[petab.SIMULATION_CONDITION_ID] = df_meas[
        petab.SIMULATION_CONDITION_ID
    ].apply(lambda x: x.split("__")[1])

    # reorder columns as in output_train
    df_meas = df_meas[
        [
            petab.OBSERVABLE_ID,
            petab.SIMULATION_CONDITION_ID,
            petab.TIME,
            petab.PREEQUILIBRATION_CONDITION_ID,
            petab.MEASUREMENT,
            petab.NOISE_PARAMETERS,
        ]
    ]

    # Plot -- reg_pred is either reg_pred_train or reg_pred_test
    plot_name = mode + "_" + context
    plot_cross_samples(
        df_meas, reg_pred, outdir / "simulation" / dataset, plot_name
    )

    # Process simulations/regressions, i.e. produce CSVs with residuals
    evaluations = []

    # instantiate a replacement conf for regressors,
    # only setting to 0 those parameters that are not already 0 by default
    # and ensuring to add context information
    regr_conf = Conf(
        model=conf.model,
        data=conf.data,
        context=context,
        max_lrate=0,
        lrate_span=0,
        lrate_decay=0,
    )

    for sample in samples[dataset]:
        process_simulation(
            evaluations=evaluations,
            measurement_df=output_data,
            simulation_df=reg_pred,
            conf=regr_conf,
            sample=sample,
        )

    return pd.DataFrame(evaluations), test_columns


# Get petab_base_files
petab_base_files = load_petab_base_files(conf)
del petab_base_files["condition_table"]

features_test = {
    context: None
    for context in CONTEXT_SET
}

# Evaluate regressors
for dataset, context, mode in itt.product(
    ["train", "test"], CONTEXT_SET, ["linreg", "lasso", "elasticnet"]
):
    trained_pipeline_file = REGR_TRAINED_PIPELINE.format(
        model=conf.model,
        data=conf.data,
        samples=conf.samples,
        mode=mode,
        context=context,
    )

    features_train_file = REGR_FEATURES_TRAIN.format(
        model=conf.model,
        data=conf.data,
        samples=conf.samples,
        mode=mode,
        context=context,
    )

    # if both pipeline and features exist, load them and proceed
    if os.path.exists(trained_pipeline_file) and os.path.exists(features_train_file):
        trained_pipeline = load(trained_pipeline_file)
        features_train = load(features_train_file)
    # else build and train the pipeline and extract the features
    else:
        print(
            f"Building pipeline and training estimator for {mode} on {context}..."
        )
        trained_pipeline, features_train = train_pipeline(
            pipeline_steps=["pca", mode],
            petab_base_files=petab_base_files,
            context=context,
            samples_train=samples["train"],
        )
        dump(trained_pipeline, trained_pipeline_file)
        dump(features_train, features_train_file)

    df, test_columns = evaluate_standard_regression(
        dataset=dataset,
        conf=conf,
        samples=samples,
        context=context,
        mode=mode,
        trained_pipeline=trained_pipeline,
        features_train=features_train,
        features_test=features_test[context],
        petab_base_files=petab_base_files,
    )

    df.to_csv(
        EVALUATION_REGRESSOR.format(
            model=conf.model,
            data=conf.data,
            samples=conf.samples,
            dataset=dataset,
            mode=mode,
            context=context,
        )
    )

    # Update features_test with test_columns to ensure consistency between train and test columns
    if features_test[context] is None:
        features_test[context] = test_columns

    # Added printout of RMSE on train/val datasets for each regressor (mode)
    rmse = np.sqrt(np.mean(np.square(df["res"])))
    print(f"RMSE for {mode} on {conf.samples}, {context}, {dataset} = {rmse}")

    del trained_pipeline, features_train, trained_pipeline_file, features_train_file
