import fire
import itertools as itt
import numpy as np
import os
import pandas as pd
import petab

from common import (
    Conf,
    EVALUATION_REGRESSOR,
    REGR_TRAINED_PIPELINE,
    REGR_FEATURES_TRAIN,
    MEASUREMENTS_FILE,
    OBSERVABLES_FILE,
    Wildcards,
    fig_dir,
    pretrain_dir,
    test_samples,
    training_samples,
    CONTEXT_SET
)
from dmm.analysis import process_simulation
from dmm.feature_selection import load_data
from dmm.plotting import plot_cross_samples
from joblib import dump, load
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.linear_model import (
    LinearRegression,
    MultiTaskLassoCV,
    MultiTaskElasticNetCV
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from util import load_petab_base_files



conf = fire.Fire(Conf)

outdir = fig_dir / conf.model / conf.data
indir = pretrain_dir / conf.model / conf.data

# cross_sample_dir = outdir / "pretrain_cross_sample"
# cross_sample_dir.mkdir(exist_ok=True, parents=True)

# TODO @GiacomoFabrini: NEED TO CHANGE "train" to encompass "train" and "validation" (currently called
    # "test") from the splits. Change "test" to be the untouched "test" set. This is to ensure
    # that MultiTaskLassoCV and MultiTaskElasticNetCV have the same learning opportunities in
    # CV than the full DMM (i.e. their CV should be performed on train+val, not on train only)
samples = {
    "train": training_samples(Wildcards(conf.data, conf.samples)),
    "test": test_samples(Wildcards(conf.data, conf.samples)),
}


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
    output_data, _ = load_data(
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

def evaluate_standard_regression(
        dataset,
        conf,
        samples,
        context,
        mode,  # 'linreg', 'lasso', 'elasticnet'
        trained_pipeline,
        features_train,
):


    # Check the regressors have been trained
    if trained_pipeline is None:
        raise ValueError("No trained_pipeline provided for this regressor!")
    elif (dataset == "test") and (features_train is None):
        raise ValueError(f"No features_train provided for {dataset} evaluation!")

    # Load petab files
    petab_base_files = load_petab_base_files(conf, reweight=False)
    # so far data has not been reweighed when evaluating references, only for training
    del petab_base_files["condition_table"]

    # Subset to "train"/"test"
    samples_eval = samples[dataset]

    # Load input and output data
    input_data, _ = load_data(
        contextualization=context,
        samples=samples_eval,
        features=features_train if dataset=="test" else None,
        **petab_base_files,
    )
    output_data, _ = load_data(
        contextualization="cytof_dynamic",
        samples=samples_eval,
        features=None,
        **petab_base_files,
    )

    # Process regression output/predictions (reg_pred) and output data before plotting and evaluating simulations
    # Convert into pandas dataframe with same index and column headers as output_test
    # Then process to use with plot_cross_samples() and process_simulation()
    # Finally drop index and rename column from 0 to 'simulation' to use in process_simulation()
    reg_pred = pd.DataFrame(
        trained_pipeline.predict(input_data),
        index=output_data.index,
        columns=output_data.columns
    ).T.stack().reset_index().sort_values(
        by=[
            'preequilibrationConditionId',
            'observableId',
            'simulationConditionId',
            'time'
        ]
    ).reset_index().drop(columns='index').rename(columns={0: "simulation"})

    # output_data
    # Column needs renaming from 0 to "measurement" for use in process_simulation()
    output_data = output_data.T.stack().reset_index().sort_values(
        by=[
            'preequilibrationConditionId',
            'observableId',
            'simulationConditionId',
            'time'
        ]
    ).reset_index().drop(columns='index').rename(columns={0: "measurement"})


    # Produce plots to analyse performance
    # import original output data as in avg/avg_model
    df_meas = pd.read_csv(
        MEASUREMENTS_FILE.format(**conf.__dict__), sep="\t", index_col=0
    )
    df_obs = pd.read_csv(
        OBSERVABLES_FILE.format(**conf.__dict__), sep="\t", index_col=0
    )
    df_meas = df_meas[
        df_meas[petab.OBSERVABLE_ID].apply(lambda x: x in df_obs.index)
    ]
    # Groupby to average replicates as done for regression output
    df_meas = df_meas.groupby(
        ['observableId', 'preequilibrationConditionId', 'time', 'simulationConditionId']).agg(
        {'measurement': 'mean', 'noiseParameters': 'mean'}).reset_index()
    # Sort to make comparable with output_train
    df_meas = df_meas.sort_values(
        by=['observableId', 'preequilibrationConditionId', 'simulationConditionId', 'time'])

    # Subset to cell lines that are in output_data (i.e. output_train/output_test)
    df_meas = df_meas[
        df_meas.preequilibrationConditionId.isin(output_data.preequilibrationConditionId)
    ].reset_index().drop(columns='index')

    # process simulation condition id
    df_meas[petab.SIMULATION_CONDITION_ID] = df_meas[
        petab.SIMULATION_CONDITION_ID
    ].apply(lambda x: x.split("__")[1])

    # reorder columns as in output_train
    df_meas = df_meas[['observableId', 'simulationConditionId',
                       'time', 'preequilibrationConditionId',
                       'measurement', 'noiseParameters']]

    # Plot -- reg_pred is either reg_pred_train or reg_pred_test
    plot_name = mode + "_" + context
    plot_cross_samples(
        df_meas, reg_pred, outdir / "simulation" / dataset, plot_name
    )

    # Process simulations/regressions, i.e. produce CSVs with residuals
    evaluations = []

    for sample in samples[dataset]:
        process_simulation(
            evaluations=evaluations,
            measurement_df=output_data,
            simulation_df=reg_pred,
            context=context,
            sample=sample,
            model_type=mode,
            orth_reg_strategy="None",  # not needed for regression
            job=None, # not needed here
            l1reg_inflate=0.0,
            oreg_encode=0.0,
            l1reg_encode=0.0,
            oreg_inflate=0.0,
            hidden_layers=0,
            features="none",
        )

    return pd.DataFrame(evaluations)

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
        print(f"Building pipeline and training estimator for {mode} on {context}...")
        trained_pipeline, features_train = train_pipeline(
            pipeline_steps=["pca", mode],
            conf=conf,
            context=context,
            samples_train=samples["train"],
        )
        dump(trained_pipeline, trained_pipeline_file)
        dump(features_train, features_train_file)


    df = evaluate_standard_regression(
        dataset=dataset,
        conf=conf,
        samples=samples,
        context=context,
        mode=mode,
        trained_pipeline=trained_pipeline,
        features_train=features_train
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
