import warnings
from dataclasses import replace
from pathlib import Path

import fire
import numpy as np
import pandas as pd
import petab
from sklearn.model_selection import PredefinedSplit
from sklearn.pipeline import Pipeline

from common import (
    EVALUATION_REGRESSOR,
    FEATURES_OUTFILE,
    Wildcards,
    fig_dir,
    pretrain_dir,
    val_samples,
)
from dmm.analysis import process_simulation
from dmm.config_options import Conf
from dmm.feature_selection import load_data
from dmm.initialisation import (
    get_features_filepath,
    process_features,
)
from dmm.plotting import plot_cross_samples
from evaluation_utils import get_measurements_and_obervables
from regressor_training import train_pipeline
from training_configuration import SPLITS
from util import load_petab_base_files


def evaluate_standard_regression(
    input_data: pd.DataFrame,
    output_data: pd.DataFrame,
    dataset: str,
    conf: Conf,
    samples: list,
    mode: str,  # 'linreg', 'lasso', 'elasticnet'
    trained_pipeline: Pipeline,
) -> pd.DataFrame:
    if not len(input_data):
        return pd.DataFrame([])

    # Check the regressors have been trained
    if trained_pipeline is None:
        raise ValueError("No trained_pipeline provided for this regressor!")

    # Process regression output/predictions (reg_pred) and output data before plotting and evaluating simulations
    # Convert into pandas dataframe with the same index and column headers as output_test
    # Then process to use with plot_cross_samples() and process_simulation()
    # Finally drop index and rename column from 0 to 'simulation' to use in process_simulation()
    reg_pred = (
        pd.DataFrame(
            trained_pipeline.predict(input_data) if len(input_data) else [],
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
    ].apply(lambda x: x.replace(x.split("__")[0], ""))

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
    plot_name = mode + "_" + conf.context + "_" + conf.features
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
        context=conf.context,
        max_lrate=0,
        lrate_span=0,
        lrate_decay=0,
    )

    for sample in samples:
        process_simulation(
            evaluations=evaluations,
            measurement_df=output_data,
            simulation_df=reg_pred,
            conf=regr_conf,
            sample=sample,
        )

    return pd.DataFrame(evaluations)


conf = fire.Fire(Conf)

outdir = fig_dir / conf.model / conf.data
indir = pretrain_dir / conf.model / conf.data

# Suppress all DeprecationWarning warnings (coming from petab)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Get petab_base_files
petab_base_files = load_petab_base_files(conf)
del petab_base_files["condition_table"]

# Evaluate regressors
for mode in ["linreg", "lasso", "elasticnet"]:
    if (
        (conf.context == "MOSA")
        and (conf.features not in ("all",))
        and not conf.features.startswith("RFE_")
    ):
        raise ValueError(
            "MOSA context only available for 'all' features or RFE feature selection!"
        )

    # Load input features
    features_filepath = get_features_filepath(
        replace(
            conf,
            context=conf.context,
            features=conf.features,
        ),
        FEATURES_OUTFILE,
    )

    input_features_dict = process_features(
        conf=conf,
        features_filepath=features_filepath,
        datasets=["train", "val"],
    )

    samples_train, samples_val = [
        input_features_dict[dataset].index for dataset in ["train", "val"]
    ]

    # Load output features
    output_data_train, output_columns_train, _, imputer = load_data(
        contextualization="cytof_dynamic",
        samples=samples_train,
        features=None,
        **petab_base_files,
    )
    output_data_val, _, _, _ = load_data(
        contextualization="cytof_dynamic",
        samples=samples_val,
        features=output_columns_train,
        imputer=imputer,
        **petab_base_files,
    )

    print(
        f"Building pipeline and training estimator for {mode} on {conf.context}..."
    )
    # Select alpha over the DMM's own 5 CV folds: each split holds out its cell
    # line while the rest train, mirroring training_samples() per split. All five
    # lie inside the training set, so the held-out samples never enter the fit -
    # note val_samples("all") *is* common.test_samples.
    fold_of = {
        cell_line: fold
        for fold, split in enumerate(sorted(SPLITS))
        for cell_line in val_samples(Wildcards(conf.data, split))
    }
    test_fold = [
        fold_of.get(cell_line, -1)
        for cell_line in input_features_dict["train"].index
    ]
    n_folds = len({fold for fold in test_fold if fold >= 0})
    if n_folds < 2:
        raise ValueError(
            f"expected the {len(SPLITS)} DMM CV folds inside the training set, "
            f"found {n_folds}"
        )
    print(f"Selecting alpha over the DMM's {n_folds} CV folds")

    trained_pipeline, features_train = train_pipeline(
        input_data_train=input_features_dict["train"],
        output_data_train=output_data_train,
        pipeline_steps=[mode],
        cv=PredefinedSplit(test_fold=test_fold),
    )

    for dataset in ["train", "val"]:
        df = evaluate_standard_regression(
            input_data=input_features_dict[dataset],
            output_data=output_data_train
            if dataset == "train"
            else output_data_val,
            dataset=dataset,
            conf=conf,
            samples=input_features_dict[dataset].index,
            mode=mode,
            trained_pipeline=trained_pipeline,
        ).assign(features=conf.features)

        filepath = Path(
            EVALUATION_REGRESSOR.format(
                model=conf.model,
                data=conf.data,
                samples=conf.samples,
                dataset=dataset,
                mode=mode,
                context=conf.context,
                features=conf.features,
            )
        )
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath)

        # Added printout of RMSE on train/val datasets for each regressor (mode)
        if "res" in df.columns:
            rmse = np.sqrt(np.mean(np.square(df["res"])))
            print(
                f"RMSE for {mode} on {conf.samples}, {conf.context}, {dataset}, using {conf.features} features = {rmse}"
            )

    del (
        trained_pipeline,
        features_train,
    )
