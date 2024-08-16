import numpy as np
import pandas as pd
import petab
from sklearn.cross_decomposition import CCA, PLSRegression
from sklearn.decomposition import PCA, SparsePCA
from sklearn.feature_selection import (
    RFECV,
    SelectFromModel,
    SequentialFeatureSelector,
)
from sklearn.impute import KNNImputer
from sklearn.linear_model import (
    LinearRegression,
    MultiTaskElasticNetCV,
    MultiTaskLassoCV,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def contextualize_measurements(
    measurement_table: pd.DataFrame,
    observable_table: pd.DataFrame,
    contextualization: str,
) -> pd.DataFrame:
    # Check requested contextualization is available
    if contextualization not in (
        "transcriptomics",
        "proteomics",
        "cytof_init",
        "cytof_dynamic",
        "cytof_dynamic_full",
    ):
        raise ValueError(f"Unknown contextualization: {contextualization}")

    # Make a copy of the measurements table
    input_measurements = measurement_table.copy()

    # Subset measurements to chosen contextualization
    # e.g. if transcriptomics, only keep measurements with measurementType == transcriptomics
    if contextualization == "transcriptomics":
        input_measurements = input_measurements[
            input_measurements["measurementType"] == "transcriptomics"
        ]
    elif contextualization == "proteomics":
        input_measurements = input_measurements[
            input_measurements["measurementType"] == "proteomics"
        ]
    elif contextualization.split("_")[0] == "cytof":
        input_measurements = input_measurements[
            input_measurements["measurementType"] == "cytof"
        ]
    # For transcriptomics, proteomics and cytof_init (initial) only keep time 0
    # In other words, only keep time-course info for cytof_dynamic
    if contextualization in ("transcriptomics", "proteomics", "cytof_init"):
        input_measurements = input_measurements[
            input_measurements[petab.TIME] == 0
        ]
    if contextualization.split("_")[0] == "cytof":
        # Split SIMULATION_CONDITION_ID and keep the stimulus info (0th is cell line, 1st is stimulus)
        input_measurements[petab.SIMULATION_CONDITION_ID] = input_measurements[
            petab.SIMULATION_CONDITION_ID
        ].apply(lambda x: x.split("__")[1])

        pivot_columns = (
            petab.OBSERVABLE_ID,
            petab.SIMULATION_CONDITION_ID,
            petab.TIME,
        )
        # For cytof_dynamic_full, keep all observables
        if contextualization == "cytof_dynamic":
            # For cytof_dynamic, subset observables to those within the model (ERK, MEK)
            input_measurements = input_measurements[
                input_measurements[petab.OBSERVABLE_ID].isin(
                    list(observable_table.index)
                )
            ]
        elif contextualization == "cytof_init":
            # For cytof_init, subset to EGF stimulation only
            input_measurements = input_measurements[
                input_measurements[petab.SIMULATION_CONDITION_ID].apply(
                    lambda x: x.endswith("EGF")
                )
            ]
            # and only keep observable ID as pivot columns (rather than observable, condition, time)
            pivot_columns = [petab.OBSERVABLE_ID]
    else:
        pivot_columns = [petab.OBSERVABLE_ID]

    # in all cases/contexts: average over replicates through np.nanmean aggfunc in pivot_table
    input_data = input_measurements.pivot_table(
        index=petab.PREEQUILIBRATION_CONDITION_ID,  # i.e. the cell line
        columns=pivot_columns,  # i.e. the observable/biomarkers in the case of cytof/proteomics and transcriptomics
        values=petab.MEASUREMENT,  # the actual measurement/signal
        aggfunc="mean",  # aggregate via NaN-compatible mean in case of replicates (e.g. triplicates for proteomics)
        # np.nanmean generates FutureWarning
    )

    return input_data


def load_data(
    contextualization,
    samples,
    features,
    measurement_table,
    observable_table,
):
    input_data = contextualize_measurements(
        measurement_table, observable_table, contextualization
    )

    # subset samples
    input_data = input_data.loc[samples, :]

    if contextualization == "cytof_dynamic":
        #  nn imputation
        for marker in ("pERK_Y204_obs", "pMEK_S222_obs", "pERBB2_Y1248_obs"):
            pairs = [
                ((marker, "EGF", 12.0), (marker, "EGF", 13.0)),
                ((marker, "EGF", 35.0), (marker, "EGF", 40.0)),
            ] + [
                ((marker, pert, time), (marker, pert, 17.0))
                for pert in (
                    "iMEK",
                    # "iPI3K",
                    "iEGFR",
                    # "iPKC"
                )
                for time in (14.0, 15.0, 16.0)
            ]
            for source, target in pairs:
                if source not in input_data.columns:
                    continue
                mask = input_data.loc[:, target].isna()
                input_data.loc[mask, target] = input_data.loc[mask, source]
        #  regression imputation
        for marker in (
            "pERK_Y204_obs",
            "pMEK_S222_obs",
            "pERBB2_Y1248_obs",
        ):  # all currently considered observables - might need to access, not hardcode
            for pert in (
                "EGF",
                "iMEK",
                # "iPI3K",
                "iEGFR",
                # "iPKC",
            ):  # all currently considered conditions - might need to access, not hardcode
                for missing_time, [time_before, time_after] in zip(
                    [7.0, 13.0, 40.0], [[0.0, 9.0], [9.0, 17.0], [17.0, 60.0]]
                ):
                    if (marker, pert, missing_time) not in input_data.columns:
                        continue
                  
                    mask = input_data.loc[
                        :, (marker, pert, missing_time)
                    ].isna()
                    input_data.loc[mask, (marker, pert, missing_time)] = (
                        input_data.loc[mask, (marker, pert, time_before)]
                        * (missing_time - time_before)
                        / (time_after - time_before)
                        + input_data.loc[mask, (marker, pert, time_after)]
                        * (time_after - missing_time)
                        / (time_after - time_before)
                    )

    if features:
        # for prediction, use feature set computed on training data
        input_data = input_data[features]
    else:
        # for training, compute feature set, filtering out too many nans
        # TODO @GiacomoFabrini: fix this - it needs to yield consistent numbers of columns!!!
        input_data = input_data.loc[
            :, input_data.isna().sum() / input_data.shape[0] < 0.3
        ]
        if contextualization == "transcriptomics":
            # look at mean vs variance plot
            # plt.scatter(np.nanmedian(input_data,axis=0),np.nanvar(input_data,axis=0))
            # plt.yscale('log')
            # plt.show()

            # filter low capture efficiency genes
            input_data = input_data.loc[
                :,
                np.nanmin(input_data, axis=0)
                < np.nanmedian(input_data, axis=0),
            ]
        elif contextualization == "proteomics":
            # look at mean vs variance plot
            # plt.scatter(np.nanmedian(input_data,axis=0),np.nanvar(input_data,axis=0))
            # plt.yscale('log')
            # plt.show()
            input_data = input_data.loc[
                :, np.nanmedian(input_data, axis=0) > -2.5
            ]
        features = list(input_data.columns)
    return input_data, features


def build_preprocesser(
    preprocess: str, input_data: np.ndarray, output_data: np.ndarray
):
    steps = [
        ("scaler", StandardScaler()),
        ("impute", KNNImputer()),
    ]
    if preprocess.startswith(("pca", "spca")):
        inputs = Pipeline(steps).fit_transform(input_data)
        var_expl = (
            PCA(n_components=input_data.shape[0])
            .fit(inputs)
            .explained_variance_ratio_
        )
        n_pca = np.nonzero(np.cumsum(var_expl) > 0.95)[0][0] + 1
        if preprocess.startswith("spca"):
            pipe = Pipeline(
                steps
                + [
                    ("spca", SparsePCA(n_components=n_pca)),
                    ("reg", LinearRegression()),
                ]
            )
            grid = GridSearchCV(
                pipe,
                param_grid={"spca__alpha": np.logspace(-3, 3, 7)},
                cv=5,
                scoring="neg_mean_squared_error",
            )
            grid.fit(input_data, output_data)
            steps.append(
                (
                    "pca",
                    SparsePCA(
                        n_components=n_pca,
                        alpha=grid.best_params_["spca__alpha"],
                    ),
                )
            )
        else:
            steps.append(("pca", PCA(n_components=n_pca)))
    elif preprocess == "rfe":
        steps.append(
            (
                "selector",
                RFECV(estimator=LinearRegression(), min_features_to_select=6),
            )
        )
    elif preprocess == "elastic":
        steps.append(
            (
                "selector",
                SelectFromModel(MultiTaskElasticNetCV(cv=5, n_alphas=20)),
            )
        )
    elif preprocess == "lasso":
        steps.append(
            ("selector", SelectFromModel(MultiTaskLassoCV(cv=5, n_alphas=20)))
        )
    elif preprocess == "sequential":
        steps.append(
            (
                "selector",
                SequentialFeatureSelector(
                    estimator=LinearRegression(),
                    scoring="neg_mean_squared_error",
                    cv=5,
                ),
            )
        )
    elif preprocess in ("pls", "cca"):
        model = {
            "pls": PLSRegression,
            "cca": CCA,
        }.get(preprocess)
        pipe = Pipeline(steps + [("selector", model())])
        grid = GridSearchCV(
            pipe,
            param_grid={"selector__n_components": np.arange(1, 20)},
            cv=5,
            scoring="neg_mean_squared_error",
        )
        grid.fit(input_data, output_data)
        steps.append(
            (
                "selector",
                model(
                    n_components=grid.best_params_["selector__n_components"]
                ),
            )
        )
    elif preprocess == "all":
        pass
    else:
        raise ValueError(f"Unknown preprocessing {preprocess}")

    return Pipeline(steps)
