import numpy as np
import pandas as pd
import petab
from common import features_dir
from dmm.initialisation import get_features, impute_features
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
from typing import List, Optional


def contextualize_measurements(
    measurement_table: pd.DataFrame,
    observable_table: pd.DataFrame,
    contextualization: str,
    samples: List[str],
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
    # For transcriptomics and proteomics, only keep time 0
    if contextualization in ("transcriptomics", "proteomics"):
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
            # For cytof_dynamic, subset observables to those within the model (ERK, MEK, ERBB2)
            input_measurements = input_measurements[
                input_measurements[petab.OBSERVABLE_ID].isin(
                    list(observable_table.index)
                )
            ]
        elif contextualization == "cytof_init":
            # For cytof_init, impute based on harmonised cytof_dynamic, then subset to EGF and time 0 only
            # TODO - do we need to impute only given the observables we use? In cytof_init we use all of them...
            # TODO - do we want to exclude info from inhibitors? (can subset to EGF earlier)
            # input_measurements = input_measurements[
            #     input_measurements[petab.SIMULATION_CONDITION_ID].apply(
            #         lambda x: x.endswith("EGF")
            #     )
            # ]
            harmonised_cytof_dynamic = harmonise_cytof_dynamic(
                input_measurements.pivot_table(
                    index=petab.PREEQUILIBRATION_CONDITION_ID,  # i.e. the cell line
                    columns=pivot_columns,
                    # i.e. the observable/biomarkers in the case of cytof/proteomics and transcriptomics
                    values=petab.MEASUREMENT,  # the actual measurement/signal
                    aggfunc="mean",
                )
            )
            # Filter out too many nans, including misaligned timepoints that have just been aligned
            harmonised_cytof_dynamic = harmonised_cytof_dynamic.loc[
                :, harmonised_cytof_dynamic.isna().sum() / harmonised_cytof_dynamic.shape[0] < 0.3
            ]
            # Fit imputer on training samples, transform train + val samples
            if len(samples) > 1:
                samples_train = samples
            else:
                samples_train = [sample for sample in harmonised_cytof_dynamic.index if sample not in samples]
            imputer = KNNImputer()
            imputer.fit(harmonised_cytof_dynamic.loc[samples_train, :].values)
            input_data = pd.DataFrame(
                imputer.transform(harmonised_cytof_dynamic.values),
                columns=harmonised_cytof_dynamic.columns,
                index=harmonised_cytof_dynamic.index,
            )
            # subset to EGF and time 0
            input_data = input_data[[col for col in input_data.columns if (col[1] == "EGF") and (col[2] == 0.0)]]
            # remove info on condition and timepoint, leaving markers only as column names
            input_data.columns = [col[0] for col in input_data.columns]
            return input_data
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


def preprocess_mosa_latent(conf, samples_train, samples_val):
    annot = pd.read_csv(features_dir / conf.model / conf.data / "DepMap_model_list_20241120.csv", index_col=0)
    mosa_latent = pd.read_csv(features_dir / conf.model / conf.data / "MOSA_20231023_092657_latent_joint.csv", index_col=0)
    # Map from Sanger model ID to cell-line name + process into our cell-line name format (no hyphens, "c" prefix)
    mosa_latent["cell_line"] = mosa_latent.index.map(annot["model_name"].to_dict())
    mosa_latent["cell_line"] = mosa_latent["cell_line"].str.replace("-", "")
    mosa_latent["cell_line"] = "c" + mosa_latent["cell_line"]

    # Subset to breast cancer cell lines in our data
    our_lines = samples_train + samples_val
    mosa_latent = mosa_latent[mosa_latent["cell_line"].isin(our_lines)]

    # Find available samples in pretrained MOSA latent embeddings
    available_samples_train = [sample for sample in samples_train if sample in mosa_latent["cell_line"].unique()]
    available_samples_val = [sample for sample in samples_val if sample in mosa_latent["cell_line"].unique()]
    mosa_latent = mosa_latent.set_index("cell_line")
    input_train = mosa_latent.loc[available_samples_train, :]
    features_train = list(input_train.columns)
    input_val = mosa_latent.loc[available_samples_val, :]
    return input_train, input_val, features_train


def harmonise_cytof_dynamic(input_data):
    #  nearest neighbour imputation for misaligned/heterogeneous timepoints
    def impute_missing(input_data, source, target):
        if source not in input_data.columns:
            return input_data
        mask = input_data[target].isna()
        input_data.loc[mask, target] = input_data.loc[mask, source]
        return input_data

    markers = ("pERK_Y204_obs", "pMEK_S222_obs", "pERBB2_Y1248_obs")
    egf_and_perturbations = ("EGF", "iMEK", "iEGFR")

    for marker in markers:
        for pert in egf_and_perturbations:
            # Impute 13.0 from 12.0 then 14.0 (order matters!)
            for source_time in (12.0, 14.0):
                input_data = impute_missing(
                    input_data,
                    source=(marker, pert, source_time),
                    target=(marker, pert, 13.0),
                )

            # Impute 17.0 from 15.0 then 16.0 (only for inhibitors)
            if pert != "EGF":
                for source_time in (15.0, 16.0):
                    input_data = impute_missing(
                        input_data,
                        source=(marker, pert, source_time),
                        target=(marker, pert, 17.0),
                    )

            # Impute 40.0 -> 60.0
            input_data = impute_missing(
                input_data,
                source=(marker, pert, 40.0),
                target=(marker, pert, 60.0),
            )

        # Impute 35.0 -> 40.0 (only for EGF)
        input_data = impute_missing(
            input_data,
            source=(marker, "EGF", 35.0),
            target=(marker, "EGF", 40.0),
        )

    #  linear interpolation of intermediate missing timepoints
    for marker in ("pERK_Y204_obs", "pMEK_S222_obs", "pERBB2_Y1248_obs"):  # all currently considered observables - might need to access, not hardcode
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
    return input_data


def load_data(
    contextualization,
    samples,
    features,
    measurement_table,
    observable_table,
    features_filepath = None,
):
    if contextualization != "MOSA":
        input_data = contextualize_measurements(
            measurement_table, observable_table, contextualization, samples
        )
    elif (contextualization == "MOSA") and (features_filepath is not None):
        input_data = get_features(features_filepath=features_filepath, datasets=["train", "val"])
        input_data = impute_features(input_data)
        input_data = pd.concat([input_data["train"], input_data["val"]])
        # ensure index has correct name
        input_data = input_data.rename_axis(petab.PREEQUILIBRATION_CONDITION_ID)
    else:
        raise ValueError(
            f"Received invalid combination: context {contextualization} and features_filepath {features_filepath}"
        )

    # subset samples to dataset at hand (train, val)
    if contextualization != "MOSA":
        input_data = input_data.loc[samples, :]
    else: # handle missing cell-lines for pre-trained MOSA
        input_data = input_data.loc[[sample for sample in samples if sample in input_data.index], :]

    if contextualization == "cytof_dynamic":
        input_data = harmonise_cytof_dynamic(input_data)
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


def build_preprocessor(
    preprocess: str, input_data: np.ndarray, output_data: np.ndarray, cv = None,
):
    steps = [
        ("scaler", StandardScaler()),
        ("impute", KNNImputer()),
    ]

    def get_cv():
        if cv is None:
            return 5
        return cv

    cv = get_cv()

    if preprocess.startswith(("pca", "spca")):
        # Need to keep this computation of n_pca for SparsePCA (does not accept float as percentage of variance)
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
                cv=cv,
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
                RFECV(estimator=LinearRegression(), min_features_to_select=10),
            )
        )
    elif preprocess == "elastic":
        steps.append(
            (
                "selector",
                SelectFromModel(
                    MultiTaskElasticNetCV(cv=cv, n_alphas=20, max_iter=10000),  # increased max_iter 10x
                    max_features=min(100, input_data.shape[1])
                ),  # ensures recommended max_features is not larger than the number of features (e.g. cytof_init)
            )
        )
    elif preprocess == "lasso":
        steps.append(
            (
                "selector",
                SelectFromModel(
                    MultiTaskLassoCV(cv=cv, n_alphas=20, max_iter=10000),
                    max_features=min(100, input_data.shape[1])
                ),
             )
        )
    elif preprocess == "sequential":
        steps.append(
            (
                "selector",
                SequentialFeatureSelector(
                    estimator=LinearRegression(),
                    scoring="neg_mean_squared_error",
                    cv=cv,
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
            cv=cv,
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
