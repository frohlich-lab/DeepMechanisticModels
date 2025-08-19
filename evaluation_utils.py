import itertools as itt
from typing import Any, Dict, List, Tuple, Union

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pandas as pd
import petab.v1 as petab
from amici.petab import rdatas_to_simulation_df
from jax import vmap
from scipy.sparse.csgraph import connected_components
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

from common import (
    MEASUREMENTS_FILE,
    OBSERVABLES_FILE,
    REGRESSION_MODES,
    TRAINED_MODEL,
    Wildcards,
    evaluations_dir,
    hardest_cell_lines,
    scan_attributes,
    training_samples,
    val_samples,
)
from cytof.problem import CytofProblem
from dmm.config_options import Conf
from dmm.dmm_autoencoder_eqx import DeepMechanisticModel
from dmm.initialisation import (
    get_features,
    impute_features,
)
from dmm.petab_subproblem import load_petab
from dmm.pretraining import (
    generate_average_pretraining_problem,
    generate_per_sample_pretraining_problems,
)
from dmm.training_helper_funcs import create_pypesto_problem
from evaluation_plotting import (
    random_forest_importance_plot,
)
from stat_test import statistical_significance_test
from training_configuration import HP_RUN_MODE, SPLITS


def get_measurements_and_obervables(conf: Conf):
    df_meas = pd.read_csv(
        MEASUREMENTS_FILE.format(**conf.to_dict()), sep="\t", index_col=0
    )
    df_obs = pd.read_csv(
        OBSERVABLES_FILE.format(**conf.to_dict()), sep="\t", index_col=0
    )
    df_meas = df_meas[
        df_meas[petab.OBSERVABLE_ID].apply(lambda x: x in df_obs.index)
    ]
    return df_meas, df_obs


def load_model_and_obj(
    conf: Conf,
    petab_base_files: Dict[str, pd.DataFrame],
    features: pd.DataFrame,
) -> tuple[DeepMechanisticModel, Any]:
    # Get cytof problem
    cytof_problem = CytofProblem(conf.model)

    # Define filepaths for serialized models
    trained_model_file = TRAINED_MODEL.format(**conf.to_dict())

    petab_importer = load_petab(
        problem=cytof_problem,
        dataset=conf.data,
        **petab_base_files,
        samples=list(features.index),
    )
    pypesto_subproblem = petab_importer.create_problem()

    model = DeepMechanisticModel.load(
        filename=trained_model_file,
        pypesto_problem=pypesto_subproblem,
    )

    pypesto_problem = create_pypesto_problem(pypesto_subproblem)

    return model, pypesto_problem


def process_per_sample_pretrain(
    sample: str,
    problem,
    conf: Conf,
    indir,
    petab_base_files: Dict[str, pd.DataFrame],
):
    rfile = indir / f"{sample}.csv"
    petab_base_importer = load_petab(
        problem,
        conf.data,
        **petab_base_files,
    )

    importer = generate_per_sample_pretraining_problems(
        petab_base_importer,
        problem,
        conf.data,
        sample,
    )

    problem_sample = importer.create_problem()
    df = pd.read_csv(rfile, index_col=[0])
    problem.apply_objective_settings(problem_sample.objective)

    ress = []
    fvals = []
    for ipar in range(len(df)):
        x = problem_sample.get_reduced_vector(
            df.values[ipar, :], problem_sample.x_free_indices
        )
        res = problem_sample.objective(x, return_dict=True)
        ress.append(res)
        fvals.append(res["fval"])

    # Convert the simulation to PEtab format.
    simulation_df = rdatas_to_simulation_df(
        ress[np.argmin(fvals)]["rdatas"],
        model=problem_sample.objective.amici_model,
        measurement_df=importer.petab_problem.measurement_df,
    )
    return importer, simulation_df


def simulate_avg_model(
    conf: Conf,
    indir,
    petab_base_files: Dict[str, pd.DataFrame],
    dataset: str,
) -> pd.DataFrame:
    problem = CytofProblem(conf.model)
    rfile = indir / f"model_average_{conf.samples}.csv"

    petab_base_importer = load_petab(
        problem,
        conf.data,
        **petab_base_files,
    )

    samples = (
        training_samples(Wildcards(conf.data, conf.samples))
        if dataset == "train"
        else val_samples(Wildcards(conf.data, conf.samples))
    )
    if not len(samples):
        return pd.DataFrame([])

    importer = generate_average_pretraining_problem(
        petab_base_importer,
        problem,
        conf.data,
        samples,
    )
    problem_sample = importer.create_problem()
    df = pd.read_csv(rfile, index_col=[0])
    problem.apply_objective_settings(problem_sample.objective)

    ress = []
    fvals = []
    for _ipar in range(len(df)):
        x = problem_sample.get_reduced_vector(
            df.values[0, :], problem_sample.x_free_indices
        )
        res = problem_sample.objective(x, return_dict=True)
        if np.isfinite(res["fval"]):
            ress.append(res)
            fvals.append(res["fval"])

    # Convert the simulation to PEtab format.
    if fvals:
        avg_model = rdatas_to_simulation_df(
            ress[np.argmin(fvals)]["rdatas"],
            model=problem_sample.objective.amici_model,
            measurement_df=importer.petab_problem.measurement_df,
        )
    else:
        avg_model = importer.petab_problem.measurement_df.copy().rename(
            columns={petab.MEASUREMENT: petab.SIMULATION}
        )
        avg_model[petab.SIMULATION] = np.nan

    return avg_model


def process_avg_model_simulation(
    avg_model: pd.DataFrame, df_meas: pd.DataFrame, dataset: str, samples: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    avg_model[petab.SIMULATION_CONDITION_ID] = df_meas[
        petab.SIMULATION_CONDITION_ID
    ]
    avg_model[petab.PREEQUILIBRATION_CONDITION_ID] = df_meas[
        petab.PREEQUILIBRATION_CONDITION_ID
    ]
    df_meas = df_meas[
        df_meas[petab.PREEQUILIBRATION_CONDITION_ID].isin(samples[dataset])
    ]
    avg_model = avg_model[
        avg_model[petab.PREEQUILIBRATION_CONDITION_ID].isin(samples[dataset])
    ]
    df_meas = df_meas[df_meas["measurementType"] == "cytof"]
    avg_model = avg_model[avg_model["measurementType"] == "cytof"]
    return avg_model, df_meas


def get_embedding_and_params_df(
    dmm_model: DeepMechanisticModel,
    input_features: np.ndarray | jnp.ndarray,
    context: str,
    split: str,
    dataset: str,
    job: int,
    samples: list[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not input_features.shape[0]:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    # Latent embeddings
    temp_latent_embeddings = vmap(
        eqx.nn.inference_mode(dmm_model).encode, in_axes=(0, None)
    )(input_features, jr.PRNGKey(0))
    latent_embeddings_df = pd.DataFrame(
        {
            "cell_line": samples,
            "L1": temp_latent_embeddings[:, 0],
            "L2": temp_latent_embeddings[:, 1],
        }
    ).assign(context=context, samples=split, dataset=dataset, job=job)

    # Parameter deviations
    param_deviations_df = pd.DataFrame(
        {
            "cell_line": samples,
            **dict(
                zip(
                    dmm_model.parameter_deviation_names,
                    eqx.nn.inference_mode(dmm_model)
                    .inflate_params(input_features, jr.PRNGKey(0))
                    .T,
                )
            ),
        }
    ).assign(context=context, samples=split, dataset=dataset, job=job)

    # Full parameters (deviations + medians)
    params_df = pd.DataFrame(
        {
            "cell_line": samples,
            **dict(
                zip(
                    dmm_model.parameter_deviation_names,
                    (
                        eqx.nn.inference_mode(dmm_model).inflate_params(
                            input_features, jr.PRNGKey(0)
                        )
                        + dmm_model.kin_params_combiner.learned_global_kin_params[
                            : len(dmm_model.parameter_deviation_names)
                        ]
                    ).T,
                )
            ),
        }
    ).assign(context=context, samples=split, dataset=dataset, job=job)
    return latent_embeddings_df, param_deviations_df, params_df


def pca_latent_embeddings(
    le_df: pd.DataFrame,
    hyperparam_configs: dict,
    scale: bool = True,
    center: bool = True,
    center_method: str = "auto",
    # col_grouping: str = "subtype_Luminal/Basal",
    # num_jobs_plot: int = 5,
):
    """
    :param le_df:
        latent embedding pd.DataFrame
    :param hyperparam_configs:
        dict mapping number of `samples` (CV splits) to list of hyperparameter configurations
    :param scale:
        bool flag indicating whether to (unit) scale the embeddings
    :param center:
        bool flag indicating whether to center the embeddings
    :param center_method:
        str, method to center the embeddings, either "mean" or "median"

    :return:
        pd.DataFrame containing per-configuration (optionally) centered & scaled PCA-processed latent embeddings
    """

    pca_dfs = []
    for samples, sub_df in le_df.groupby("samples"):
        # Remove "job" key from each dictionary
        unique_configs = {
            frozenset({k: v for k, v in d.items() if k != "job"}.items())
            for d in hyperparam_configs[samples]
        }

        # Convert back to list of dicts
        unique_configs = [dict(config) for config in unique_configs]
        for config in unique_configs:
            # Apply filtering using a vectorized mask
            mask = np.all(
                [
                    sub_df[key] == value
                    for key, value in config.items()
                    if key != "job"
                ],  # keep all multistarts
                axis=0,
            )
            filtered_df = sub_df[mask].copy()
            dataset_mapping = (
                filtered_df[["cell_line", "dataset"]]
                .iloc[: filtered_df.cell_line.nunique()]
                .set_index("cell_line")
                .to_dict()
            )

            # Check filtered_df is not empty - if empty, skip
            if filtered_df.empty:
                continue

            # Get latent embeddings
            les = filtered_df[["cell_line", "L1", "L2", "job"]].set_index(
                "cell_line"
            )
            les_pivot = les.set_index("job", append=True).unstack("job")
            les_pivot.columns = [
                f"{col[0]}_{col[1]}" for col in les_pivot.columns
            ]

            all_job_les = les_pivot.values

            # Center and scale if necessary
            if (
                center != "auto"
            ):  # auto: centering automatically performed by PCA
                all_job_les -= (
                    all_job_les.mean(axis=0)
                    if center_method == "mean"
                    else np.median(all_job_les, axis=0)
                )
            if scale:
                all_job_les = StandardScaler().fit_transform(all_job_les)
            # Get 2D PCA to try and remove potential rotations between multistart embeddings
            pca = PCA(n_components=2)
            les_pca = pca.fit_transform(all_job_les)
            # Append to growing list of processed DataFrames
            temp_df = pd.DataFrame(
                index=les_pivot.index, data=les_pca, columns=["L1", "L2"]
            ).assign(
                **{
                    key: value for key, value in config.items() if key != "job"
                },
                variance_explained=pca.explained_variance_ratio_.sum(),  # keep info on explained variance to compare across regularisation strengths
            )
            temp_df["dataset"] = temp_df.index.map(dataset_mapping["dataset"])
            pca_dfs.append(temp_df)

    # Concatenate all processed DataFrames
    return pd.concat(pca_dfs)


def cosine_similarity_embeddings(
    pca_le_df: pd.DataFrame,
    hyperparam_configs: dict,
) -> pd.DataFrame:
    # # Pairwise comparison within same configuration, different multistarts/jobs
    # job_results_dfs = []
    # # Compute cosine similarity between latent embeddings among all pairs of multistarts and average
    # for samples, sub_df in pca_le_df.groupby("samples"):
    #     # Obtain unique subconfigurations (excluding job info)
    #     config_df = pd.DataFrame(hyperparam_configs[samples]).drop(columns=["job"]).drop_duplicates()
    #     subconfigs = config_df.to_dict(orient="records")
    #
    #     for subconfig in subconfigs:
    #         # Apply filtering using a vectorized mask
    #         mask = np.all(
    #             [sub_df[key] == value for key, value in subconfig.items()],
    #             axis=0
    #         )
    #         # mask = (sub_df[list(subconfig)] == pd.Series(subconfig)).all(axis=1)
    #         filtered_df = sub_df[mask]
    #         if filtered_df.empty:  # skip if no rows match the subconfig
    #             continue
    #         # Get latent embeddings
    #         jobs = sorted(filtered_df.job.unique())
    #         les = [filtered_df[filtered_df.job == job].sort_values(by="cell_line")[["L1", "L2"]].values for job in jobs]
    #         val_les = [filtered_df[(filtered_df.job == job) & (filtered_df.dataset == "test")][["L1", "L2"]].values for
    #                    job in jobs]
    #         # Compute pairwise cosine similarities
    #         cos_sims = [cosine_similarity(les[i], les[j]).mean()
    #                     for i in range(len(les)) for j in range(i + 1, len(les))]
    #         val_cos_sims = [cosine_similarity(val_les[i], val_les[j]).mean()
    #                         for i in range(len(val_les)) for j in range(i + 1, len(val_les))]
    #         # Store results
    #         temp_df = pd.DataFrame([subconfig]).assign(
    #             mean_cos_sim=np.mean(cos_sims),
    #             max_cos_sim=np.max(cos_sims),
    #             min_cos_sim=np.min(cos_sims),
    #             mean_val_cos_sim=np.mean(val_cos_sims),
    #             max_val_cos_sim=np.max(val_cos_sims),
    #             min_val_cos_sim=np.min(val_cos_sims),
    #         )
    #         job_results_dfs.append(temp_df)

    # Comparison across CV splits (train/test) with same hyperparameters and jobs - only on validation cell-lines
    # Step 1: Subset to validation cell-lines
    val_pca_le_df = pca_le_df.copy()
    if "cell_line" not in val_pca_le_df.columns:
        val_pca_le_df.reset_index(inplace=True)
    val_pca_le_df = val_pca_le_df[
        val_pca_le_df.cell_line.isin(hardest_cell_lines)
    ]
    # Step 2: Find cell lines that appear in both train and test datasets
    # TODO @GiacomoFabrini replace with subsetting to hardest_cell_lines corresponding to investigated CV-splits -- easier!
    valid_cell_lines = (
        val_pca_le_df.groupby("cell_line")["dataset"]
        .apply(lambda x: set(x) == {"train", "test"})
        .loc[lambda x: x]  # Keep only True values
        .index
    )
    if valid_cell_lines.empty:
        return pd.DataFrame()
    # Step 3: Keep only rows where cell_line is in valid_cell_lines
    val_pca_le_df = val_pca_le_df[
        val_pca_le_df.cell_line.isin(valid_cell_lines)
    ]
    # Step 4: For each configuration and cell-line, compute cosine similarities between the CV split where
    # the cell-line is in validation and those where it is in training and average
    cosine_results = []

    # TODO if analysing top_n (top_10) different CV splits (and dataset) will not have consistent job numbering -- cannot order by jobs and rather have to compute all similarities
    # Group by configuration, job, and cell_line to process each group separately
    group_cols = [
        col
        for col in val_pca_le_df.columns
        if col
        not in ["L1", "L2", "samples", "dataset", "job", "variance_explained"]
    ]
    for (group_params), group in val_pca_le_df.groupby(group_cols):
        # Split into train and test subsets
        train_subset = group[group["dataset"] == "train"][["L1", "L2"]].values
        test_subset = group[group["dataset"] == "test"][["L1", "L2"]].values
        # Compute cosine similarity between train and test subsets
        similarity = cosine_similarity(train_subset, test_subset).mean()
        # Store the result
        cosine_results.append(
            {
                **dict(zip(group_cols, group_params)),
                "cosine_similarity": similarity,
            }
        )

    # Step 5: create dataframe where each configuration has associated mean + list of CV split cosine similarities
    cosine_df = pd.DataFrame(cosine_results).sort_values(
        by="cell_line"
    )  # ensure consistent ordering of CV splits
    # Group by config and job to compute the mean cosine similarity across cell lines
    cosine_summary_df = (
        cosine_df.groupby([col for col in group_cols if col != "cell_line"])[
            "cosine_similarity"
        ]
        .agg(
            ["mean", list]
        )  # Compute mean and keep list of all cosine similarities
        .reset_index()
        .rename(
            columns={
                "mean": "mean_cosine_similarity",
                "list": "cv_split_cosine_similarities",
            }
        )
        # .sort_values(by=["job"])  # ensures consistent ordering of jobs prior to operation below, not needed for top10
    )
    # REMOVED LAST STEP AS IT IS NOT NECESSARY WHEN ANALYSING TOP 10 JOBS (UNPAIRED)
    # # Step 6: average across the multistarts for each configuration (keep mean list of CV split-wise mean cosine similarities)
    # # Group by configuration only and average across jobs (multistarts)
    # cosine_summary_df = (
    #     cosine_summary_df.groupby([col for col in group_cols if col not in ["cell_line", "job"]])
    #     .agg({
    #         'mean_cosine_similarity': 'mean',
    #         'cv_split_cosine_similarities': lambda lists: np.mean(lists.tolist(), axis=0)
    #     })
    #     .reset_index()
    # )
    # Concatenate all processed DataFrames
    # return pd.concat(job_results_dfs, ignore_index=True), cosine_summary_df
    return cosine_summary_df


def silhouette_embeddings(
    pca_le_df: pd.DataFrame,
    hyperparam_configs: dict,
) -> Union[pd.DataFrame, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    # Obtain unique subconfigurations (excluding job & CV-split info)
    config_dfs = []
    for samples in hyperparam_configs.keys():
        config_dfs.append(
            pd.DataFrame(hyperparam_configs[samples])
            .drop(columns=["job", "samples"])
            .drop_duplicates()
        )
    config_df = pd.concat(config_dfs, ignore_index=True).drop_duplicates()
    subconfigs = config_df.to_dict(orient="records")

    # DISABLED WITH TOP 10 JOBS
    # # Job-wise, all cell-lines
    # all_lines_results = []
    # for samples in hyperparam_configs.keys():
    #     sub_df = pca_le_df[pca_le_df.samples == samples]
    #     for subconfig in subconfigs:
    #         # Apply filtering using a vectorized mask
    #         mask = np.all(
    #             [sub_df[key] == value for key, value in subconfig.items()],
    #             axis=0
    #         )
    #         embeddings = sub_df[mask][["L1", "L2"]].values
    #         labels = sub_df[mask].job.values
    #         # Compute silhouette score, mean and standard deviation and store results
    #         all_lines_results.append(
    #             pd.DataFrame([subconfig]).assign(
    #                 samples=samples,
    #                 mean_silhouette_score=silhouette_score(embeddings, labels),
    #                 stddev_silhouette_score=np.std(silhouette_samples(embeddings, labels)),
    #                 all_scores=[silhouette_samples(embeddings, labels)],
    #             )
    #         )

    val_cell_lines = pca_le_df[pca_le_df.dataset == "test"].cell_line.unique()
    # # Job-wise, validation only: subset w.r.t. validation cell-line & config.; label with job ID; compute silhouette score
    # job_results_dfs = []
    # CV-split/Dataset-wise, validation only: subset w.r.t. validation cell-line & config.; label with dataset; compute silhouette score
    cv_results = []
    for cell_line in val_cell_lines:
        sub_df = pca_le_df[pca_le_df.cell_line == cell_line]
        for subconfig in subconfigs:
            # Apply filtering using a vectorized mask
            mask = np.all(
                [sub_df[key] == value for key, value in subconfig.items()],
                axis=0,
            )
            if sub_df[mask].empty:
                continue  # skip missing subconfigs
            embeddings = sub_df[mask][["L1", "L2"]].values
            # job_labels = sub_df[mask].job.values
            dataset_labels = sub_df[mask].dataset.values
            # # Compute silhouette score, mean and standard deviation and store results
            # job_results_dfs.append(
            #     pd.DataFrame([subconfig]).assign(
            #         cell_line=cell_line,
            #         mean_silhouette_score=silhouette_score(embeddings, job_labels),
            #         stddev_silhouette_score=np.std(silhouette_samples(embeddings, job_labels)),
            #         all_scores=[silhouette_samples(embeddings, job_labels)]
            #     )
            # )
            cv_results.append(
                pd.DataFrame([subconfig]).assign(
                    cell_line=cell_line,
                    mean_silhouette_score=silhouette_score(
                        embeddings, dataset_labels
                    ),
                    stddev_silhouette_score=np.std(
                        silhouette_samples(embeddings, dataset_labels)
                    ),
                    all_scores=[
                        silhouette_samples(embeddings, dataset_labels)
                    ],
                )
            )
    return (
        # pd.concat(all_lines_results, ignore_index=True),
        # pd.concat(job_results_dfs, ignore_index=True),
        pd.concat(cv_results, ignore_index=True)
    )


def connectivity_score(
    pca_le_df: pd.DataFrame,
    hyperparam_configs: dict,
) -> Union[pd.DataFrame, tuple[pd.DataFrame, pd.DataFrame]]:
    # Obtain unique subconfigurations (excluding job & CV-split info)
    config_dfs = [
        pd.DataFrame(hyperparam_configs[samples])
        .drop(columns=["job", "samples"])
        .drop_duplicates()
        for samples in hyperparam_configs.keys()
    ]
    subconfigs = (
        pd.concat(config_dfs, ignore_index=True)
        .drop_duplicates()
        .to_dict(orient="records")
    )

    # job_results = []
    cv_results = []
    for subconfig in subconfigs:
        # Apply filtering using a vectorized mask
        mask = np.all(
            [pca_le_df[key] == value for key, value in subconfig.items()],
            axis=0,
        )
        for cell_line in pca_le_df[mask][
            pca_le_df[mask].dataset == "test"
        ].cell_line.unique():
            sub_df = pca_le_df[mask & (pca_le_df.cell_line == cell_line)]
            for attribute, num_neighbours, results in zip(
                [
                    # "job",
                    "samples"
                ],
                [
                    # sub_df.samples.nunique(),
                    10,  # number of top multistarts per configuration
                ],
                [
                    # job_results,
                    cv_results
                ],
            ):
                # Build KNN graph on whole configuration set based on embeddings (L1, L2)
                knn_graph = kneighbors_graph(
                    sub_df[["L1", "L2"]],
                    n_neighbors=num_neighbours,  # having trouble finding a good value for this! Intuitively, I would choose num_samples for job-wise, and num_jobs for split-wise
                    mode="connectivity",
                )
                for attribute_value in sub_df[attribute].unique():
                    # Extract job/CV-split specific subgraph
                    attribute_mask = sub_df[attribute] == attribute_value
                    knn_subgraph = knn_graph[attribute_mask][:, attribute_mask]
                    # Get largest connected component size
                    n_components, labels = connected_components(
                        knn_subgraph, directed=False
                    )
                    largest_component_size = np.max(np.bincount(labels))
                    # Compute connectivity score
                    connectivity_score = largest_component_size / np.sum(
                        attribute_mask
                    )
                    results.append(
                        pd.DataFrame([subconfig]).assign(
                            **{
                                "cell_line": cell_line,
                                f"{attribute}": attribute_value,
                                "connectivity_score": connectivity_score,
                            }
                        )
                    )

    return (
        # pd.concat(job_results, ignore_index=True),
        pd.concat(cv_results, ignore_index=True)
    )


def train_rf_features_to_rmse(
    dmm_results: pd.DataFrame, conf, num_top_features: int = 10
):
    # Drop Infs & NaNs (incompatible with RandomForestRegressor)
    dmm_results = dmm_results.replace([np.inf, -np.inf], np.nan).dropna()
    # Get targets
    reg_targets_train = dmm_results.pop("rmse_train")
    reg_targets_val = dmm_results.pop("rmse_test")

    # Replace categorical variables with dummy one-hot-encodings
    reg_features = pd.get_dummies(dmm_results)

    # Fit regressors
    rfr_train, rfr_val = RandomForestRegressor(), RandomForestRegressor()
    rfr_train.fit(X=reg_features, y=reg_targets_train)
    rfr_val.fit(X=reg_features, y=reg_targets_val)

    # Store and plot feature importances
    results_dfs = {
        dataset: pd.DataFrame(
            {
                "importances": regressor.feature_importances_[
                    regressor.feature_importances_.argsort()
                ][::-1][:num_top_features],
                "features": reg_features.columns[
                    regressor.feature_importances_.argsort()
                ][::-1][:num_top_features],
            }
        )
        for dataset, regressor in zip(["train", "val"], [rfr_train, rfr_val])
    }
    random_forest_importance_plot(results_dfs, conf, "show")


def convert_dataframe_dtypes(df: pd.DataFrame):
    cols = [
        "n_hidden",
        "depth",
        "nn_structure_multiplier",
        "inflater_output_reg_epoch",
        "opt_steps",
        "opt_mult",
        "job",
    ]
    for col in cols:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], downcast="integer")
    additional_cols = [
        "l1reg_encode",
        "oreg_encode",  # encoder
        "l1reg_inflate",
        "oreg_inflate",
        "l1reg_inflater_output",  # inflater
        "recon_loss",
        "symm_reg",  # decoder / reconstruction
        "median_reg",  # kinetic params median regularisation
        "opt_steps",
        "opt_mult",
        "momentum",  # parameters that can be pruned by generate_run_configs
    ]
    for col in additional_cols:
        if col not in df.columns:
            continue
        if (len(df[col].unique()) == 1) and (df[col].unique()[0] == 0):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        else:
            df[col] = df[col].astype("float")
    for col in [
        "pretrain",
        "use_layer_bias",
        "last_layer_activation",
    ]:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str)
    for col in [
        "reconstruct",
        "use_simple_linear_schedule",
        "use_early_stopping",
    ]:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(bool)
    return df


def get_best_regressor(
    dataframe: pd.DataFrame,
    group_attributes: List,
    target_attribute="rmse",
) -> pd.DataFrame:
    """
    Returns a pd.DataFrame with the best performing regressor on each context (cytof_init, proteomics,
    transcriptomics) / dataset (train/val) pair.
    """
    min_rmse = (
        dataframe.reset_index()
        .groupby(group_attributes)[target_attribute]
        .min()
        .reset_index()
    )
    best_regressor_df = dataframe.merge(
        min_rmse[["context", "dataset", target_attribute]],
        on=["context", "dataset", target_attribute],
    )
    return best_regressor_df


def aggregate_and_log(
    df: pd.DataFrame,
    conf: Conf,
    top_reg_param: str,
    return_stat_tests: bool,
    num_best: int = 10,
):
    outdir = evaluations_dir / conf.model / conf.data
    # Define aggregation groups for DMM and refs
    gbs_dmm = ["dataset", "ref"] + scan_attributes

    df["res"] = df["res"].astype(float)

    temp_dfs = []
    for ref_subset, group_cols in {
        "DMM": gbs_dmm,
        "BY_CL_COND_OBS": [
            "sample",
            "condition",
            "observable",
            "dataset",
            "ref",
        ]
        + scan_attributes,
        "refs": ["dataset", "context", "features", "samples", "ref"],
    }.items():
        if ref_subset != "BY_CL_COND_OBS":
            if ref_subset == "DMM":
                subset_df = df[df.ref == "DMM"]
            else:
                subset_df = df[~df.ref.isin(["DMM"])]
            temp_dfs.append(
                pd.DataFrame(
                    [
                        dict(
                            zip(group_cols, group),
                            rmse=np.sqrt(np.square(group_df["res"]).mean()),
                        )
                        for group, group_df in subset_df.groupby(group_cols)
                    ]
                )
            )
        else:
            by_cl_cond_obs = pd.DataFrame(
                [
                    dict(
                        zip(group_cols, group),
                        rmse=np.sqrt(np.square(group_df["res"]).mean()),
                    )
                    for group, group_df in df.groupby(group_cols, dropna=False)
                ]
            )

    data = pd.concat(temp_dfs).sort_values(by="ref")
    # print("Overall evaluation DataFrame is now ready.")
    # cleanup
    del temp_dfs

    # Statistical tests -- currently disabled
    # TODO @GiacomoFabrini -- review statistical tests? What do we want to do with them?
    if return_stat_tests:
        # Prepare statistical test dataframe
        # Create pivot table for statistical testing
        cols = [
            "dataset",
            "context",
            "features",
            "ref",
            "n_hidden",
            "orth_reg_strategy",
            "l1reg_inflate",
            "oreg_inflate",
            "l1reg_encode",
            "oreg_encode",
        ]
        # pivot table and create one column per cross-validation split and multistart/job
        pivot_data = data.pivot_table(
            index=cols, columns=["samples", "job"], values="rmse"
        )
        pivot_data = pivot_data.reset_index()
        # Create list of the MultiIndex RMSE columns created above
        JOBS = tuple(range(conf.n_starts))
        multiindex_rmse_cols = [
            (sample, job) for sample in SPLITS for job in JOBS
        ]
        # Create a single column 'rmse_list' listing all values from each of the MultiIndex columns
        pivot_data["rmse_list"] = pivot_data.apply(
            lambda row: np.array([row[col] for col in multiindex_rmse_cols]),
            axis=1,
        )
        # Add the newly created column to the list of columns to be kept (cols)
        cols += ["rmse_list"]
        # Subset the pivot table and reduce MultiIndex back to single-level index
        data_stat_tests = pivot_data[cols]
        data_stat_tests.columns = data_stat_tests.columns.droplevel(level=1)
        print("DataFrame for statistical testing is now ready.")

        stat_test_res_df = statistical_significance_test(data_stat_tests)

    # Get the best regressor for each context/dataset pair
    best_regressors = get_best_regressor(
        dataframe=data[data.ref.isin(REGRESSION_MODES)],
        group_attributes=["dataset", "context"],
        target_attribute="rmse",
    )
    print("Computed best regressor for each context/dataset pair.")

    # Combine train/val RMSE results for regressors and baselines
    nodmm_results = {
        dataset: (
            data[(data.dataset == dataset) & (~data.ref.isin(["DMM"]))].drop(
                columns=["dataset"]
            )
        )
        for dataset in ["train", "test"]
    }
    unified_nodmm_results = pd.merge(
        nodmm_results["train"],
        nodmm_results["test"],
        how="inner",
        on=list(data.columns.difference(["rmse", "dataset"])),
        suffixes=("_train", "_test"),
    )
    unified_baselines = unified_nodmm_results[
        unified_nodmm_results.ref.isin(["avg_model", "sample"])
    ]
    unified_baselines["model"] = unified_baselines["ref"]
    unified_regressors = unified_nodmm_results[
        unified_nodmm_results.ref.isin(REGRESSION_MODES)
    ]
    unified_best_regressors = unified_regressors.merge(
        best_regressors[best_regressors.dataset == "train"][
            ["context", "ref"]
        ],  # keep regressors that perform best on training set (same as DMM)
        how="inner",
        on=["context", "ref"],
    )
    unified_best_regressors["model"] = (
        unified_best_regressors["ref"] + unified_best_regressors["context"]
    )
    unified_nodmm_results = pd.concat(
        [unified_baselines, unified_best_regressors]
    )

    # Combine train/val RMSE results for DMMs
    dmm_results = {
        dataset: (
            data[(data.dataset == dataset) & (data.ref == "DMM")].drop(
                columns=["ref", "dataset"]
            )
        )
        for dataset in ["train", "test"]
    }
    unified_dmm_results = pd.merge(
        dmm_results["train"],
        dmm_results["test"],
        how="inner",
        on=list(data.columns.difference(["rmse", "dataset", "ref"])),
        suffixes=("_train", "_test"),
    )
    unified_dmm_results.to_csv(outdir / "unified_dmm_rmse_train_test.csv")
    print("Finished saving unified (train/test) DMM RMSE results.")

    # Restrict DMM results to top 10 jobs for each configuration according to training performance
    config_cols = [
        attribute
        for attribute in gbs_dmm
        if attribute not in ["ref", "job", "dataset"]
    ]
    top_n_dmm_train = (
        unified_dmm_results.groupby(config_cols)
        .apply(lambda x: x.nsmallest(num_best, "rmse_train"))
        .reset_index(drop=True)
    )
    # Ensure same dtypes as original dataframe
    top_n_dmm_train = convert_dataframe_dtypes(top_n_dmm_train)
    # Plot and store
    # plot_rmse_val_cell_lines(top_n_results_by_cl, conf, top_reg_param)
    by_cl_cond_obs.to_csv(outdir / "by_cl_cond_obs.csv")
    # Average over jobs, then over CV splits and get top (1) configuration per context based on validation performance
    top_n_dmm_train_cv = (
        top_n_dmm_train.groupby(config_cols)
        .agg(rmse_test_agg=("rmse_test", "mean"))
        .reset_index()
    )

    for cvsplit_label in ["MOSAsplits", "allsplits"]:
        if cvsplit_label == "MOSAsplits":
            sub_df = top_n_dmm_train_cv[
                top_n_dmm_train_cv.samples.isin([f"{i}of5" for i in range(4)])
            ]
        else:
            sub_df = top_n_dmm_train_cv
        best_configs_dmm = (
            sub_df.groupby([col for col in config_cols if col != "samples"])
            .agg(
                mean_rmse_test=("rmse_test_agg", "mean"),
                rmse_test_list=("rmse_test_agg", lambda x: list(x)),
            )
            .reset_index()
            .sort_values(by="mean_rmse_test", ascending=True)
            .groupby("context")
            .head(1)
        )
        best_configs_dmm.to_csv(outdir / f"top1_best_dmm_{HP_RUN_MODE}.csv")

        # Get the top 10 jobs corresponding to best validation config
        best_configs_dmm_jobs = (
            top_n_dmm_train.merge(
                best_configs_dmm,
                on=[
                    col
                    for col in best_configs_dmm.columns
                    if col not in ["mean_rmse_test", "rmse_test_list"]
                ],
            )
            .drop(columns=["mean_rmse_test", "rmse_test_list"])
            .assign(ref="DMM")
        )

        best_configs_dmm_jobs["model"] = (
            best_configs_dmm_jobs["ref"] + best_configs_dmm_jobs["context"]
        )
        if cvsplit_label == "MOSAsplits":
            best_configs_dmm_jobs = best_configs_dmm_jobs[
                best_configs_dmm_jobs.samples.isin(
                    [f"{i}of5" for i in range(4)]
                )
            ]
        barplot_df = pd.concat([unified_nodmm_results, best_configs_dmm_jobs])
        barplot_df.to_csv(
            outdir / f"top_{num_best}_best_dmm_with_refs.{cvsplit_label}.csv"
        )

    evaluation_dfs = [
        data,
    ]
    evaluation_tags = [
        "evaluate_all",
    ]
    if return_stat_tests:
        evaluation_dfs.append(stat_test_res_df)
        evaluation_tags.append("stat_tests_all")
    for evaluation_df, evaluation_tag in zip(evaluation_dfs, evaluation_tags):
        # Save dataframes to CSV
        evaluation_df.to_csv(outdir / f"{evaluation_tag}.csv")

    if return_stat_tests:
        return (
            data,
            stat_test_res_df,
            top_n_dmm_train,
            best_configs_dmm_jobs,
            best_regressors,
            unified_dmm_results,
        )
    else:
        return (
            data,
            top_n_dmm_train,
            best_configs_dmm_jobs,
            best_regressors,
            unified_dmm_results,
        )


def compute_deviation_ratio(
    param_dev_df: pd.DataFrame, param_cols: list, reg_param: str
):
    val_param_dev = param_dev_df[
        param_dev_df.cell_line.isin(hardest_cell_lines)
    ]

    def compute_stats(group):
        return pd.Series(
            {
                "mean": group[param_cols].mean().mean(),
                "median": group[param_cols].median().median(),
                "min": group[param_cols].min().min(),
                "max": group[param_cols].max().max(),
            }
        )

    stats_df = (
        val_param_dev.groupby(["cell_line", reg_param, "samples"])
        .apply(compute_stats)
        .reset_index()
    )
    stats_df["range"] = stats_df["max"] - stats_df["min"]

    results_dfs = []
    for samples, reg_param_val in itt.product(
        stats_df.samples.unique(), stats_df[reg_param].unique()
    ):
        val_cell_line = hardest_cell_lines[int(samples.split("of")[0])]
        subset_df = stats_df[
            (stats_df.samples == samples)
            & (stats_df[reg_param] == reg_param_val)
        ]
        val_cell = subset_df[subset_df.cell_line == val_cell_line]
        train_cells = subset_df[subset_df.cell_line != val_cell_line]
        average_range = train_cells["range"].mean()
        results_dfs.append(
            pd.DataFrame(
                {
                    "cell_line": [val_cell["cell_line"].values[0]],
                    "samples": [samples],
                    reg_param: [reg_param_val],
                    "deviation_ratio": [
                        val_cell["range"].values[0] / average_range
                    ],
                }
            )
        )

    return pd.concat(results_dfs)


def add_annotations(
    df: pd.DataFrame,
    brca_annot_df: pd.DataFrame,
    subtypes_pam50: dict,
    subtypes_lb: dict,
    subtypes_hr: dict,
    subtypes_her2: dict,
) -> pd.DataFrame:
    annotated_df = df.copy()
    annotated_df = annotated_df.merge(
        brca_annot_df.reset_index()[
            ["cell_line", "Site", "MS_Status", "Disease"]
        ],
        on="cell_line",
    )
    for col_name, subtypes_dict in zip(
        ["PAM50", "LB", "HR_Status", "HER2_Status"],
        [subtypes_pam50, subtypes_lb, subtypes_hr, subtypes_her2],
    ):
        annotated_df[col_name] = annotated_df.cell_line.map(subtypes_dict)
    return annotated_df


# TODO do we even need this function? Can use process_and_transform_features + extra step?
def load_and_transform_features(
    conf: Conf,
    dataset: str,
    features_filepath,
) -> np.ndarray:
    features = get_features(
        features_filepath=features_filepath, datasets=["train", "val"]
    )
    features = impute_features(features)
    return features[dataset].values
