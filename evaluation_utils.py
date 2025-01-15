import jax.numpy as jnp
import numpy as np
import pandas as pd
import petab

from amici.petab_objective import rdatas_to_simulation_df
from common import (
    hardest_cell_lines,
    MEASUREMENTS_FILE,
    OBSERVABLES_FILE,
    TRAINED_BEST_MODELS,
    Wildcards,
    test_samples,
    training_samples,
)
from cytof.problem import CytofProblem
from dmm.config_options import Conf
from dmm.dmm_autoencoder_eqx import DeepMechanisticModel
from dmm.petab_subproblem import load_petab
from dmm.pretraining import generate_average_pretraining_problem, generate_per_sample_pretraining_problems
from dmm.training_helper_funcs import create_pypesto_problem
from jax import vmap
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from training_configuration import N_ENSEMBLE_MEMBERS
from typing import Dict, Tuple, Any, Union


def get_measurements_and_obervables(conf: Conf):
    df_meas = pd.read_csv(
        MEASUREMENTS_FILE.format(**conf.__dict__), sep="\t", index_col=0
    )
    df_obs = pd.read_csv(
        OBSERVABLES_FILE.format(**conf.__dict__), sep="\t", index_col=0
    )
    df_meas = df_meas[
        df_meas[petab.v1.OBSERVABLE_ID].apply(lambda x: x in df_obs.index)
    ]
    return df_meas, df_obs


def load_model_and_obj(
        conf: Conf, petab_base_files: Dict[str, pd.DataFrame], dataset: str, num_ensemble_members: int
) -> tuple[list[DeepMechanisticModel], Any]:
    # Get cytof problem
    cytof_problem = CytofProblem(conf.model)

    # Define filepaths for serialized models -- need to be formatted for ensemble_id
    trained_model_file = TRAINED_BEST_MODELS.format(
        **{**conf.__dict__, **dict(ensemble_id="{ensemble_id}")}
    )

    models = []
    for ensemble_id in range(
            min(num_ensemble_members, N_ENSEMBLE_MEMBERS)
    ):
        ensemble_member_file = Path(trained_model_file.format(ensemble_id=ensemble_id))

        # Load ensemble member model
        model = DeepMechanisticModel.load(
            filename=ensemble_member_file,
            problem=cytof_problem,
            dataset=dataset,
            petab_base_files=petab_base_files,
        )
        models.append(model)

    # Create pypesto problem from any of the loaded models to extract objective
    pypesto_problem = create_pypesto_problem(models[0])
    obj = pypesto_problem.objective.base_objective.base_objective
    return models, obj


def process_per_sample_pretrain(
        sample: str,
        problem,
        conf: Conf,
        indir,
        petab_base_files: Dict[str, pd.DataFrame]
):
    rfile = indir / f"{sample}.csv"
    if not rfile.exists():
        return None

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

    importer = generate_average_pretraining_problem(
        petab_base_importer,
        problem,
        conf.data,
        training_samples(Wildcards(conf.data, conf.samples))
        if dataset == "train"
        else test_samples(Wildcards(conf.data, conf.samples)),
    )
    problem_sample = importer.create_problem()
    df = pd.read_csv(rfile, index_col=[0])
    problem.apply_objective_settings(problem_sample.objective)

    ress = []
    fvals = []
    for ipar in range(len(df)):
        x = problem_sample.get_reduced_vector(
            df.values[0, :], problem_sample.x_free_indices
        )
        res = problem_sample.objective(x, return_dict=True)
        ress.append(res)
        fvals.append(res["fval"])

    # Convert the simulation to PEtab format.
    avg_model = rdatas_to_simulation_df(
        ress[np.argmin(fvals)]["rdatas"],
        model=problem_sample.objective.amici_model,
        measurement_df=importer.petab_problem.measurement_df,
    )
    return avg_model


def process_avg_model_simulation(
        avg_model: pd.DataFrame,
        df_meas: pd.DataFrame,
        dataset: str,
        samples: dict
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
    return avg_model, df_meas


def get_embedding_and_params_df(
        dmm_model: DeepMechanisticModel,
        input_features: Union[np.ndarray, jnp.ndarray],
        context: str,
        split: str,
        dataset: str,
        job: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Latent embeddings
    temp_latent_embeddings = vmap(dmm_model.deep_encoder)(input_features)
    latent_embeddings_df = pd.DataFrame(
        {
            "cell_line": dmm_model.sample_name_list,
            "L1": temp_latent_embeddings[:, 0],
            "L2": temp_latent_embeddings[:, 1],
        }
    ).assign(context=context, samples=split, dataset=dataset, job=job)

    # Get cell-line specific kinetic parameter names for dataframe column names
    specific_param_names = [
        param.replace("MED_", "") for param in dmm_model.pypesto_subproblem.x_names
        if "MED" in param
    ]

    # Parameter deviations
    param_deviations_df = pd.DataFrame(
        {
            "cell_line": dmm_model.sample_name_list,
            **{
                key: value
                for key, value in zip(
                    specific_param_names,
                    vmap(dmm_model)(input_features)["inflated"].T
                )
            },
        }
    ).assign(context=context, samples=split, dataset=dataset, job=job)

    # Full parameters (deviations + medians)
    params_df = pd.DataFrame(
        {
            "cell_line": dmm_model.sample_name_list,
            **{
                key: value
                for key, value in zip(
                    specific_param_names,
                    (vmap(dmm_model)(input_features)["inflated"] +
                     dmm_model.kin_params_combiner.learned_global_kin_params[:len(specific_param_names)]).T
                )
            },
        }
    ).assign(context=context, samples=split, dataset=dataset, job=job)
    return latent_embeddings_df, param_deviations_df, params_df


def pca_latent_embeddings(
    le_df: pd.DataFrame,
    hyperparam_configs: dict,
    scale: bool = True,
    center: bool = True,
    center_method: str = "mean",
    # col_grouping: str = "subtype_Luminal/Basal",
    # num_jobs_plot: int = 5,
):
    pca_dfs = []
    for samples, sub_df in le_df.groupby("samples"):
        for hyperparam_config in hyperparam_configs[samples]:
            # Apply filtering using a vectorized mask
            mask = np.all(
                [sub_df[key] == value for key, value in hyperparam_config.items()],
                axis=0
            )
            filtered_df = sub_df[mask].copy()
            # Check filtered_df is not empty - if empty, skip
            if filtered_df.empty:
                continue

            # Get latent embeddings
            les = filtered_df[["L1", "L2"]].values
            # Center and scale if necessary
            if center:
                les -= les.mean(axis=0) if center_method == "mean" else np.median(les, axis=0)
            if scale:
                les = StandardScaler().fit_transform(les)
            # Get 2D PCA to try and remove potential rotations between multistart embeddings
            pca = PCA(n_components=2)
            les_pca = pca.fit_transform(les)
            # Reassign to sub-DataFrame and append to context-specific list of DataFrames
            filtered_df[["L1", "L2"]] = les_pca
            pca_dfs.append(filtered_df)

        # pca_df.to_csv(
        #     evaluations_dir
        #     / f"{conf.model}"
        #     / f"{conf.data}"
        #     / f"{conf.model}.{conf.data}.{context}.latent_embeddings_pca.csv"
        # )
        # top_jobs = df.sort_values('rmse').drop_duplicates(subset='job').nsmallest(num_jobs_plot, 'rmse')["job"].values
        # plot_df = pca_df[pca_df.job.isin(top_jobs)]
        # g = sns.FacetGrid(plot_df, col=col_grouping, row="samples", hue="job")
        # g.map(plt.scatter, "L1", "L2")
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig(
        #     fig_dir / conf.model / conf.data / f"{conf.model}.{conf.data}.{context}.latent_embeddings_pca.top{num_jobs_plot}.pdf"
        # )
        # plt.show()

    # Concatenate all processed DataFrames
    return pd.concat(pca_dfs, ignore_index=True)


def cosine_similarity_embeddings(
        pca_le_df: pd.DataFrame,
        hyperparam_configs: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Pairwise comparison within same configuration, different multistarts/jobs
    job_results_dfs = []
    # Compute cosine similarity between latent embeddings among all pairs of multistarts and average
    for samples, sub_df in pca_le_df.groupby("samples"):
        # Obtain unique subconfigurations (excluding job info)
        config_df = pd.DataFrame(hyperparam_configs[samples]).drop(columns=["job"]).drop_duplicates()
        subconfigs = config_df.to_dict(orient="records")

        for subconfig in subconfigs:
            # Apply filtering using a vectorized mask
            mask = np.all(
                [sub_df[key] == value for key, value in subconfig.items()],
                axis=0
            )
            # mask = (sub_df[list(subconfig)] == pd.Series(subconfig)).all(axis=1)
            filtered_df = sub_df[mask]
            # Get latent embeddings
            jobs = sorted(filtered_df.job.unique())
            les = [filtered_df[filtered_df.job == job][["L1", "L2"]].values for job in jobs]
            val_les = [filtered_df[(filtered_df.job == job) & (filtered_df.dataset == "test")][["L1", "L2"]].values for
                       job in jobs]
            # Compute pairwise cosine similarities
            cos_sims = [cosine_similarity(les[i], les[j]).mean()
                        for i in range(len(les)) for j in range(i + 1, len(les))]
            val_cos_sims = [cosine_similarity(val_les[i], val_les[j]).mean()
                            for i in range(len(val_les)) for j in range(i + 1, len(val_les))]
            # Store results
            temp_df = pd.DataFrame([subconfig]).assign(
                mean_cos_sim=np.mean(cos_sims),
                max_cos_sim=np.max(cos_sims),
                min_cos_sim=np.min(cos_sims),
                mean_val_cos_sim=np.mean(val_cos_sims),
                max_val_cos_sim=np.max(val_cos_sims),
                min_val_cos_sim=np.min(val_cos_sims),
            )
            job_results_dfs.append(temp_df)
    # Comparison across CV splits (train/test) with same hyperparameters and jobs - only on validation cell-lines
    # Step 1: Subset to validation cell-lines
    val_pca_le_df = pca_le_df[pca_le_df.cell_line.isin(hardest_cell_lines)]
    # Step 2: Find cell lines that appear in both train and test datasets
    valid_cell_lines = (
        val_pca_le_df.groupby('cell_line')['dataset']
        .apply(lambda x: set(x) == {'train', 'test'})
        .loc[lambda x: x]  # Keep only True values
        .index
    )
    # Step 3: Keep only rows where cell_line is in valid_cell_lines
    val_pca_le_df = val_pca_le_df[val_pca_le_df.cell_line.isin(valid_cell_lines)]
    # Step 4: For each configuration and cell-line, compute cosine similarities between the CV split where
    # the cell-line is in validation and those where it is in training and average
    cosine_results = []

    # Group by configuration, job, and cell_line to process each group separately
    group_cols = [col for col in val_pca_le_df.columns if col not in ['L1', 'L2', 'samples', 'dataset']]
    for (group_params), group in val_pca_le_df.groupby(group_cols):
        # Split into train and test subsets
        train_subset = group[group['dataset'] == 'train'][["L1", "L2"]].values
        test_subset = group[group['dataset'] == 'test'][["L1", "L2"]].values
        # Compute cosine similarity between train and test subsets
        similarity = cosine_similarity(train_subset, test_subset).mean()
        # Store the result
        cosine_results.append({**dict(zip(group_cols, group_params)), 'cosine_similarity': similarity})

    # Step 5: create dataframe where each configuration (+job) has associated this mean cosine similarity and the list of all CV split cosine similarities
    cosine_df = pd.DataFrame(cosine_results).sort_values(by="cell_line") # ensure consistent ordering of CV splits
    # Group by config and job to compute the mean cosine similarity across cell lines
    cosine_summary_df = (
        cosine_df.groupby([col for col in group_cols if col != "cell_line"])['cosine_similarity']
        .agg(['mean', list])  # Compute mean and keep list of all cosine similarities
        .reset_index()
        .rename(columns={'mean': 'mean_cosine_similarity', 'list': 'cv_split_cosine_similarities'})
        .sort_values(by=["job"])  # ensures consistent ordering of jobs prior to operation below
    )
    # Step 6: average across the multistarts for each configuration (keep mean list of CV split-wise mean cosine similarities)
    # Group by configuration only and average across jobs (multistarts)
    final_summary_df = (
        cosine_summary_df.groupby([col for col in group_cols if col not in ["cell_line", "job"]])
        .agg({
            'mean_cosine_similarity': 'mean',
            'cv_split_cosine_similarities': lambda lists: np.mean(lists.tolist(), axis=0)
        })
        .reset_index()
    )
    # Concatenate all processed DataFrames
    return pd.concat(job_results_dfs, ignore_index=True), final_summary_df


def silhouette_embeddings(
        pca_le_df: pd.DataFrame,
        hyperparam_configs: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Obtain unique subconfigurations (excluding job & CV-split info)
    config_dfs = []
    for samples in hyperparam_configs.keys():
        config_dfs.append(pd.DataFrame(hyperparam_configs[samples]).drop(columns=["job", "samples"]).drop_duplicates())
    config_df = pd.concat(config_dfs, ignore_index=True).drop_duplicates()
    subconfigs = config_df.to_dict(orient="records")

    # Job-wise, all cell-lines
    all_lines_results = []
    for samples in hyperparam_configs.keys():
        sub_df = pca_le_df[pca_le_df.samples == samples]
        for subconfig in subconfigs:
            # Apply filtering using a vectorized mask
            mask = np.all(
                [sub_df[key] == value for key, value in subconfig.items()],
                axis=0
            )
            embeddings = sub_df[mask][["L1", "L2"]].values
            labels = sub_df[mask].job.values
            # Compute silhouette score, mean and standard deviation and store results
            all_lines_results.append(
                pd.DataFrame([subconfig]).assign(
                    samples=samples,
                    mean_silhouette_score=silhouette_score(embeddings, labels),
                    stddev_silhouette_score=np.std(silhouette_samples(embeddings, labels)),
                    all_scores=[silhouette_samples(embeddings, labels)],
                )
            )

    val_cell_lines = pca_le_df[pca_le_df.dataset == "test"].cell_line.unique()
    # Job-wise, validation only: subset w.r.t. validation cell-line & config.; label with job ID; compute silhouette score
    job_results_dfs = []
    # CV-split/Dataset-wise, validation only: subset w.r.t. validation cell-line & config.; label with dataset; compute silhouette score
    cv_results = []
    for cell_line in val_cell_lines:
        sub_df = pca_le_df[pca_le_df.cell_line == cell_line]
        for subconfig in subconfigs:
            # Apply filtering using a vectorized mask
            mask = np.all(
                [sub_df[key] == value for key, value in subconfig.items()],
                axis=0
            )
            embeddings = sub_df[mask][["L1", "L2"]].values
            job_labels = sub_df[mask].job.values
            dataset_labels = sub_df[mask].dataset.values
            # Compute silhouette score, mean and standard deviation and store results
            job_results_dfs.append(
                pd.DataFrame([subconfig]).assign(
                    cell_line=cell_line,
                    mean_silhouette_score=silhouette_score(embeddings, job_labels),
                    stddev_silhouette_score=np.std(silhouette_samples(embeddings, job_labels)),
                    all_scores=[silhouette_samples(embeddings, job_labels)]
                )
            )
            cv_results.append(
                pd.DataFrame([subconfig]).assign(
                    cell_line=cell_line,
                    mean_silhouette_score=silhouette_score(embeddings, dataset_labels),
                    stddev_silhouette_score=np.std(silhouette_samples(embeddings, dataset_labels)),
                    all_scores=[silhouette_samples(embeddings, dataset_labels)]
                )
            )
    return (pd.concat(all_lines_results, ignore_index=True),
            pd.concat(job_results_dfs, ignore_index=True),
            pd.concat(cv_results, ignore_index=True))
