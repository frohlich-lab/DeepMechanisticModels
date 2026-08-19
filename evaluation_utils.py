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
from sklearn.neighbors import kneighbors_graph

from common import (
    MEASUREMENTS_FILE,
    OBSERVABLES_FILE,
    REGRESSION_MODES,
    TRAINED_MODEL,
    Wildcards,
    evaluations_dir,
    scan_attributes,
    training_samples,
    val_samples,
)
from cytof.problem import CytofProblem
from dmm.config_options import Conf
from dmm.dmm_autoencoder_eqx import DeepMechanisticModel
from dmm.petab_subproblem import load_petab
from dmm.pretraining import (
    generate_average_pretraining_problem,
    generate_per_sample_pretraining_problems,
)


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


def load_model(
    conf: Conf,
    pypesto_subproblem,
) -> tuple[DeepMechanisticModel, Any]:
    # Define filepaths for serialized models
    trained_model_file = TRAINED_MODEL.format(**conf.to_dict())

    model = DeepMechanisticModel.load(
        filename=trained_model_file,
        pypesto_problem=pypesto_subproblem,
    )
    return model


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
    if not len(input_features):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    # Latent embeddings
    temp_latent_embeddings = vmap(
        eqx.nn.inference_mode(dmm_model).encode, in_axes=(0, None)
    )(jnp.array(input_features), jr.PRNGKey(0))

    n_components = min(
        temp_latent_embeddings.shape[1], dmm_model.conf.n_hidden
    )

    latent_embeddings_df = pd.DataFrame(
        {
            "cell_line": samples,
            **{
                f"L{i + 1}": temp_latent_embeddings[:, i]
                for i in range(n_components)
            },
        }
    ).assign(
        context=context,
        samples=split,
        dataset=dataset,
        job=job,
    )

    # Parameter deviations
    param_deviations_df = pd.DataFrame(
        {
            "cell_line": samples,
            **dict(
                zip(
                    dmm_model.parameter_deviation_names,
                    eqx.nn.inference_mode(dmm_model)
                    .inflate_params(jnp.array(input_features), jr.PRNGKey(0))
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
                            jnp.array(input_features), jr.PRNGKey(0)
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
                ["samples"],
                [
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

    return pd.concat(cv_results, ignore_index=True)


def convert_dataframe_dtypes(df: pd.DataFrame):
    cols = [
        "n_hidden",
        "depth",
        "nn_structure_multiplier",
        "inflater_output_reg_epoch",
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
    num_best: int = 10,
):
    outdir = evaluations_dir / conf.model / conf.data
    # Define aggregation groups for DMM and refs
    gbs_dmm = ["dataset", "ref"] + scan_attributes

    df["res"] = df["res"].astype(float)
    df = df[np.isfinite(df["res"])]

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
    # cleanup
    del temp_dfs

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
    by_cl_cond_obs.to_csv(outdir / f"by_cl_cond_obs_{conf.figure}.csv")
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
        best_configs_dmm.to_csv(outdir / f"top1_best_dmm_{conf.figure}.csv")

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
        f"evaluate_all_{conf.figure}",
    ]
    for evaluation_df, evaluation_tag in zip(evaluation_dfs, evaluation_tags):
        # Save dataframes to CSV
        evaluation_df.to_csv(outdir / f"{evaluation_tag}.csv")

    return (
        data,
        top_n_dmm_train,
        best_configs_dmm_jobs,
        best_regressors,
        unified_dmm_results,
    )
