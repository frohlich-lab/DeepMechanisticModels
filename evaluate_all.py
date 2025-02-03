import fire
import itertools as itt
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
# import subprocess
# import wandb

from common import (
    CONTEXT_SET,
    default_attributes,
    evaluations_dir,
    EVALUATION_REFERENCE,
    EVALUATION_REGRESSOR,
    EVALUATION_TRAINING,
    EVALUATION_EMBEDDING,
    EVALUATION_PARAMETER_DEVIATIONS,
    EVALUATION_FULL_PARAMETERS,
    FEATURES_OUTFILE,
    FEATURES_PIPELINE,
    fig_dir,
    hardest_cell_lines,
    pretrain_dir,
    REGR_FEATURES_TRAIN,
    REGR_TRAINED_PIPELINE,
    subtypes_tognetti,
    training_samples,
    test_samples,
    Wildcards,
)
from cytof.problem import CytofProblem
from dataclasses import replace
from dmm.analysis import plot_loss_vs_regularization, simulate_dmm
# from dmm.autoencoder import DeepMechanisticModel
from dmm.config_options import Conf
from dmm.feature_selection import load_data
from dmm.initialisation import get_features, get_features_filepaths, pca_transform_features, impute_features
from dmm.plotting import plot_cross_samples_multiple_simulations
from evaluation_plotting import (n_hidden_pairwise_heatmap, performance_barplot,
                                 volcano_hyperparameter_significance,
                                 plot_latent_embeddings, plot_val_param_dev_spread, plot_parameter_heatmaps)
from evaluation_utils import (get_measurements_and_obervables,
                              load_model_and_obj,
                              simulate_avg_model,
                              process_avg_model_simulation,
                              process_per_sample_pretrain, get_embedding_and_params_df,
                              pca_latent_embeddings,
                              cosine_similarity_embeddings, silhouette_embeddings, connectivity_score)
from generate_run_configs import generate_run_configs
from jax import vmap
from joblib import load
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from stat_test import statistical_significance_test
from training_configuration import (
    CONTEXTS_FEATURES, SPLITS, RETURN_STAT_TESTS, HP_RUN_MODE, REFINE_HPS, SPLITS
)
from typing import List
from util import load_petab_base_files


REGRESSION_MODES = ["linreg", "lasso", "elasticnet"]


def get_dmm_conf(
        conf: Conf,
        dmm_params: dict,
        dataset: str,
        context: str,
) -> Conf:
    """
    Get the DMM configuration for a given dataset and context, ensuring integer values are cast as integers.
    :param conf: configuration object (Conf)
    :param dmm_params: dictionary of parameters for DMM
    :param dataset: dataset (train/val)
    :param context: context (cytof_init/proteomics/transcriptomics/MOSA)

    return: updated configuration object (Conf)
    """
    dmm_conf = Conf(model=conf.model, data=conf.data)
    for key, value in dmm_params[dataset][context].items():
        # Cast values to integer (always or if the value is zero)
        if hasattr(dmm_conf, key) and key not in ["model", "data"]:
            if key in [
                "n_hidden", "depth", "nn_structure_multiplier",
                "inflater_output_reg_epoch",
                "opt_steps", "opt_mult",
                "job"
            ]:
                value = int(value)
            elif key in [
                "l1reg_encode", "oreg_encode",  # encoder
                "l1reg_inflate", "oreg_inflate", "l1reg_inflater_output", # inflater
                "recon_loss", "symm_reg",  # decoder / reconstruction
                "median_reg",  # kinetic params median regularisation
                "opt_steps", "opt_mult", "momentum"  # parameters that can be pruned by generate_run_configs
            ] and value == 0:
                value = int(value)

            setattr(dmm_conf, key, value)
    return dmm_conf


def convert_dataframe_dtypes(df: pd.DataFrame):
    cols = ["n_hidden", "depth", "nn_structure_multiplier",
            "inflater_output_reg_epoch",
            "opt_steps", "opt_mult",
            "job"]
    for col in cols:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    additional_cols = ["l1reg_encode", "oreg_encode",  # encoder
            "l1reg_inflate", "oreg_inflate", "l1reg_inflater_output", # inflater
            "recon_loss", "symm_reg",  # decoder / reconstruction
            "median_reg",  # kinetic params median regularisation
            "opt_steps", "opt_mult", "momentum"  # parameters that can be pruned by generate_run_configs
    ]
    for col in additional_cols:
        if (len(df[col].unique()) == 1) and (df[col].unique()[0] == 0):
            df[col] = pd.to_numeric(df[col], downcast='integer')
        else:
            df[col] = df[col].astype("float")
    for col in ["pretrain", "linear_benchmark", "use_layer_bias", "last_layer_activation",
                "drop_reg_after_pretrain"]:
        df[col] = df[col].astype(str)
    for col in ["reconstruct", "use_simple_linear_schedule", "use_early_stopping"]:
        df[col] = df[col].astype(bool)
    return df


def load_and_transform_features(
        conf: Conf,
        dataset: str
) -> np.ndarray:
    # Compute features filepath given conf and dataset
    features_filepath = FEATURES_OUTFILE.format(
        **{**conf.__dict__, **dict(dataset='{dataset}')}
    )
    # Compute filepath for feature transformation pipeline
    feature_transform_pipeline_filepath = Path(
        FEATURES_PIPELINE.format_map(conf.__dict__)
    )
    features = get_features(
        features_filepath=features_filepath,
        datasets=['train', 'val']
    )
    if conf.features_transform == "pca":
        # Load pre-trained pipeline if it exists
        if os.path.exists(feature_transform_pipeline_filepath):
            pipeline = load(feature_transform_pipeline_filepath)
        else:
            pipeline = None
        features = pca_transform_features(
            features=features,
            pipeline_filepath=feature_transform_pipeline_filepath,
            pipeline=pipeline,
        )
    else:
        features = impute_features(features)
    if dataset == 'train':
        features_dataset = 'train'
    elif dataset == 'test':
        features_dataset = 'val'
    # TODO @GiacomoFabrini - will need to change this when we resolve 'val' vs 'test' ambiguity
    return features[features_dataset].values


def process_reference(
        conf: Conf,
        samples: str,
        dataset: str,
        mode: str,
        ref_name: str
) -> pd.DataFrame:
    print(f'Processing {mode} model for {samples}, {dataset}')
    ref = pd.read_csv(
        EVALUATION_REFERENCE.format(
            **{
                **conf.__dict__,
                **dict(
                    samples=samples,
                    dataset=dataset,
                ),
            },
            mode=mode,
        ),
        index_col=0,
    )
    ref["ref"] = ref_name
    print(f'Finished processing {mode} model for {samples}, {dataset}')
    return ref


def get_best_performer_across_jobs(
        dataframe: pd.DataFrame,
        group_attributes: List,
        hyperparam_attributes: List,
        mode: str,
        num_best: int,
        target_attribute='rmse',
):
    """
    Returns a pandas DataFrame with the hyperparameter setting producing
    the lowest mean RMSE across all jobs and all cross-validation SPLITS (samples)
    for each combination of the group_attributes (dataset = 'train'/'test',
    context = 'cytof_init' / 'proteomics', 'transcriptomics',
    ref = 'DMM'). The returned DataFrame reports both the mean and
    the standard deviation of RMSEs across the 10 jobs and however many SPLITS.
    """
    if mode == 'DMM':
        # Ensure dataframe is sorted by "job" so that "rmse_list" always follows the same order of increasing job number
        temp_df = dataframe.sort_values(by="job").reset_index().groupby(
            group_attributes + hyperparam_attributes
        ).agg({
            target_attribute: ['mean', 'std',  ("list", lambda x: list(x))],
        })
        temp_df = temp_df.reset_index()
        temp_df.columns = [
            ' '.join(col).strip() if isinstance(col, tuple) else col
            for col in temp_df.columns.values
        ]
        # Check passed num_best is acceptable (at least 1, integer)
        if num_best < 1:
            raise ValueError(f"num_best must be >=1, {num_best} was found instead.")
        elif not isinstance(num_best, int):
            raise TypeError(f"num_best must be of type int, {type(num_best)} was found instead.")

        # sort (default: ascending order - from lowest rmse mean to highest -- keep num_best (>=1)
        result = temp_df.sort_values(by=[target_attribute + ' mean']).groupby(group_attributes).head(num_best)
        result_dict = {
            dataset: {
                context: result[(result.dataset == dataset) & (result.context == context)].iloc[:num_best].to_dict(
                    orient='records')
                for context in result[result.dataset == dataset].context.unique()
            }
            for dataset in result.dataset.unique()
        }
        # TODO need to ensure that this is returning the same configurations across train/test, not the best
        #  in each, even though we can simply subset to the best in `val` and simulate across both train and test
        return result_dict
    elif mode == 'regressor':
        min_rmse = dataframe.reset_index().groupby(
            group_attributes + hyperparam_attributes
        )[target_attribute].min()
        result = pd.merge(dataframe, min_rmse, on=['context', 'dataset', target_attribute])
        result_dict = {
            dataset: {
                context: result[(result.dataset == dataset) & (result.context == context)].iloc[0].ref
                for context in result[result.dataset == dataset].context.unique()
            }
            for dataset in result.dataset.unique()
        }
        return result_dict
    else:
        raise ValueError(f"Invalid mode: {mode}")


def get_absolute_best_performer(
        dataframe: pd.DataFrame,
        group_attributes: List,
        target_attribute='rmse',
):
    # Potential issues: this returns the first occurring minimum - there might be
    # ties, but this is unlikely within numerical accuracy.
    temp_dataframe = dataframe.reset_index()
    min_rmse_indices = temp_dataframe.groupby(
        by=group_attributes
    )[target_attribute].idxmin()
    result = temp_dataframe.loc[min_rmse_indices]
    result_dict = {
        dataset: {
            context: result[(result.dataset == dataset) & (result.context == context)].iloc[0].to_dict()
            for context in result[result.dataset == dataset].context.unique()
        }
        for dataset in result.dataset.unique()
    }
    return result_dict


def aggregate_and_log(df: pd.DataFrame, return_stat_tests: bool, num_best: int):
    # Define aggregation groups for DMM
    gbs_dmm = ["dataset", "ref"] + default_attributes
    # replace missing values in features_transform (None instead of nan)
    df["features_transform"] = df["features_transform"].replace(np.nan, "None")
    data_dmm = pd.DataFrame(
        [
            dict(
                zip(gbs_dmm, group),
                rmse=np.sqrt(np.square(group_df["res"]).mean()),  # RMSE
            )
            for group, group_df in df.groupby(gbs_dmm)
        ]
    )

    # Define aggregation groups for references
    gbs_refs = [
        "dataset",
        "context",
        "samples",
        "ref",
    ]

    df_refs = df[~df.ref.isin(["DMM"])]
    data_refs = pd.DataFrame(
        [
            dict(
                zip(gbs_refs, group_ref),
                rmse=np.sqrt(np.square(group_df_ref["res"]).mean()),  # mean RMSE = RMSE (single values)
            )
            for group_ref, group_df_ref in df_refs.groupby(gbs_refs)
        ]
    )

    data = pd.concat([data_dmm, data_refs]).sort_values(by="ref")
    # print("Overall evaluation DataFrame is now ready.")
    # cleanup
    del df, df_refs, data_dmm, data_refs

    if return_stat_tests:
        # Prepare statistical test dataframe
        # Create pivot table for statistical testing
        cols = [
            'dataset', 'context', 'features', 'ref',
            'n_hidden', 'orth_reg_strategy',
            'l1reg_inflate', 'oreg_inflate', 'l1reg_encode', 'oreg_encode'
        ]
        # pivot table and create one column per cross-validation split and multistart/job
        pivot_data = data.pivot_table(index=cols, columns=['samples', 'job'], values='rmse')
        pivot_data = pivot_data.reset_index()
        # Create list of the MultiIndex RMSE columns created above
        multiindex_rmse_cols = [(sample, job) for sample in SPLITS for job in JOBS]
        # Create a single column 'rmse_list' listing all values from each of the MultiIndex columns
        pivot_data['rmse_list'] = pivot_data.apply(lambda row: np.array([row[col] for col in multiindex_rmse_cols]),
                                                   axis=1)
        # Add the newly created column to the list of columns to be kept (cols)
        cols += ['rmse_list']
        # Subset the pivot table and reduce MultiIndex back to single-level index
        data_stat_tests = pivot_data[cols]
        data_stat_tests.columns = data_stat_tests.columns.droplevel(level=1)
        print("DataFrame for statistical testing is now ready.")

        stat_test_res_df = statistical_significance_test(data_stat_tests)

    # Get best performing hyperparameter set across jobs for each dataset/context/ref combination
    top_n_hyperparam_dmm = get_best_performer_across_jobs(
        dataframe=data[data.ref == 'DMM'],
        group_attributes=['dataset', 'context', 'features', 'pretrain'],
        hyperparam_attributes=[x for x in default_attributes if x not in [
            'context', 'features', 'pretrain',  # in group_attributes
            'samples', 'job'  # need to average across samples and job
        ]],
        mode='DMM',
        num_best=num_best,  # select the best `num_best` per context (e.g. 10 best configurations per context)
        target_attribute='rmse',
    )
    # Keep top 1 on validation set for plotting across samples vs references and regressors
    # Assign the best validation configuration to both train and test -- ensures consistency in plots
    # TODO: from this point onwards, RMSE is only valid for 'test', not for 'train' (not used anyway). 'train' RMSE
    #  is overwritten with 'test' RMSE
    best_hyperparam_dmm = {
        dataset: {
            context: top_n_hyperparam_dmm["test"][context][0]
            for context in top_n_hyperparam_dmm[dataset].keys()
        }
        for dataset in top_n_hyperparam_dmm.keys()
    }
    best_regressors = get_best_performer_across_jobs(
        dataframe=data[data.ref.isin(REGRESSION_MODES)],
        group_attributes=['dataset', 'context'],
        hyperparam_attributes=[],
        mode='regressor',
        num_best=1,  # does not have an effect anyway, but we are selecting the top 1
        target_attribute='rmse',
    )
    # DISABLED best single-job/multistart DMM configuration - not useful, interested in mean multistart performance
    # # Get absolute best performing hyperparameter set (single job) for each dataset/context/ref combination
    # absolute_best_dmm = get_absolute_best_performer(
    #     dataframe=data[data.ref == 'DMM'],
    #     group_attributes=['dataset', 'context', 'ref'],
    #     target_attribute='rmse',
    # )
    # # Ensure consistency in plot - only keep the best absolute in validation and plot it across both train and val
    # # TODO: from this point onwards, RMSE is only valid for 'test', not for 'train' (not used anyway). 'train' RMSE
    # #  is overwritten with 'test' RMSE
    # absolute_best_dmm = {
    #     dataset: {
    #         context: absolute_best_dmm["test"][context]
    #         for context in absolute_best_dmm[dataset].keys()
    #     }
    #     for dataset in absolute_best_dmm.keys()
    # }

    # # Log via W&B -- DISABLED WANDB
    # wandb.init(
    #     project=f"DeepMechanisticModels.{conf.data}.{conf.model}",
    #     config={
    #         **conf.__dict__,
    #     },
    # )

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
        evaluation_df.to_csv(
            evaluations_dir
            / f"{conf.model}"
            / f"{conf.data}"
            / f"{conf.model}.{conf.data}.{evaluation_tag}.csv"
        )

        # DISABLED WANDB
        # # Instantiate artifact
        # evaluation_artifact = wandb.Artifact(
        #     name=f"{evaluation_tag}_{conf.model}_{conf.data}",
        #     description=evaluation_tag,
        #     type="evaluation",
        # )
        # # Add and log artifact
        # evaluation_artifact.add(wandb.Table(dataframe=data), f"{evaluation_tag}.csv")
        # wandb.log_artifact(evaluation_artifact)

    # Close W&B session and upload artifacts -- DISABLED WANDB
    # wandb_stripped_dir = wandb.run.dir.rsplit('/files', 1)[0]
    # command = f"wandb sync {wandb_stripped_dir}"
    # wandb.finish()
    # TODO restore once done fixing script
    # try:
    #     _ = subprocess.run(command, shell=True)
    # except subprocess.CalledProcessError as e:
    #     raise ValueError(f"Error syncing wandb directory: {e}")

    # Removed absolute_best_dmm
    if return_stat_tests:
        return data, stat_test_res_df, top_n_hyperparam_dmm, best_hyperparam_dmm, best_regressors
    else:
        return data, top_n_hyperparam_dmm, best_hyperparam_dmm, best_regressors


conf = fire.Fire(Conf)

outdir = fig_dir / conf.model / conf.data

# METHODS = ("pca embedding", "end-to-end")  # not used at the moment

JOBS = tuple([i for i in range(conf.n_starts)])

# Compute run configurations and arrange by CV split
hyperparam_configs = generate_run_configs(
    n_starts=conf.n_starts,
    hp_run_mode=HP_RUN_MODE,
    refine_hps=REFINE_HPS,
)
hyperparam_configs = {
    samples: [
        hyperparam_config for hyperparam_config in hyperparam_configs
        if hyperparam_config['samples'] == samples
    ]
    for samples in SPLITS
}
dfs, le_dfs, param_dev_dfs, param_dfs = [], [], [], []
for samples in sorted(list(SPLITS)):  # process from 0of5 to 4of5
    for dataset in ["train","test"]:
        # DMM evaluations
        training = pd.concat(
            pd.read_csv(efile, index_col=0)
            for hyperparam_configuration in hyperparam_configs[samples]
            if os.path.exists(
                efile := EVALUATION_TRAINING.format_map(
                    {
                        **hyperparam_configuration,
                        'model': conf.model,
                        'data': conf.data,
                        'dataset': dataset,
                        'samples': samples
                    }
                )
            )
        ).assign(
            ref="DMM",  # previously called "meth"
            dataset=dataset
        )
        print(f'Finished concatenating training evaluations for {samples}, {dataset}')

        # Loss vs regularization plot -- DISABLED, not useful right now
        # print(f'Starting to plot loss_vs_regularization for {samples}, {dataset}')
        # plot_loss_vs_regularization(training)
        # plt.savefig(outdir / f"{samples}_evaluate_training_{dataset}.pdf")
        # print(f'Saved loss_vs_regularization plot for {samples}, {dataset}')

        # Add necessary attributes to training DataFrame
        training["ref"] = "DMM"  # previously "meth"
        training["dataset"] = dataset


        # concatenate embeddings, parameter deviations and parameters
        temp_results = {}
        for result_type, filepath_format in zip(
                ["latent_embeddings", "parameter_deviations", "full_parameters"],
                [EVALUATION_EMBEDDING, EVALUATION_PARAMETER_DEVIATIONS, EVALUATION_FULL_PARAMETERS]
        ):
            temp_results[result_type] = pd.concat(
                pd.read_csv(efile, index_col=0).assign(**hyperparam_configuration)
                for hyperparam_configuration in hyperparam_configs[samples]
                if os.path.exists(
                    efile := filepath_format.format_map(
                        {
                            **hyperparam_configuration,
                            'model': conf.model,
                            'data': conf.data,
                            'dataset': dataset,
                            'samples': samples
                        }
                    )
                )
            )
            print(f'Finished concatenating {result_type} for {samples}, {dataset}')

        # average (not in use)
        # avg = process_reference(conf, samples, dataset, "average", "avg")

        # model average (avg_model)
        avg_model = process_reference(conf, samples, dataset, "avg_model", "avg_model")

        # Get references (avg_model, per_sample)
        avg_model, ps = [
            process_reference(conf, samples, dataset, mode, ref_name)
            for mode, ref_name in zip(["avg_model", "per_sample"], ["avg_model", "sample"])
        ]

        # Process regressors - linreg, lasso, elasticnet
        print(f'Processing regressors model for {samples}, {dataset}')
        regressor_dfs = {
            mode: pd.concat(
                pd.read_csv(
                    EVALUATION_REGRESSOR.format(
                        **{
                            **conf.__dict__,
                            **dict(
                                samples=samples,
                                dataset=dataset,
                                context=ctxt,
                            ),
                        },
                        mode=mode,
                    ),
                    index_col=0,
                )
                for ctxt, features in CONTEXTS_FEATURES
            ).assign(ref=mode, samples=samples, dataset=dataset)
            for mode in REGRESSION_MODES
        }
        print(f'Finished processing regressors for {samples}, {dataset}')

        print(f'Starting to build hyperparam/job combination copies for references models - {samples}, {dataset}')
        # Removed addition of None hyperparameters - already done before the `process_simulations` step
        avg_ps_dfs = []
        for context in CONTEXT_SET:
            # need to replicate info across contexts for "avg_model" and "sample"
            for rdf in [  # lack context
                # avg,
                avg_model,
                ps,
            ]:
                avg_ps_df = rdf.copy()
                avg_ps_df = (avg_ps_df
                             .assign(context=context, samples=samples, dataset=dataset)
                             .replace(np.nan, "N/A"))  # replace NaNs with "N/A" to avoid FutureWarning re. empty/NaN entries
                avg_ps_dfs.append(avg_ps_df)
                # Once appended, this can be deleted
                del avg_ps_df

        # regression baselines already have context
        for _, rdf in regressor_dfs.items():
            avg_ps_df = rdf.copy()
            avg_ps_dfs.append(avg_ps_df)
            # Once appended, this can be deleted
            del avg_ps_df
        print(f"Finished processing reference models for {samples}, {dataset}")

        # dfd = pd.concat([training, pretraining])
        # TODO @GiacomoFabrini might it be better to have default activation, optimiser, orth_reg_strategy as "None"?
        dfd = pd.concat([training.convert_dtypes(), *avg_ps_dfs])
        print(f"Finished concatenating training and reference models for {samples}, {dataset}")
        # Deleting DataFrames once concatenated into dfd
        del training, avg_ps_dfs, rdf
        dfs.append(dfd)
        le_dfs.append(temp_results["latent_embeddings"])
        param_dev_dfs.append(temp_results["parameter_deviations"])
        param_dfs.append(temp_results["full_parameters"])
        # Deleting dfd once appended to dfs
        del dfd, temp_results

df = pd.concat(dfs).reset_index()
# Now that dfs have been concatenated into df, delete them
del dfs
le_df, param_dev_df, param_df = [pd.concat(dfs) for dfs in [le_dfs, param_dev_dfs, param_dfs]]
for results_df in [le_df, param_dev_df, param_df]:
    results_df["job"] = results_df["job"].astype(int)
del le_dfs, param_dev_dfs, param_dfs

# ########################################################################### #
# ############################### Aggregation ############################### #
# ########################################################################### #

# Aggregate data, save CSVs and log W&B artifacts (currently disabled)
num_best = 10
aggregated_results = aggregate_and_log(
    df=df, return_stat_tests=RETURN_STAT_TESTS, num_best=num_best
)
if RETURN_STAT_TESTS:
    data, stat_test_res_df, top_n_dmm, best_hyperparam_dmm, best_regressors = aggregated_results
else:
    data, top_n_dmm, best_hyperparam_dmm, best_regressors = aggregated_results


# ########################################################################### #
# ################### Save train/test RMSE dataset in CSV ################### #
# ########################################################################### #
dmm_results = {
    dataset: (data[(data.dataset == dataset) & (data.ref == 'DMM')]
              .drop(columns=['ref', 'dataset']))
    for dataset in ['train', 'test']
}
merge_cols = data.columns.difference(['rmse', 'dataset', 'ref'])
unified_dmm_results = pd.merge(
    dmm_results['train'],
    dmm_results['test'],
    how="inner",
    on=list(data.columns.difference(['rmse', 'dataset', 'ref'])),
    suffixes=('_train', '_test')
)
unified_dmm_results.to_csv(
    evaluations_dir
    / f"{conf.model}"
    / f"{conf.data}"
    / f"{conf.model}.{conf.data}.unified_dmm_rmse_train_test.csv"
)
print("Finished saving unified (train/test) DMM RMSE results.")


# ########################################################################### #
# ##################### Top 10 jobs (train) per config ###################### #
# ########################################################################### #
config_cols = list(next(iter(hyperparam_configs.values()))[0].keys())
config_cols.remove("job")
top_n_train = 10
# Subset to top N=10 jobs for each configuration according to rmse_train metric
top_n_dmm_train = unified_dmm_results.groupby(config_cols).apply(
    lambda x: x.nsmallest(top_n_train, "rmse_train")
).reset_index(drop=True)
# Ensure same dtypes as original dataframe
top_n_dmm_train = convert_dataframe_dtypes(top_n_dmm_train)

# Subset parameter deviation, parameter and latent embeddings to top N=10 jobs
top_n_param_dev_df_train, top_n_param_df_train, top_n_le_df_train = [
    df.merge(
        top_n_dmm_train,
        how="inner",
        on=(config_cols + ["job"])
    )[df.columns]
    for df in [param_dev_df, param_df, le_df]
]

# Compute PCA latent embeddings -- from [2 (LE1, LE2) * num_top_jobs]
# features/columns down to [2 (LE1*, LE2*)] components. Auto-centering
# performed through PCA itself.
top_n_pca_le_df_train = pca_latent_embeddings(
    top_n_le_df_train, hyperparam_configs, scale=False
).reset_index()
top_n_pca_le_df_train.to_csv(
    evaluations_dir
    / f"{conf.model}"
    / f"{conf.data}"
    / f"{conf.model}.{conf.data}.top_{num_best}_pca_latent_embeddings.csv"
)

# Select reg_param for plotting based on the number of unique investigated values
reg_params = [
    "l1reg_inflate", "oreg_inflate",   # inflater
    "l1reg_encode", "oreg_encode",  # encoder
    "l1reg_inflater_output", "median_reg"  # param dev, param medians
]
num_unique_regs = [len(top_n_pca_le_df_train[reg_param].unique()) for reg_param in reg_params]
reg_param = reg_params[num_unique_regs.index(max(num_unique_regs))]

for (latent_embedding_df, df_label), which_cells in itt.product(
    zip([top_n_le_df_train, top_n_pca_le_df_train], ["pristine", "pca"]),
        ["all", "val_only"]
):
    plot_latent_embeddings(
        le_df=latent_embedding_df,
        df_label=df_label,
        reg_param=reg_param,
        save_path=str(
            outdir / "{context}.latent_embeddings.{df_label}.{which_cells}.{plot_by}.pdf"
        ),
        which_cells=which_cells,
    )

plt.close("all")
sns.boxplot(top_n_pca_le_df_train, x=reg_param, y="variance_explained")
plt.tight_layout()
plt.savefig(
    outdir / f"pca.latent_embeddings.{reg_param}.variance_explained.pdf"
)
plt.close()
del dmm_results, unified_dmm_results, merge_cols, rmse_train_targets, rmse_val_targets, dummy_unified_dmm_results

# ########################################################################### #
# ################### Save information on top N best DMM #################### #
# ########################################################################### #

# Step 1: Extract data and flatten the structure
flat_top_n_dmm = []
for dataset, contexts in top_n_dmm.items():
    for context, context_list in contexts.items():
        for context_dict in context_list:
            flat_top_n_dmm.append({**{'dataset': dataset, 'context': context}, **context_dict})

# Step 2: Create a DataFrame
best_n_dmm_df = pd.DataFrame(flat_top_n_dmm)
best_n_dmm_df.to_csv(
    evaluations_dir
    / f"{conf.model}"
    / f"{conf.data}"
    / f"{conf.model}.{conf.data}.top_{num_best}_best_dmm_{HP_RUN_MODE}.csv"
)

# ########################################################################### #
# ############################ Performance Plots ############################ #
# ########################################################################### #

# group_plots(
#     dataframe=data,
#     conf=conf
# )
#

# TODO @GiacomoFabrini -- review this in light of new subsetting on RMSE_train!?!
# Get top 10 train jobs in `data` and plot barplots for all jobs and top10 train
data_dmm = data[data.ref == 'DMM']
data_nodmm = data[data.ref != 'DMM']
top_dmm_for_merge = top_n_dmm_train[
    [col for col in top_n_dmm_train if col not in ["rmse_train", "rmse_test", "dataset"]]
]
data_dmm = convert_dataframe_dtypes(data_dmm)
data_dmm_top10 = data_dmm.merge(
    top_dmm_for_merge,
    how='inner',
    on=[col for col in top_dmm_for_merge.columns]
)
data_top10_refs = pd.concat(
    [data_dmm_top10, data_nodmm]
).reset_index().drop(columns=['index'])
del data_dmm, data_nodmm, top_dmm_for_merge, data_dmm_top10

for dataframe, barplot_label in zip(
        [data, data_top10_refs], ["baseline_barplot", "baseline_barplot_top10"]
):
    performance_barplot(
        dataframe=dataframe,
        conf=conf,
        group_name=barplot_label,
    )


# ########################################################################### #
# ########################### Embedding Similarity ########################## #
# ########################################################################### #
# Cosine-similarity
cv_cos_sim = cosine_similarity_embeddings(top_n_pca_le_df_train, hyperparam_configs)
cv_cos_sim.to_csv(
    evaluations_dir
    / f"{conf.model}"
    / f"{conf.data}"
    / f"{conf.model}.{conf.data}.cosine_sim_cv.csv"
)
# Silhouette score
cv_silhouette = silhouette_embeddings(top_n_pca_le_df_train, hyperparam_configs)
cv_silhouette.to_csv(
    evaluations_dir
    / f"{conf.model}"
    / f"{conf.data}"
    / f"{conf.model}.{conf.data}.silhouette_cv.csv"
)
g = sns.FacetGrid(
    cv_silhouette,
    row="context", row_order=sorted(cv_silhouette.context.unique()),
    hue="cell_line"
)
g.map_dataframe(sns.scatterplot, x=reg_param, y="mean_silhouette_score")
plt.tight_layout()
plt.legend()
plt.xscale("symlog")
plt.savefig(fig_dir / conf.model / conf.data / f"mean_silhouette_score_{reg_param}.pdf")
plt.close()
print("Computed similarity scores for latent embeddings.")


# ########################################################################### #
# ######################### Param Deviation Analysis ######################## #
# ########################################################################### #
# List of parameter prefixes
prefixes = ('EGFR', 'ERK', 'ERBB2', 'MEK', 'iMEK', 'iEGFR')
# Compute ratios between deviations and medians
param_cols = [col for col in param_df.columns if col.startswith(prefixes)]
# Choose whether to plot the average of all multistarts or only the top 10 with respect to training performance (rmse_train)
plot_top_n_train = True

## Plot spread of parameter deviations across multistarts for validation cell-lines
plot_val_param_dev_spread(
    top_n_param_dev_df_train if plot_top_n_train else param_dev_df,
    param_cols,
    reg_param,
    ["l1reg_inflater_output", "median_reg"],  # TODO any better way of dynamically defining this?
    fig_dir / conf.model / conf.data / f"param_dev_boxplot_val_only_{reg_param}.pdf"
)

for context in CONTEXT_SET:
    for plot_label, parameter_dataframe in zip(
            ["param", "param_dev"],
            [
                top_n_param_df_train if plot_top_n_train else param_df,
                top_n_param_dev_df_train if plot_top_n_train else param_dev_df
            ]
    ):
        # Heatmaps VS Regularisation strength
        # Subset to context and compute the median over all jobs
        group_cols = [col for col in parameter_dataframe.columns if
                      (not col.startswith(prefixes)) and (col != "job")]
        plot_df = (
            parameter_dataframe[parameter_dataframe.context == context].groupby(group_cols)[param_cols]
            .agg("median")  # CHANGED FROM MEAN TO MEDIAN
            .reset_index()
        )

        for val_only, val_label in zip([True, False], ["val_only", "all"]):
            filtered_df = plot_df if not val_only else plot_df[
                plot_df.cell_line.isin(hardest_cell_lines)
            ]

            plot_parameter_heatmaps(
                filtered_df,
                param_cols,
                group_cols,
                reg_param,
                plot_label,
                fig_dir / conf.model / conf.data / f"{conf.model}.{conf.data}.{context}.{plot_label}.{val_label}.pdf",
                val_only,
            )

        # Adjust layout for better spacing
        g.set_titles(col_template="{col_name}")
        g.tight_layout()
        # plt.legend()
        plt.savefig(
            fig_dir / conf.model / conf.data / f"{conf.model}.{conf.data}.{context}.param_dev_{val_cell_line}.pdf")
        plt.show()



# ########################################################################## #
# ######################### Statistical Test Plots ######################### #
# ########################################################################## #
if RETURN_STAT_TESTS:
    # n_hidden pairwise comparisons:
    # subset to where n_hidden is null (n_hidden1 and n_hidden2 will be not null)
    n_hidden_pairwise_heatmap(
        dataframe=stat_test_res_df[
            stat_test_res_df.n_hidden.isnull()
        ],
        conf=conf
    )
    # Volcano plot of hyperparameter significance in improving (reducing) rmse_val:
    # subset to where n_hidden1 is null (for pairwise n_hidden comparisons above)
    volcano_hyperparameter_significance(
        dataframe=stat_test_res_df[
            stat_test_res_df.n_hidden1.isnull()
        ],
        conf=conf
    )

# ########################################################################### #
# ########################## Time-varying Response ########################## #
# ########################################################################### #
# Load measurement and observable dataframes
df_meas, df_obs = get_measurements_and_obervables(conf)

# Initialise latent embeddings, parameter medians and parameter deviations list of dataframes
latent_embeddings_dfs, param_deviations_dfs, params_dfs = [], [], []

# Setup features_test for regressors - need to ensure all contexts and splits have the same number of features/columns
features_test = {
    context: None
    for context in CONTEXT_SET
}

# TODO: overall structure is good, but I am wrongly fetching evaluation files - those contain residuals, whereas
#  I need to fetch the actual simulation files to plot the time-varying response!!!
# TODO: need to loop through SPLITS and get samples + the best configuration does not specify a single job!
# We need to fetch all of them and plot mean ± std
for dataset, context, split in itt.product(
        [
            "train",
            "test",
        ],
        CONTEXT_SET,
        sorted(list(SPLITS)) # ensure processing from 0of5 to 4of5
):
    # Load petab base files and training/validation split
    conf.samples = split
    petab_base_files = load_petab_base_files(conf)
    samples_dict = {
        "train": training_samples(Wildcards(conf.data, split)),
        "test": test_samples(Wildcards(conf.data, split)),
    }

    # Get per-sample simulation
    problem = CytofProblem(conf.model)
    per_sample_sim_dfs = []
    for sample in samples_dict[dataset]:
        output = process_per_sample_pretrain(
            sample,
            problem,
            conf,
            pretrain_dir / conf.model / conf.data,
            petab_base_files
        )
        if output is None:
            # file not found
            continue
        _, simulation_df = output
        per_sample_sim_dfs.append(simulation_df)
    per_sample_sim_df = pd.concat(per_sample_sim_dfs)

    # Get avg_model simulation
    avg_model_sim_df = simulate_avg_model(
        conf,
        pretrain_dir / conf.model / conf.data,
        petab_base_files,
        dataset
    )
    # Process and subset measurement dataset
    avg_model_sim_df, df_meas_subset = process_avg_model_simulation(
        avg_model_sim_df,
        df_meas,
        dataset,
        samples_dict
    )

    # Get best-regressor simulation
    regressor_mode = best_regressors[dataset][context]
    trained_pipeline_file = REGR_TRAINED_PIPELINE.format(
        model=conf.model,
        data=conf.data,
        samples=conf.samples,
        mode=regressor_mode,
        context=context,
    )
    features_train_file = REGR_FEATURES_TRAIN.format(
        model=conf.model,
        data=conf.data,
        samples=conf.samples,
        mode=regressor_mode,
        context=context,
    )
    # load using joblib
    trained_pipeline = load(trained_pipeline_file)
    features_train = load(features_train_file)

    # Load input and output data and fit regression pipeline to get simulation
    input_data, _ = load_data(
        contextualization=context,
        samples=samples_dict[dataset],
        features=features_train if dataset == "test" else None,
        measurement_table=petab_base_files["measurement_table"],
        observable_table=petab_base_files["observable_table"],
        features_filepath=get_features_filepaths(
            replace(conf, context=context, features="all"), FEATURES_OUTFILE, FEATURES_PIPELINE
        )[0] if context == "MOSA" else None,
    )
    output_data, test_columns = load_data(
        contextualization="cytof_dynamic",
        samples=samples_dict[dataset] if context != "MOSA" else input_data.index,  # restrict samples for MOSA (not all cell-lines available)
        features=features_test[context] if dataset == "test" else None,
        measurement_table=petab_base_files["measurement_table"],
        observable_table=petab_base_files["observable_table"],
    )
    # Get features_test in "train" to later use with "test"
    if features_test[context] is None:
        features_test[context] = test_columns

    best_regressor_sim_df = pd.DataFrame(
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
    # TODO @GiacomoFabrini - both avg_model_sim_df and per_sample_sim_df have 698 rows, but
    #  best_regressor_sim_df only has 608 - what are those 90 rows missing from the latter?
    #  Is this related to the missing/inconsistent timepoints in some samples?

    # BEST DMM
    # Overall (across jobs and splits)
    # Generate confs for all jobs -- include info on split
    overall_best_confs = [
        replace(get_dmm_conf(conf, best_hyperparam_dmm, dataset, context), job=job, samples=split)
        for job in JOBS
    ]
    rmse_best_confs = best_hyperparam_dmm[dataset][context]["rmse list"]
    # Compute features once (same across all jobs) - depend on SPLIT
    input_features = load_and_transform_features(overall_best_confs[0], dataset)
    overall_best_dmm_sim_dfs = []

    # temp_latent_embeddings, temp_parameter_medians, temp_parameter_deviations = [], [], []
    for job, overall_best_conf, rmse_conf in zip(JOBS, overall_best_confs, rmse_best_confs):
        # simulation errors might result in missing files -> skip
        try:
            models, obj = load_model_and_obj(
                conf=overall_best_conf,
                petab_base_files=petab_base_files,
                dataset=dataset,
                num_ensemble_members=1,   # use the best ensemble member by default
            )
        except FileNotFoundError:
            continue

        # ############## Latent embeddings, parameter deviations, parameters ############## #
        # Get latent embeddings, parameter deviations and full parameters
        # partial_le_df, partial_pd_df, partial_p_df = get_embedding_and_params_df(
        #     dmm_model=models[0],
        #     input_features=input_features,
        #     context=context,
        #     split=split,
        #     dataset=dataset,
        #     job=job,
        # )
        # # Append to growing list of dataframes
        # latent_embeddings_dfs.append(partial_le_df.assign(rmse=rmse_conf))
        # param_deviations_dfs.append(partial_pd_df.assign(rmse=rmse_conf))
        # params_dfs.append(partial_p_df.assign(rmse=rmse_conf))

        # Simulate and append to growing pd.DataFrame list
        overall_best_dmm_sim_dfs.append(
            simulate_dmm(
                model=models[0],
                input_features=input_features,
                obj=obj,
                petab_problem=models[0].petab_importer.petab_problem,
                jit_fn=False,
            ).assign(job=job)
        )

    # Concatenate simulations for time-varying response plot
    overall_best_dmm_sim_df = pd.concat(overall_best_dmm_sim_dfs)
    del overall_best_dmm_sim_dfs
    # Add identifier column (to keep replicate datapoints)
    overall_best_dmm_sim_df["unique_id"] = list(range(int(len(overall_best_dmm_sim_df) / len(JOBS)))) * int(len(JOBS))
    # Group by all necessary columns, compute mean for "simulation" column and drop unnecessary columns
    overall_best_dmm_sim_df = overall_best_dmm_sim_df.groupby(
        ["observableId", "preequilibrationConditionId",
         "time", "noiseParameters", "simulationConditionId",
         "measurementType", "observableParameters", "unique_id"]
    ).mean().reset_index().drop(
        columns=["job", "unique_id"]
    )

    # Single-shot (single split, single job) -- NOT IN USE
    # absolute_best_conf = get_dmm_conf(conf, absolute_best_dmm, dataset, context)
    # absolute_best_dmm_model, obj = load_model_and_obj(
    #     absolute_best_conf,
    #     petab_base_files,
    #     dataset
    # )
    # absolute_best_dmm_sim_df = simulate_dmm(
    #     model=absolute_best_dmm_model,
    #     input_features=load_and_transform_features(absolute_best_conf, dataset),
    #     obj=obj,
    #     petab_problem=absolute_best_dmm_model.petab_importer.petab_problem,
    # )

    # Plot the time-varying response
    plot_cross_samples_multiple_simulations(
        measurement_df=df_meas_subset,
        simulation_dfs=[
            per_sample_sim_df,
            avg_model_sim_df,
            best_regressor_sim_df,
            overall_best_dmm_sim_df,
            # absolute_best_dmm_sim_df
        ],
        labels=[  # TODO @GiacomoFabrini need to find a way to use these to produce secondary legend (currently unused)
            "per_sample",
            "avg_model",
            "best_regressor",
            "best_dmm_overall",
            # "best_singleshot_dmm"
        ],
        linetypes=[
            (0, (1, 5)),  # similar to "loosely dotted" in matplotlib but with twice more frequent dots
            "dotted",
            "dashed",
            "solid",
            # "dashdot",
        ],  # TODO change this to something more appropriate
        linesizes=[
            1,
            1,
            1,
            1.25,  # slightly thicker lines for DMM models
            # 1.25,
        ],  # TODO change this to something more appropriate
        figdir=outdir / dataset,
        prefix="__".join(
            [
                dataset,
                split,
                context,
                "comparison_best",
            ]
        ),
    )

# TODO: MODIFY OR DELETE
# Concatenate and save in CSV format for record and further analysis
# for dfs, df_label in zip(
#     [latent_embeddings_dfs, param_deviations_dfs, params_dfs],
#     ["latent_embeddings", "parameter_deviations", "parameters_full"]
# ):
#     # Concatenate
#     df = pd.concat(dfs)
#     # Add subtype information, both PAM50 and rough Luminal/Basal
#     df['subtype_PAM50'] = df['cell_line'].map(
#         {
#             cell_line: subtype["PAM50"]
#             for cell_line, subtype in subtypes_tognetti.items()
#         }
#     )
#     df['subtype_Luminal/Basal'] = df['cell_line'].map(
#         {
#             cell_line: subtype["Luminal/Basal"]
#             for cell_line, subtype in subtypes_tognetti.items()
#         }
#     )
#     # Save
#     df.to_csv(
#         evaluations_dir
#         / f"{conf.model}"
#         / f"{conf.data}"
#         / f"{conf.model}.{conf.data}.{df_label}.csv"
#     )
#
#     if df_label == "latent_embeddings":
#         center = True
#         scale = True
#
#         pca_latent_embeddings(
#             le_df=df,
#             scale=scale,
#             center=center,
#             num_jobs_plot=10  # use top 10 performing multistarts (lowest RMSE)
#         )

print("Done.")  # TODO remove + TODO: consider moving all helper functions into separate scripts
