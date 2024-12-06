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
    FEATURES_OUTFILE,
    FEATURES_PIPELINE,
    fig_dir,
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
from dmm.autoencoder import DeepMechanisticModel
from dmm.config_options import Conf
from dmm.feature_selection import load_data
from dmm.initialisation import get_features, pca_transform_features
from dmm.plotting import plot_cross_samples_multiple_simulations
from evaluation_plotting import (n_hidden_pairwise_heatmap, performance_barplot,
                                 volcano_hyperparameter_significance)
from evaluation_utils import (get_measurements_and_obervables,
                              load_model_and_obj,
                              simulate_avg_model,
                              process_avg_model_simulation,
                              process_per_sample_pretrain)
from generate_run_configs import generate_run_configs
from jax import vmap
from joblib import load
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from stat_test import statistical_significance_test
from training_configuration import (
    CONTEXTS_FEATURES, SPLITS, RETURN_STAT_TESTS, HP_RUN_MODE, REFINE_HPS
)
from typing import List, Tuple, Union
from util import load_petab_base_files


REGRESSION_MODES = ["linreg", "lasso", "elasticnet"]


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
            "cell_line": models[0].sample_name_list,
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
    scale: bool = True,
    center: bool = True,
    center_method: str = "mean",
    col_grouping: str = "subtype_Luminal/Basal",
    num_jobs_plot: int = 5,
):
    for context in le_df.context.unique():
        pca_dfs = []
        for samples, job in itt.product(
            le_df.samples.unique(),
            le_df.job.unique()
        ):
            # Filter to current context, samples and job
            sub_df = le_df[
                (le_df.context == context) & (le_df.samples == samples) & (le_df.job == job)
            ].copy()

            # Get latent embeddings
            les = sub_df[["L1", "L2"]].values
            # Center and scale if necessary
            if center:
                if center_method == "mean":
                    les = les - les.mean(axis=0)
                elif center_method == "median":
                    les = les - np.median(les, axis=0)
            if scale:
                les = StandardScaler().fit_transform(les)

            # Get 2D PCA to try and remove potential rotations between multistart embeddings
            pca = PCA(n_components=2)
            les_pca = pca.fit_transform(les)
            # Reassign to sub-DataFrame and append to context-specific list of DataFrames
            sub_df[["L1", "L2"]] = les_pca
            pca_dfs.append(sub_df)

        # Obtain latent embedding PCA DataFrame for given context, save it and produce scatterplot
        pca_df = pd.concat(pca_dfs)
        pca_df.to_csv(
            evaluations_dir
            / f"{conf.model}"
            / f"{conf.data}"
            / f"{conf.model}.{conf.data}.{context}.latent_embeddings_pca.csv"
        )
        top_jobs = df.sort_values('rmse').drop_duplicates(subset='job').nsmallest(num_jobs_plot, 'rmse')["job"].values
        plot_df = pca_df[pca_df.job.isin(top_jobs)]
        g = sns.FacetGrid(plot_df, col=col_grouping, row="samples", hue="job")
        g.map(plt.scatter, "L1", "L2")
        plt.legend()
        plt.show()


def get_dmm_conf(
        conf: Conf,
        dmm_params: dict,
        dataset: str,
        context: str,
) -> Conf:
    dmm_conf = Conf(model=conf.model, data=conf.data)
    for key, value in dmm_params[dataset][context].items():
        if hasattr(dmm_conf, key) and key not in ["model", "data"]:
            if key in [
                "n_hidden", "depth", "nn_structure_multiplier",
                "inflater_output_reg_epoch",
                "opt_steps", "opt_mult",
                "job"
            ]:
                value = int(value)  # cast the value to integer
            elif key in [
                "l1reg_encode", "oreg_encode",  # encoder
                "l1reg_inflate", "oreg_inflate",  # inflater
                "l1reg_inflater_output",  # inflater output
                "recon_loss", "symm_reg",  # decoder / reconstruction
                "median_reg",  # kinetic params median regularisation
                "opt_steps", "opt_mult", "momentum"  # parameters that can be pruned by generate_run_configs
            ] and value == 0:
                value = int(value)

            setattr(dmm_conf, key, value)  # assign
    return dmm_conf


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
    samples: [hyperparam_config for hyperparam_config in hyperparam_configs if hyperparam_config['samples'] == samples]
    for samples in SPLITS
}
dfs = []
for samples in sorted(list(SPLITS)):  # process from 0of5 to 4of5
    for dataset in [
        "train",
        "test"
    ]:
        print(f'Starting to concatenate training evaluations for {samples}, {dataset}')
        # training
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

        # average (not in use)
        # avg = process_reference(conf, samples, dataset, "average", "avg")

        # model average (avg_model)
        avg_model = process_reference(conf, samples, dataset, "avg_model", "avg_model")

        # per sample
        ps = process_reference(conf, samples, dataset, "per_sample", "sample")

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
                avg_ps_df = avg_ps_df.assign(context=context, samples=samples, dataset=dataset)
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
        # Deleting DataFrames once concatenated into dfd
        del training, avg_ps_dfs, rdf
        dfs.append(dfd)
        # Deleting dfd once appended to dfs
        del dfd
        print(f"Finished concatenating training and reference models for {samples}, {dataset}")

df = pd.concat(dfs).reset_index()
# Now that dfs have been concatenated into df, delete them
del dfs

# Aggregate data into DataFrames for plotting, save the results as CSVs and log them
# as W&B artifacts
num_best = 10
aggregated_results = aggregate_and_log(df=df, return_stat_tests=RETURN_STAT_TESTS, num_best=num_best)
# Removed absolute_best_dmm from aggregated results
if RETURN_STAT_TESTS:
    data, stat_test_res_df, top_n_dmm, best_hyperparam_dmm, best_regressors = aggregated_results
else:
    data, top_n_dmm, best_hyperparam_dmm, best_regressors = aggregated_results


# ########################################################################### #
# ################### Save train/test RMSE dataset in CSV ################### #
# ########################################################################### #
dmm_results = {
    dataset: data[(data.dataset == dataset) & (data.ref == 'DMM')].drop(columns=['ref', 'dataset'])
    for dataset in ['train', 'test']
}
merge_cols = data.columns.difference(['rmse', 'dataset', 'ref'])
unified_dmm_results = pd.merge(
    dmm_results['train'],
    dmm_results['test'],
    how="inner",
    on=list(merge_cols),
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
# ################ Most predictive hyperparams for RMSE_val ################# #
# ########################################################################### #
# TODO @GiacomoFabrini - clean this up!
# First drop any Infs or NaNs (incompatible with RandomForestRegressor)
unified_dmm_results = unified_dmm_results.replace([np.inf, -np.inf], np.nan)
unified_dmm_results = unified_dmm_results.dropna()
# Then train two RandomForests, one on train and the other on val scores (RMSE)
rfr_train, rfr_val = RandomForestRegressor(), RandomForestRegressor()
rmse_val_targets = unified_dmm_results.pop("rmse_test")
rmse_train_targets = unified_dmm_results.pop("rmse_train")
# Replace categorical variables with dummy one-hot-encodings
dummy_unified_dmm_results = pd.get_dummies(unified_dmm_results)
# Fit and get feature importances for top 10 features
rfr_train.fit(X=dummy_unified_dmm_results, y=rmse_train_targets)
rfr_val.fit(X=dummy_unified_dmm_results, y=rmse_val_targets)
results_dfs = {
    dataset: pd.DataFrame(
        {
            "importances":regressor.feature_importances_[regressor.feature_importances_.argsort()][::-1][:10],
            "features": dummy_unified_dmm_results.columns[regressor.feature_importances_.argsort()][::-1][:10],
        }
    )
    for dataset, regressor in zip(["train", "val"], [rfr_train, rfr_val])
}
top_features_train = dummy_unified_dmm_results.columns[rfr_train.feature_importances_.argsort()][::-1][:10]
top_features_val = dummy_unified_dmm_results.columns[rfr_val.feature_importances_.argsort()][::-1][:10]
importances_train = rfr_train.feature_importances_[rfr_train.feature_importances_.argsort()][::-1][:10]
importances_val = rfr_val.feature_importances_[rfr_val.feature_importances_.argsort()][::-1][:10]


# Rudimentary bar-subplot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ind, dataset in enumerate(["train", "val"]):
    plt.subplot(1, 2, ind+1)
    sns.barplot(data=results_dfs[dataset], x='features', y='importances')
    plt.title('Feature Importances')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.xticks(rotation=90)

plt.tight_layout()
plt.savefig(
    fig_dir / f"{conf.model}" / f"{conf.data}" / f"{conf.model}.{conf.data}.top_10_feature_importances.svg"
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
performance_barplot(
    dataframe=data,
    conf=conf
)


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
    )
    output_data, test_columns = load_data(
        contextualization="cytof_dynamic",
        samples=samples_dict[dataset],
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

    temp_latent_embeddings, temp_parameter_medians, temp_parameter_deviations = [], [], []
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
        partial_le_df, partial_pd_df, partial_p_df = get_embedding_and_params_df(
            dmm_model=models[0],
            input_features=input_features,
            context=context,
            split=split,
            dataset=dataset,
            job=job,
        )
        # Append to growing list of dataframes
        latent_embeddings_dfs.append(partial_le_df.assign(rmse=rmse_conf))
        param_deviations_dfs.append(partial_pd_df.assign(rmse=rmse_conf))
        params_dfs.append(partial_p_df.assign(rmse=rmse_conf))

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

# Concatenate and save in CSV format for record and further analysis
for dfs, df_label in zip(
    [latent_embeddings_dfs, param_deviations_dfs, params_dfs],
    ["latent_embeddings", "parameter_deviations", "parameters_full"]
):
    # Concatenate
    df = pd.concat(dfs)
    # Add subtype information, both PAM50 and rough Luminal/Basal
    df['subtype_PAM50'] = df['cell_line'].map(
        {
            cell_line: subtype["PAM50"]
            for cell_line, subtype in subtypes_tognetti.items()
        }
    )
    df['subtype_Luminal/Basal'] = df['cell_line'].map(
        {
            cell_line: subtype["Luminal/Basal"]
            for cell_line, subtype in subtypes_tognetti.items()
        }
    )
    # Save
    df.to_csv(
        evaluations_dir
        / f"{conf.model}"
        / f"{conf.data}"
        / f"{conf.model}.{conf.data}.{df_label}.csv"
    )

    if df_label == "latent_embeddings":
        center = True
        scale = True

        pca_latent_embeddings(
            le_df=df,
            scale=scale,
            center=center,
            num_jobs_plot=10  # use top 10 performing multistarts (lowest RMSE)
        )
