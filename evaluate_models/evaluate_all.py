import fire
import itertools as itt
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import petab
import wandb

from common import (
    Conf,
    EVALUATION_REFERENCE,
    EVALUATION_REGRESSOR,
    EVALUATION_TRAINING,
    fig_dir,
    evaluations_dir,
    CONTEXT_SET,
    training_samples,
    test_samples,
    Wildcards,
)
from dmm.analysis import plot_loss_vs_regularization
from training_configuration import (
    CONTEXTS_FEATURES, SPLITS, PRETRAIN,
    LATENT_DIMS, NETWORK_LAYOUT, USE_BIAS, NN_INIT_FN,
    RECONSTRUCT, ACTIVATION_FNS, OPTIMISERS,
    ORTH_REG_STRATEGIES, ALPHAS, BETAS, GAMMAS, DELTAS, EPSILONS, ZETAS,
    MAX_LEARNING_RATES, LEARNING_RATE_SPANS, LEARNING_RATE_DECAYS, WARMUP_FCTS, OPT_STEPS, OPT_MULT,
    LINEAR_SCHEDULE, USE_EARLY_STOP, DROP_REG_POST_PRETRAIN, RETURN_STAT_TESTS
)
from dmm.plotting import plot_cross_samples_multiple_simulations
from evaluate_models.evaluation_plotting import (n_hidden_pairwise_heatmap,
                                                 volcano_hyperparameter_significance)
from evaluate_models.evaluation_utils import get_measurements_and_obervables, process_sim_df
from evaluate_models.stat_test import statistical_significance_test
from typing import List

REGRESSION_MODES = ["linreg", "lasso", "elasticnet"]


def convert_attributes_to_int(dictionary, attributes):
    """
    Convert the values of specified attributes in the dictionary to integers if possible.

    Parameters:
    dictionary (dict): The dictionary containing the attributes.
    attributes (list): The list of attributes whose values need to be converted to integers.

    Returns:
    dict: The dictionary with specified attributes' values converted to integers.
    """
    for attr in attributes:
        if attr in dictionary:
            try:
                dictionary[attr] = int(dictionary[attr])
            except (ValueError, TypeError):
                # If conversion fails, leave the value as is
                pass
    return dictionary


def get_best_performer_across_jobs(
        dataframe: pd.DataFrame,
        group_attributes: List,
        hyperparam_attributes: List,
        mode: bool,
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
        temp_df = dataframe.reset_index().groupby(
            group_attributes + hyperparam_attributes
        ).agg({target_attribute: ['mean', 'std']})
        temp_df = temp_df.reset_index()
        temp_df.columns = [
            ' '.join(col).strip()
            for col in temp_df.columns.values
        ]
        min_rmse_indices = temp_df.groupby(
            by=group_attributes
        )[target_attribute + ' mean'].idxmin()
        result = temp_df.loc[min_rmse_indices]
        result_dict = {
            dataset: {
                context: result[(result.dataset == dataset) & (result.context == context)].iloc[0].to_dict()
                for context in result[result.dataset == dataset].context.unique()
            }
            for dataset in result.dataset.unique()
        }
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


def aggregate_and_log(df: pd.DataFrame, return_stat_tests: bool):
    # TODO @GiacomoFabrini - define this list somewhere and import it!
    # Define aggregation groups for DMM
    gbs = [
        "dataset",
        "context", "features", "samples", "pretrain",
        "ref",
        "n_hidden",
        "encoder_layer_sizes", "inflater_layer_sizes", "linear_benchmark",
        "use_layer_bias", "nn_init_fn", "reconstruct", "activation_fn_name", "optimiser",
        "orth_reg_strategy",
        "l1reg_inflate", "oreg_inflate", "l1reg_encode", "oreg_encode", "recon_loss", "symm_reg",
        "max_lrate", "lrate_span", "lrate_decay", "warmup_fct", "opt_steps", "opt_mult",
        "use_simple_linear_schedule", "use_early_stopping", "drop_reg_after_pretrain",
        "job",
    ]

    data_dmm = pd.DataFrame(
        [
            dict(
                zip(gbs, group),
                rmse=np.sqrt(np.square(group_df["res"]).mean()),  # mean RMSE across all jobs (not best result)
            )
            for group, group_df in df.groupby(gbs)
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
        pivot_data['rmse_list'] = pivot_data.apply(lambda row: np.array([row[col] for col in multiindex_rmse_cols]), axis=1)
        # Add the newly created column to the list of columns to be kept (cols)
        cols += ['rmse_list']
        # Subset the pivot table and reduce MultiIndex back to single-level index
        data_stat_tests = pivot_data[cols]
        data_stat_tests.columns = data_stat_tests.columns.droplevel(level=1)
        print("DataFrame for statistical testing is now ready.")

        stat_test_res_df = statistical_significance_test(data_stat_tests)

    # Get best performing hyperparameter set across jobs for each dataset/context/ref combination
    best_hyperparam_dmm = get_best_performer_across_jobs(
        dataframe=data[data.ref == 'DMM'],
        group_attributes=['dataset', 'context', 'features', 'pretrain'],
        hyperparam_attributes=[
            "n_hidden",
            "encoder_layer_sizes", "inflater_layer_sizes", "linear_benchmark",
            "use_layer_bias", "nn_init_fn", "reconstruct", "activation_fn_name", "optimiser",
            "orth_reg_strategy",
            "l1reg_inflate", "oreg_inflate", "l1reg_encode", "oreg_encode", "recon_loss", "symm_reg",
            "max_lrate", "lrate_span", "lrate_decay", "warmup_fct", "opt_steps", "opt_mult",
            "use_simple_linear_schedule", "use_early_stopping", "drop_reg_after_pretrain",
        ],
        mode='DMM',
        target_attribute='rmse',
    )
    best_regressors = get_best_performer_across_jobs(
        dataframe=data[data.ref.isin(REGRESSION_MODES)],
        group_attributes=['dataset', 'context'],
        hyperparam_attributes=[],
        mode='regressor',
        target_attribute='rmse',
    )
    # Get absolute best performing hyperparameter set (single job) for each dataset/context/ref combination
    absolute_best_dmm = get_absolute_best_performer(
        dataframe=data[data.ref == 'DMM'],
        group_attributes=['dataset', 'context', 'ref'],
        target_attribute='rmse',
    )

    # Log via W&B
    wandb.init(
        project=f"DeepMechanisticModels.{conf.data}.{conf.model}",
        config={
            **conf.__dict__,
        },
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
        evaluation_df.to_csv(
            evaluations_dir
            / f"{conf.model}"
            / f"{conf.data}"
            / f"{conf.model}.{conf.data}.{evaluation_tag}.csv"
        )

        # Instantiate artifact
        evaluation_artifact = wandb.Artifact(
            name=f"{evaluation_tag}_{conf.model}_{conf.data}",
            description=evaluation_tag,
            type="evaluation",
        )
        # Add and log artifact
        evaluation_artifact.add(wandb.Table(dataframe=data), f"{evaluation_tag}.csv")
        wandb.log_artifact(evaluation_artifact)

    # Close W&B session
    wandb.finish()

    if return_stat_tests:
        return data, stat_test_res_df, best_hyperparam_dmm, best_regressors, absolute_best_dmm
    else:
        return data, best_hyperparam_dmm, best_regressors, absolute_best_dmm


conf = fire.Fire(Conf)

outdir = fig_dir / conf.model / conf.data

# METHODS = ("pca embedding", "end-to-end")  # not used at the moment

JOBS = tuple([i for i in range(2)])  # need to change this - NO HARDCODING - TODO change back to 10
dfs = []
for samples in SPLITS:
    for dataset in [
        # "train",  # TODO @GiacomoFabrini: re-enable once hyperparam grid is narrower
        "test"
    ]:
        print(f'Starting to concatenate training evaluations for {samples}, {dataset}')
        # training
        training = pd.concat(
            pd.read_csv(efile, index_col=0)
            for ((ctxt, features), pretrain, ldim,
                 reconstruct, activation_fn_name, optimiser,
                 (encoder_layer_sizes, inflater_layer_sizes, linear_benchmark),
                 use_layer_bias, nn_init_fn,
                 orth_reg_strategy, alpha, beta, gamma, delta, epsilon, zeta,
                 max_lrate, lrate_span, lrate_decay, warmup_fct, opt_steps, opt_mult,
                 use_simple_linear_schedule, use_early_stopping, drop_reg_after_pretrain, job,
                 ) in itt.product(
                CONTEXTS_FEATURES,
                PRETRAIN,
                LATENT_DIMS,
                RECONSTRUCT,
                ACTIVATION_FNS,
                OPTIMISERS,
                NETWORK_LAYOUT,
                USE_BIAS,
                NN_INIT_FN,
                ORTH_REG_STRATEGIES,
                ALPHAS,
                BETAS,
                GAMMAS,
                DELTAS,
                EPSILONS,
                ZETAS,
                MAX_LEARNING_RATES,
                LEARNING_RATE_SPANS,
                LEARNING_RATE_DECAYS,
                WARMUP_FCTS,
                OPT_STEPS,
                OPT_MULT,
                LINEAR_SCHEDULE,
                USE_EARLY_STOP,
                DROP_REG_POST_PRETRAIN,
                JOBS,
            )
            if os.path.exists(
                efile := EVALUATION_TRAINING.format(
                    **{
                        **conf.__dict__,
                        **dict(
                            dataset=dataset,
                            context=ctxt,
                            features=features,
                            samples=samples,
                            pretrain=pretrain,
                            n_hidden=ldim,
                            encoder_layer_sizes=encoder_layer_sizes,
                            inflater_layer_sizes=inflater_layer_sizes,
                            linear_benchmark=linear_benchmark,
                            use_layer_bias=use_layer_bias,
                            nn_init_fn=nn_init_fn,
                            reconstruct=reconstruct,
                            activation_fn_name=activation_fn_name,
                            optimiser=optimiser,
                            orth_reg_strategy=orth_reg_strategy,
                            l1reg_inflate=alpha,
                            oreg_inflate=beta,
                            l1reg_encode=gamma,
                            oreg_encode=delta,
                            recon_loss=epsilon,
                            symm_reg=zeta,
                            max_lrate=max_lrate,
                            lrate_span=lrate_span,
                            lrate_decay=lrate_decay,
                            warmup_fct=warmup_fct,
                            opt_steps=opt_steps,
                            opt_mult=opt_mult,
                            use_simple_linear_schedule=use_simple_linear_schedule,
                            use_early_stopping=use_early_stopping,
                            drop_reg_after_pretrain=drop_reg_after_pretrain,
                            job=job,
                        ),
                    },
                )
            )
        )
        print(f'Finished concatenating training evaluations for {samples}, {dataset}')

        # Loss vs regularization plot
        print(f'Starting to plot loss_vs_regularization for {samples}, {dataset}')
        plot_loss_vs_regularization(training)
        plt.savefig(outdir / f"{samples}_evaluate_training_{dataset}.pdf")
        print(f'Saved loss_vs_regularization plot for {samples}, {dataset}')

        # Add necessary attributes to training DataFrame
        training["ref"] = "DMM"  # previously "meth"
        training["dataset"] = dataset
        training["samples"] = samples

        # # average
        # avg = pd.read_csv(
        #     EVALUATION_REFERENCE.format(
        #         **{
        #             **conf.__dict__,
        #             **dict(
        #                 samples=samples,
        #                 dataset=dataset,
        #             ),
        #         },
        #         mode="average",
        #     ),
        #     index_col=0,
        # )
        # avg["ref"] = "avg"

        # model average
        print(f'Processing avg_model for {samples}, {dataset}')
        avg_model = pd.read_csv(
            EVALUATION_REFERENCE.format(
                **{
                    **conf.__dict__,
                    **dict(
                        samples=samples,
                        dataset=dataset,
                    ),
                },
                mode="avg_model",
            ),
            index_col=0,
        )
        avg_model["ref"] = "avg_model"
        print(f'Finished processing avg_model for {samples}, {dataset}')

        # per sample
        print(f'Processing per_sample model for {samples}, {dataset}')
        ps = pd.read_csv(
            EVALUATION_REFERENCE.format(
                **{
                    **conf.__dict__,
                    **dict(
                        samples=samples,
                        dataset=dataset,
                    ),
                },
                mode="per_sample",
            ),
            index_col=0,
        )
        ps["ref"] = "sample"
        print(f'Finished processing per_sample model for {samples}, {dataset}')

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
            ).assign(ref=mode)
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
                avg_ps_df["context"] = context
                # avg_ps_df["type"] = method
                avg_ps_dfs.append(avg_ps_df)
                # Once appended, this can be deleted
                del avg_ps_df

        # regression baselines already have context
        for _, rdf in regressor_dfs.items():
            avg_ps_df = rdf.copy()
            # avg_ps_df["type"] = method
            avg_ps_dfs.append(avg_ps_df)
            # Once appended, this can be deleted
            del avg_ps_df
        print(f"Finished processing reference models for {samples}, {dataset}")

        # dfd = pd.concat([training, pretraining])
        dfd = pd.concat([training.convert_dtypes(), *avg_ps_dfs])
        # Deleting DataFrames once concatenated into dfd
        del training, avg_ps_dfs, rdf
        dfd["dataset"] = dataset
        dfd["samples"] = samples
        dfs.append(dfd)
        # Deleting dfd once appended to dfs
        del dfd
        print(f"Finished concatenating training and reference models for {samples}, {dataset}")

df = pd.concat(dfs).reset_index()
# Now that dfs have been concatenated into df, delete them
del dfs

# Aggregate data into DataFrames for plotting, save the results as CSVs and log them
# as W&B artifacts
aggregated_results = aggregate_and_log(df, RETURN_STAT_TESTS)
if RETURN_STAT_TESTS:
    data, stat_test_res_df, best_hyperparam_dmm, best_regressors, absolute_best_dmm = aggregated_results
else:
    data, best_hyperparam_dmm, best_regressors, absolute_best_dmm = aggregated_results

# ########################################################################### #
# ############################ Performance Plots ############################ #
# ########################################################################### #

# group_plots(
#     dataframe=data,
#     conf=conf
# )
#
# performance_barplot(
#     dataframe=data,
#     conf=conf
# )

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
# Fetch measurement dataframe
df_meas, df_obs = get_measurements_and_obervables(conf)
# TODO: overall structure is good, but I am wrongly fetching evaluation files - those contain residuals, whereas
#  I need to fetch the actual simulation files to plot the time-varying response!!!
# TODO: need to loop through SPLITS and get samples + the best configuration does not specify a single job!
# We need to fetch all of them and plot mean ± std
for dataset, context, split in itt.product(
    [
        # "train",
        "test",
    ],
    CONTEXT_SET,
    SPLITS
):
    samples_dict = {
        "train": training_samples(Wildcards(conf.data, split)),
        "test": test_samples(Wildcards(conf.data, split)),
    }
    # Subset measurement dataframe
    df_meas_subset = df_meas[
        df_meas[petab.PREEQUILIBRATION_CONDITION_ID].isin(samples_dict[dataset])
    ]

    # Fetch per-sample pretraining
    per_sample_df = pd.read_csv(
        EVALUATION_REFERENCE.format(
            **{
                **conf.__dict__,
                **dict(
                    samples=split,
                    dataset=dataset,
                ),
            },
            mode="per_sample",
        ),
        index_col=0,
    )
    per_sample_df = process_sim_df(per_sample_df)

    # Fetch best regressor
    best_regressor_sim_df = regressor_dfs[best_regressors[dataset][context]].copy()
    best_regressor_sim_df = process_sim_df(best_regressor_sim_df)

    # Fetch the training evaluation files for the best performing DMM (across jobs and splits)
    # training_efiles = [
    #     EVALUATION_TRAINING.format(
    #         **{
    #             **convert_attributes_to_int(
    #                 best_hyperparam_dmm[dataset][context],
    #                 ["n_hidden", "opt_steps", "opt_mult", "job"]
    #             ),
    #             **dict(
    #                 model=conf.model,
    #                 data=conf.data,
    #                 job=job,
    #                 samples=samples
    #             ),
    #         },
    #     ) for job in JOBS for samples in SPLITS
    # ]
    # best_dmm_sim_dfs = [
    #     pd.read_csv(efile.replace(" ", ""), index_col=0) for efile in training_efiles
    # ]
    # best_dmm_overall_sim_df =

    # Fetch the training evaluation file for the best performing DMM (single job and SPLIT)
    absolute_best_dmm_sim_df = pd.read_csv(
        EVALUATION_TRAINING.format(
            **{
                **convert_attributes_to_int(
                    absolute_best_dmm[dataset][context],
                    ["n_hidden", "opt_steps", "opt_mult", "job"]
                ),
                **dict(
                    model=conf.model,
                    data=conf.data,
                ),
            },
        ).replace(" ", ""),
        index_col=0
    )
    absolute_best_dmm_sim_df = process_sim_df(absolute_best_dmm_sim_df)

    # Plot the time-varying response
    plot_cross_samples_multiple_simulations(
        measurement_df=df_meas_subset,
        simulation_dfs=[
            per_sample_df,
            best_regressor_sim_df,
            # best_dmm_overall_sim_df,
            absolute_best_dmm_sim_df
        ],
        labels=[
            "per_sample",
            "best_regressor",
            # "best_dmm_overall",
            "best_singleshot_dmm"
        ],
        linetypes=[
            "dotted",
            "dashed",
            # "solid",
            "dashdot",
        ],  # TODO change this to something more appropriate
        linesizes=[
            1,
            1,
            # 2,
            2,
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
