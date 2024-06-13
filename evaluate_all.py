import fire
import itertools as itt
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import wandb

from common import (
    Conf,
    CONTEXT_SET,
    default_attributes,
    evaluations_dir,
    EVALUATION_REFERENCE,
    EVALUATION_REGRESSOR,
    EVALUATION_TRAINING,
    FEATURES_PIPELINE,
    fig_dir,
    pretrain_dir,
    REGR_FEATURES_TRAIN,
    REGR_TRAINED_PIPELINE,
    training_samples,
    test_samples,
    Wildcards,
)
from cytof.problem import CytofProblem
from dataclasses import replace
from dmm.analysis import plot_loss_vs_regularization, simulate_dmm
from dmm.feature_selection import load_data
from dmm.initialisation import get_features, pca_transform_features
from dmm.plotting import plot_cross_samples_multiple_simulations
from evaluation_plotting import (n_hidden_pairwise_heatmap,
                                 volcano_hyperparameter_significance)
from evaluation_utils import (get_measurements_and_obervables,
                              load_model_and_obj,
                              simulate_avg_model,
                              process_avg_model_simulation,
                              process_per_sample_pretrain)
from joblib import load
from stat_test import statistical_significance_test
from training_configuration import (
    CONTEXTS_FEATURES, FEATURES_TRANSFORM, SPLITS, PRETRAIN,
    LATENT_DIMS, NETWORK_LAYOUT, USE_BIAS, NN_INIT_FN,
    RECONSTRUCT, ACTIVATION_FNS, OPTIMISERS,
    ORTH_REG_STRATEGIES, ALPHAS, BETAS, GAMMAS, DELTAS, EPSILONS, ZETAS,
    MAX_LEARNING_RATES, LEARNING_RATE_SPANS, LEARNING_RATE_DECAYS, WARMUP_FCTS, OPT_STEPS, OPT_MULT,
    LINEAR_SCHEDULE, USE_EARLY_STOP, DROP_REG_POST_PRETRAIN, RETURN_STAT_TESTS, SPARSITY_THRESHOLD
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
    dmm_conf = Conf(model=conf.model, data=conf.data)
    for key, value in dmm_params[dataset][context].items():
        if hasattr(dmm_conf, key) and key not in ["model", "data"]:
            if key in ["n_hidden", "opt_steps", "opt_mult", "job"]:
                value = int(value)
            setattr(dmm_conf, key, value)
    return dmm_conf


def load_and_transform_features(
        conf: Conf,
        dataset: str
) -> np.ndarray:
    features = get_features(
        conf=conf,
        datasets=['train', 'val']
    )
    if conf.features_transform == "pca":
        # Load pre-trained pipeline if it exists
        pipeline_file = FEATURES_PIPELINE.format_map(conf.__dict__)
        if os.path.exists(pipeline_file):
            pipeline = load(FEATURES_PIPELINE.format_map(conf.__dict__))
        else:
            pipeline = None
        features = pca_transform_features(features, conf, pipeline)
    if dataset == 'train':
        features_dataset = 'train'
    elif dataset == 'test':
        features_dataset = 'val'
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
    # Define aggregation groups for DMM
    gbs_dmm = ["dataset", "ref"] + default_attributes

    data_dmm = pd.DataFrame(
        [
            dict(
                zip(gbs_dmm, group),
                rmse=np.sqrt(np.square(group_df["res"]).mean()),  # mean RMSE across all jobs (not best result)
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
    best_hyperparam_dmm = get_best_performer_across_jobs(
        dataframe=data[data.ref == 'DMM'],
        group_attributes=['dataset', 'context', 'features', 'pretrain'],
        hyperparam_attributes=[x for x in default_attributes if x not in ['context', 'features', 'pretrain']],
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
        "train",  # TODO @GiacomoFabrini: re-enable once hyperparam grid is narrower
        "test"
    ]:
        print(f'Starting to concatenate training evaluations for {samples}, {dataset}')
        # training
        training = pd.concat(
            pd.read_csv(efile, index_col=0)
            for (
                (ctxt, features), features_transform, pretrain, n_hidden,
                reconstruct, activation_fn_name, optimiser,
                (encoder_layer_sizes, inflater_layer_sizes, linear_benchmark),
                use_layer_bias, nn_init_fn, orth_reg_strategy,
                alpha, beta, gamma, delta, epsilon, zeta,
                max_lrate, lrate_span, lrate_decay, warmup_fct, opt_steps, opt_mult,
                use_simple_linear_schedule, use_early_stopping, drop_reg_after_pretrain, sparsity_threshold,
                job,
            ) in itt.product(
                CONTEXTS_FEATURES, FEATURES_TRANSFORM, PRETRAIN, LATENT_DIMS,
                RECONSTRUCT, ACTIVATION_FNS, OPTIMISERS,
                NETWORK_LAYOUT,
                USE_BIAS, NN_INIT_FN, ORTH_REG_STRATEGIES,
                ALPHAS, BETAS, GAMMAS, DELTAS, EPSILONS, ZETAS,
                MAX_LEARNING_RATES, LEARNING_RATE_SPANS, LEARNING_RATE_DECAYS, WARMUP_FCTS, OPT_STEPS, OPT_MULT,
                LINEAR_SCHEDULE, USE_EARLY_STOP, DROP_REG_POST_PRETRAIN, SPARSITY_THRESHOLD,
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
                            features_transform=features_transform,
                            samples=samples,
                            pretrain=pretrain,
                            n_hidden=n_hidden,
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
                            sparsity_threshold=sparsity_threshold,
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
# Load measurement and observable dataframes
df_meas, df_obs = get_measurements_and_obervables(conf)

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
        SPLITS
):
    # Load petab base files
    conf.samples = split
    petab_base_files = load_petab_base_files(conf, reweight=True)
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

    # Simulate best regressor
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
    output_data, _ = load_data(
        contextualization="cytof_dynamic",
        samples=samples_dict[dataset],
        features=None,
        measurement_table=petab_base_files["measurement_table"],
        observable_table=petab_base_files["observable_table"],
    )
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
    # Generate confs for all jobs
    overall_best_confs = [
        replace(get_dmm_conf(conf, best_hyperparam_dmm, dataset, context), job=job)
        for job in JOBS
    ]
    overall_best_dmm_sim_dfs = []
    for job, overall_best_conf in zip(JOBS, overall_best_confs):
        model, obj = load_model_and_obj(
            overall_best_conf,
            petab_base_files,
            dataset,
        )
        input_features = load_and_transform_features(overall_best_conf, dataset)
        overall_best_dmm_sim_dfs.append(simulate_dmm(
            model=model,
            input_features=input_features,
            obj=obj,
            petab_problem=model.petab_importer.petab_problem,
        ).assign(job=job))
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

    # Single-shot (single split, single job)
    absolute_best_conf = get_dmm_conf(conf, absolute_best_dmm, dataset, context)
    absolute_best_dmm_model, obj = load_model_and_obj(
        absolute_best_conf,
        petab_base_files,
        dataset
    )
    absolute_best_dmm_sim_df = simulate_dmm(
        model=absolute_best_dmm_model,
        input_features=load_and_transform_features(absolute_best_conf, dataset),
        obj=obj,
        petab_problem=absolute_best_dmm_model.petab_importer.petab_problem,
    )

    # Plot the time-varying response
    plot_cross_samples_multiple_simulations(
        measurement_df=df_meas_subset,
        simulation_dfs=[
            per_sample_sim_df,
            avg_model_sim_df,
            best_regressor_sim_df,
            overall_best_dmm_sim_df,
            absolute_best_dmm_sim_df
        ],
        labels=[  # TODO @GiacomoFabrini need to find a way to use these to produce secondary legend (currently unused)
            "per_sample",
            "avg_model",
            "best_regressor",
            "best_dmm_overall",
            "best_singleshot_dmm"
        ],
        linetypes=[
            (0, (1, 5)),  # similar to "loosely dotted" in matplotlib but with twice more frequent dots
            "dotted",
            "dashed",
            "solid",
            "dashdot",
        ],  # TODO change this to something more appropriate
        linesizes=[
            1,
            1,
            1,
            1.25,  # slightly thicker lines for DMM models
            1.25,
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
