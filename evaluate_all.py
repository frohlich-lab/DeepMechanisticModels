import os

import fire
import numpy as np
import pandas as pd

# import subprocess
# import wandb
from common import (
    CONTEXT_SET,
    EVALUATION_EMBEDDING,
    EVALUATION_FULL_PARAMETERS,
    EVALUATION_PARAMETER_DEVIATIONS,
    EVALUATION_REFERENCE,
    EVALUATION_REGRESSOR,
    EVALUATION_TRAINING,
    REGRESSION_MODES,
    fig_dir,
    subtypes_tognetti,
)

# from dmm.autoencoder import DeepMechanisticModel
from dmm.config_options import Conf
from evaluation_utils import (
    aggregate_and_log,
)
from generate_run_configs import generate_run_configs
from training_configuration import (
    CONTEXTS_FEATURES,
    HP_RUN_MODE,
    REFINE_HPS,
    RETURN_STAT_TESTS,
    SPLITS,
)


def process_reference(
    conf: Conf, samples: str, dataset: str, mode: str, ref_name: str
) -> pd.DataFrame:
    print(f"Processing {mode} model for {samples}, {dataset}")
    ref = pd.read_csv(
        EVALUATION_REFERENCE.format(
            **{
                **conf.__dict__,
                "samples": samples,
                "dataset": dataset,
            },
            mode=mode,
        ),
        index_col=0,
    )
    ref["ref"] = ref_name
    print(f"Finished processing {mode} model for {samples}, {dataset}")
    return ref


conf = fire.Fire(Conf)
outdir = fig_dir / conf.model / conf.data
# METHODS = ("pca embedding", "end-to-end")  # not used at the moment
JOBS = tuple(i for i in range(conf.n_starts))

# Compute subtype dictionaries
subtypes_pam50, subtypes_lb = (
    {
        cl: subtypes_tognetti[cl][subtype_scheme]
        for cl in subtypes_tognetti.keys()
    }
    for subtype_scheme in ["PAM50", "Luminal/Basal"]
)
subtypes_hr = {
    cl: (
        "Positive"
        if subtypes_pam50[cl] in ["LA", "LB"]
        else "Negative"
        if subtypes_pam50[cl] in ["HER2", "Basal"]
        else "Unknown"
    )
    for cl in subtypes_pam50.keys()
}

subtypes_her2 = {
    cl: (
        "Negative"
        if subtypes_pam50[cl] in ["LA", "Basal"]
        else "Positive"
        if subtypes_pam50[cl] in ["LB", "HER2"]
        else "Unknown"
    )
    for cl in subtypes_pam50.keys()
}

# Compute run configurations and arrange by CV split
hyperparam_configs = generate_run_configs(
    n_starts=conf.n_starts,
    hp_run_mode=HP_RUN_MODE,
    refine_hps=REFINE_HPS,
)
hyperparam_configs = {
    samples: [
        hyperparam_config
        for hyperparam_config in hyperparam_configs
        if hyperparam_config["samples"] == samples
    ]
    for samples in SPLITS
}

# Load evaluations (DMMs, baselines, regressors), latent embeddings, parameters and parameter deviations
dfs, le_dfs, param_dev_dfs, param_dfs = [], [], [], []
for samples in sorted(SPLITS):
    for dataset in ["train", "val"]:
        # DMM evaluations
        training = pd.concat(
            pd.read_csv(efile, index_col=0)
            for hyperparam_configuration in hyperparam_configs[samples]
            if os.path.exists(
                efile := EVALUATION_TRAINING.format_map(
                    {
                        **hyperparam_configuration,
                        "model": conf.model,
                        "data": conf.data,
                        "dataset": dataset,
                        "samples": samples,
                    }
                )
            )
        ).assign(
            ref="DMM",  # previously called "meth"
            dataset=dataset,
        )
        print(
            f"Finished concatenating training evaluations for {samples}, {dataset}"
        )

        # Loss vs regularization plot -- DISABLED, not useful right now
        # print(f'Starting to plot loss_vs_regularization for {samples}, {dataset}')
        # plot_loss_vs_regularization(training)
        # plt.savefig(outdir / f"{samples}_evaluate_training_{dataset}.pdf")
        # plt.close()
        # print(f'Saved loss_vs_regularization plot for {samples}, {dataset}')

        # concatenate embeddings, parameter deviations and parameters
        temp_results = {}
        for result_type, filepath_format in zip(
            ["latent_embeddings", "parameter_deviations", "full_parameters"],
            [
                EVALUATION_EMBEDDING,
                EVALUATION_PARAMETER_DEVIATIONS,
                EVALUATION_FULL_PARAMETERS,
            ],
        ):
            temp_results[result_type] = pd.concat(
                pd.read_csv(efile, index_col=0).assign(
                    **hyperparam_configuration
                )
                for hyperparam_configuration in hyperparam_configs[samples]
                if os.path.exists(
                    efile := filepath_format.format_map(
                        {
                            **hyperparam_configuration,
                            "model": conf.model,
                            "data": conf.data,
                            "dataset": dataset,
                            "samples": samples,
                        }
                    )
                )
            )
        print(
            f"Finished concatenating embeddings, parameters and parameter deviations for {samples}, {dataset}"
        )

        # Get references (avg_model, per_sample)
        avg_model, ps = [
            process_reference(conf, samples, dataset, mode, ref_name)
            for mode, ref_name in zip(
                ["avg_model", "per_sample"], ["avg_model", "sample"]
            )
        ]

        # Process regressors - linreg, lasso, elasticnet
        regressor_dfs = {
            mode: pd.concat(
                pd.read_csv(
                    EVALUATION_REGRESSOR.format(
                        **{
                            **conf.__dict__,
                            "samples": samples,
                            "dataset": dataset,
                            "context": ctxt,
                            "features": features,
                        },
                        mode=mode,
                    ),
                    index_col=0,
                ).assign(features=features)
                for ctxt, features in CONTEXTS_FEATURES
            ).assign(ref=mode, samples=samples, dataset=dataset)
            for mode in REGRESSION_MODES
        }
        print(f"Finished processing regressors for {samples}, {dataset}")

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
                avg_ps_df = avg_ps_df.assign(
                    context=context, samples=samples, dataset=dataset, features="None"
                ).replace(
                    np.nan, "N/A"
                )  # replace NaNs with "N/A" to avoid FutureWarning re. empty/NaN entries
                avg_ps_dfs.append(avg_ps_df)
                # Once appended, this can be deleted
                del avg_ps_df

        # regression baselines already have context
        for _, rdf in regressor_dfs.items():
            avg_ps_df = rdf.copy()
            avg_ps_dfs.append(avg_ps_df)
            # Once appended, this can be deleted
            del avg_ps_df

        # TODO @GiacomoFabrini might it be better to have default activation, optimiser, orth_reg_strategy as "None"?
        dfd = pd.concat([training.convert_dtypes(), *avg_ps_dfs])
        print(
            f"Finished concatenating training and reference models for {samples}, {dataset}"
        )
        dfs.append(dfd)
        le_dfs.append(temp_results["latent_embeddings"])
        param_dev_dfs.append(temp_results["parameter_deviations"])
        param_dfs.append(temp_results["full_parameters"])
        # Cleanup
        del training, avg_ps_dfs, rdf, dfd, temp_results

df = pd.concat(dfs, ignore_index=True)
del dfs

le_df = pd.concat(le_dfs, ignore_index=True)
del le_dfs

param_dev_df = pd.concat(param_dev_dfs, ignore_index=True)
del param_dev_dfs

param_df = pd.concat(param_dfs, ignore_index=True)
del param_dfs

for results_df in (le_df, param_dev_df, param_df):
    results_df["job"] = results_df["job"].astype(int)


# Select reg_param for plotting based on the number of unique investigated values
reg_params = [
    "l1reg_inflate",
    "oreg_inflate",  # inflater
    "l1reg_encode",
    "oreg_encode",  # encoder
    "l1reg_inflater_output",
    "l2reg_inflater_output",
    "median_reg",
    "inflater_output_reg_epoch",  # param dev, param medians
    "sparse_threshold_perc",
]
num_unique_regs = [
    len(df[df.ref == "DMM"][reg_param].unique()) for reg_param in reg_params
]
reg_param = reg_params[num_unique_regs.index(max(num_unique_regs))]

# ########################################################################### #
# ############################### Aggregation ############################### #
# ########################################################################### #

# Aggregate data, save CSVs and log W&B artifacts (currently disabled)
num_best = 10
aggregated_results = aggregate_and_log(
    df=df,
    conf=conf,
    top_reg_param=reg_param,
    return_stat_tests=RETURN_STAT_TESTS,
    num_best=num_best,
)
if RETURN_STAT_TESTS:
    (
        data,
        stat_test_res_df,
        top_n_dmm_train,
        best_hyperparam_dmm,
        best_regressors,
        unified_dmm_results,
    ) = aggregated_results
else:
    (
        data,
        top_n_dmm_train,
        best_hyperparam_dmm,
        best_regressors,
        unified_dmm_results,
    ) = aggregated_results

#
# # ########################################################################### #
# # ################## Most predictive hyperparams for RMSE ################### #
# # ########################################################################### #
#
# # train_rf_features_to_rmse(
# #     dmm_results=unified_dmm_results,
# #     conf=conf,
# #     num_top_features=10
# # )
#
# # ########################################################################### #
# # #### Restrict embeddings and params to top 10 jobs (train) per config ##### #
# # ########################################################################### #
# # Subset parameter deviation, parameter and latent embeddings to top N=10 jobs
# top_n_param_dev_df_train, top_n_param_df_train, top_n_le_df_train = [
#     df.merge(top_n_dmm_train, how="inner", on=default_attributes)[df.columns]
#     for df in [param_dev_df, param_df, le_df]
# ]
#
# # Compute PCA latent embeddings -- from [2 (LE1, LE2) * num_top_jobs]
# # features/columns down to [2 (LE1*, LE2*)] components. Auto-centering
# # performed through PCA itself.
# top_n_pca_le_df_train = pca_latent_embeddings(
#     top_n_le_df_train, hyperparam_configs, scale=False
# ).reset_index()
#
# for df_to_save, df_label in zip(
#     [
#         top_n_param_dev_df_train,
#         top_n_param_df_train,
#         top_n_le_df_train,
#         top_n_pca_le_df_train,
#     ],
#     ["param_dev", "param", "le", "pca_le"],
# ):
#     df_to_save.to_csv(
#         evaluations_dir
#         / f"{conf.model}"
#         / f"{conf.data}"
#         / f"{conf.model}.{conf.data}.top_{num_best}_{df_label}.csv"
#     )
#
# # Get Cellosaurus annotations
# brca_annot_df = get_cell_line_cellosaurus_annotations(
#     file_dir=features_dir / conf.model / conf.data
# )
# for (latent_embedding_df, df_label), which_cells in itt.product(
#     zip(
#         [
#             # top_n_le_df_train,
#             top_n_pca_le_df_train
#         ],
#         [
#             # "pristine",
#             "pca"
#         ],
#     ),
#     ["all", "val_only"],
# ):
#     # Add Cellosaurus and PAM50/LB annotations
#     plotting_df = add_annotations(
#         latent_embedding_df,
#         brca_annot_df,
#         subtypes_pam50,
#         subtypes_lb,
#         subtypes_hr,
#         subtypes_her2,
#     )
#     plot_latent_embeddings(
#         le_df=plotting_df,
#         df_label=df_label,
#         reg_param=reg_param,
#         save_path=str(
#             outdir
#             / "{context}.latent_embeddings.{df_label}.{which_cells}.{plot_by}.pdf"
#         ),
#         which_cells=which_cells,
#     )
#
# plt.close("all")
# sns.boxplot(top_n_pca_le_df_train, x=reg_param, y="variance_explained")
# plt.tight_layout()
# plt.savefig(
#     outdir / f"pca.latent_embeddings.{reg_param}.variance_explained.pdf"
# )
# plt.close()
#
# # ########################################################################### #
# # ############################ Performance Plots ############################ #
# # ########################################################################### #
#
# # group_plots(
# #     dataframe=data,
# #     conf=conf
# # )
# #
#
# data_nodmm = data[data.ref != "DMM"]
# data_dmm = convert_dataframe_dtypes(data[data.ref == "DMM"])
#
# for barplot_label in ["all", "top_val"]:
#     if barplot_label == "all":
#         data_top_dmm = data_dmm.merge(
#             top_n_dmm_train,
#             on=[
#                 col
#                 for col in top_n_dmm_train.columns
#                 if col not in ["rmse_train", "rmse_test"]
#             ],
#         ).drop(columns=["rmse_train", "rmse_test"])
#     else:
#         data_top_dmm = data_dmm.merge(
#             best_hyperparam_dmm,
#             on=[
#                 col
#                 for col in best_hyperparam_dmm.columns
#                 if col not in ["rmse_train", "rmse_test", "model"]
#             ],
#         ).drop(columns=["rmse_train", "rmse_test", "model"])
#
#     barplot_df = pd.concat([data_top_dmm, data_nodmm])
#
#     performance_barplot(
#         dataframe=barplot_df,
#         conf=conf,
#         group_name=f"baseline_barplot_{barplot_label}",
#     )
#
#
# # ########################################################################### #
# # ########################### Embedding Similarity ########################## #
# # ########################################################################### #
# # Cosine-similarity
# cv_cos_sim = cosine_similarity_embeddings(
#     top_n_pca_le_df_train, hyperparam_configs
# )
# cv_cos_sim.to_csv(
#     evaluations_dir
#     / f"{conf.model}"
#     / f"{conf.data}"
#     / f"{conf.model}.{conf.data}.cosine_sim_cv.csv"
# )
# # Silhouette score
# cv_silhouette = silhouette_embeddings(
#     top_n_pca_le_df_train, hyperparam_configs
# )
# cv_silhouette.to_csv(
#     evaluations_dir
#     / f"{conf.model}"
#     / f"{conf.data}"
#     / f"{conf.model}.{conf.data}.silhouette_cv.csv"
# )
# g = sns.FacetGrid(
#     cv_silhouette,
#     row="context",
#     row_order=sorted(cv_silhouette.context.unique()),
#     hue="cell_line",
# )
# g.map_dataframe(sns.scatterplot, x=reg_param, y="mean_silhouette_score")
# plt.tight_layout()
# plt.legend()
# plt.xscale("symlog")
# plt.savefig(
#     fig_dir / conf.model / conf.data / f"mean_silhouette_score_{reg_param}.pdf"
# )
# plt.close()
# print("Computed similarity scores for latent embeddings.")
#
#
# # ########################################################################### #
# # ######################### Param Deviation Analysis ######################## #
# # ########################################################################### #
# # List of parameter prefixes
# prefixes = ("EGFR", "ERK", "ERBB2", "MEK", "iMEK", "iEGFR")
# # Compute ratios between deviations and medians
# param_cols = [col for col in param_df.columns if col.startswith(prefixes)]
# # Choose whether to plot the average of all multistarts or only the top 10 with respect to training performance (rmse_train)
# plot_top_n_train = True
# samples_val = sorted(param_df[param_df.dataset != "train"].cell_line.unique())
# cell_lines = sorted(param_df.cell_line.unique())
# samples_train = [
#     cell_line for cell_line in cell_lines if cell_line not in samples_val
# ]
#
# # Plot spread of parameter deviations across multistarts for validation cell-lines
# plot_val_param_dev_spread(
#     top_n_param_dev_df_train if plot_top_n_train else param_dev_df,
#     param_cols,
#     reg_param,
#     reg_params,  # TODO any better way of dynamically defining this?
#     fig_dir
#     / conf.model
#     / conf.data
#     / f"param_dev_boxplot_val_only_{reg_param}.pdf",
# )
# plt.close("all")
#
# # Plot range of values of parameter deviations (either single jobs or median)
#
#
# for context in CONTEXT_SET:
#     for plot_label, parameter_dataframe in zip(
#         ["param", "param_dev"],
#         [
#             top_n_param_df_train if plot_top_n_train else param_df,
#             top_n_param_dev_df_train if plot_top_n_train else param_dev_df,
#         ],
#     ):
#         # Heatmaps VS Regularisation strength
#         # Subset to context and compute the median over all jobs
#         group_cols = [
#             col
#             for col in parameter_dataframe.columns
#             if (not col.startswith(prefixes)) and (col != "job")
#         ]
#         # # TODO once we reinclude this!
#         # group_cols = [col for col in group_cols if col != "sparse_threshold_perc"]
#         plot_df = (
#             parameter_dataframe[parameter_dataframe.context == context]
#             .groupby(group_cols)[param_cols]
#             .agg("median")  # CHANGED FROM MEAN TO MEDIAN
#             .reset_index()
#         )
#
#         plot_df = add_annotations(
#             plot_df,
#             brca_annot_df,
#             subtypes_pam50,
#             subtypes_lb,
#             subtypes_hr,
#             subtypes_her2,
#         )
#
#         if plot_label == "param_dev":
#             # Get ratios between param deviation ranges across train/val per samples/reg_param combo
#             # This uses the median across jobs, but we could directly use ALL JOBS (parameter_dataframe
#             val_param_dev_ratios = compute_deviation_ratio(
#                 plot_df, param_cols, reg_param
#             )
#             sns.boxplot(
#                 val_param_dev_ratios,
#                 x=reg_param,
#                 y="deviation_ratio",
#                 color="gray",
#             )
#             sns.stripplot(
#                 val_param_dev_ratios,
#                 x=reg_param,
#                 y="deviation_ratio",
#                 hue="samples",
#             )
#             plt.tight_layout()
#             plt.savefig(
#                 fig_dir
#                 / conf.model
#                 / conf.data
#                 / f"{context}.param_dev_ratio.{reg_param}.pdf"
#             )
#             plt.close()
#
#             val_param_dev_df = plot_df[
#                 plot_df.cell_line.isin(hardest_cell_lines)
#             ]
#             results_dfs = []
#             for cell_line, reg_param_val in itt.product(
#                 val_param_dev_df.cell_line.unique(),
#                 val_param_dev_df[reg_param].unique(),
#             ):
#                 # Select a single cell-line and reg strength
#                 sub_df = val_param_dev_df[
#                     (val_param_dev_df.cell_line == cell_line)
#                     & (val_param_dev_df[reg_param] == reg_param_val)
#                 ]
#                 # Get parameter for cell-line when in val set
#                 params_val = sub_df[sub_df.dataset == "val"]
#                 for samples in sub_df[
#                     sub_df.dataset != "val"
#                 ].samples.unique():
#                     # Pick sets of parameters one CV split at a time and compute MSE among all parameter deviations
#                     params = sub_df[sub_df.samples == samples]
#                     mse = np.mean(
#                         (
#                             params_val[param_cols].values
#                             - params[param_cols].values
#                         )
#                         ** 2
#                     )
#                     results_dfs.append(
#                         pd.DataFrame(
#                             {
#                                 "cell_line": [cell_line],
#                                 reg_param: [reg_param_val],
#                                 "samples": [samples],
#                                 "MSE": [mse],
#                             }
#                         )
#                     )
#             diffs = pd.concat(results_dfs).sort_values(
#                 by=["cell_line", "samples"]
#             )
#             plot_mse_param_dev_val_across_splits(
#                 diffs=diffs, conf=conf, context=context, reg_param=reg_param
#             )
#
#         for val_only, val_label in zip([True, False], ["val_only", "all"]):
#             filtered_df = (
#                 plot_df
#                 if not val_only
#                 else plot_df[plot_df.cell_line.isin(hardest_cell_lines)]
#             )
#
#             plot_parameter_heatmaps(
#                 filtered_df,
#                 param_cols,
#                 group_cols,
#                 reg_param,
#                 samples_train,
#                 samples_val,
#                 plot_label,
#                 fig_dir
#                 / conf.model
#                 / conf.data
#                 / f"{conf.model}.{conf.data}.{context}.{plot_label}.{val_label}",
#                 val_only=val_only,
#                 plot_type="heatmap",
#             )
#
#     # # PARAMETER DEVIATION HISTOGRAMS PER CELL-LINE -- REMOVED FOR NOW
#     # for val_cell_line, split_val in zip(
#     #         hardest_cell_lines[:len(SPLITS)], SPLITS
#     # ):
#     #     param_dev_val = param_dev_df[(param_dev_df.cell_line.isin([val_cell_line])) & (param_dev_df.context == context)]
#     #     # Extract column names that start with any of the prefixes
#     #     parameters = [col for col in param_dev_val.columns if col.startswith(prefixes)]
#     #     columns = parameters + ["l1reg_inflater_output", "samples"]
#     #     l1reg_values = sorted(param_dev_val.l1reg_inflater_output.unique())
#     #     # Reshape the dataframe using melt to get a 'parameter' column and corresponding values
#     #     df_melted = param_dev_val.melt(
#     #         id_vars=["l1reg_inflater_output","samples"],
#     #         value_vars=parameters,
#     #         var_name="parameter",
#     #         value_name="value",
#     #     )
#     #
#     #     # Create FacetGrid with one column per parameter, one CV split per row
#     #     g = sns.FacetGrid(
#     #         df_melted, col="parameter", row="samples",
#     #         row_order=[f"{i}of5" for i in range(5)],
#     #         sharex=False,
#     #         sharey=False,
#     #         height=3,
#     #         aspect=2
#     #     )
#     #
#     #     # Map histogram plots to each facet
#     #     g.map_dataframe(
#     #         sns.histplot,
#     #         x="value",
#     #         hue="l1reg_inflater_output",
#     #         hue_order=l1reg_values,
#     #         palette="tab10",
#     #     )
#     #
#     #     # Adjust layout for better spacing
#     #     g.set_titles(col_template="{col_name}")
#     #     g.tight_layout()
#     #     # plt.legend()
#     #     plt.savefig(
#     #         fig_dir / conf.model / conf.data / f"{conf.model}.{conf.data}.{context}.param_dev_{val_cell_line}.pdf")
#     #     plt.show()
#     #     plt.close()
#
#
# # ########################################################################## #
# # ######################### Statistical Test Plots ######################### #
# # ########################################################################## #
# if RETURN_STAT_TESTS:
#     # n_hidden pairwise comparisons:
#     # subset to where n_hidden is null (n_hidden1 and n_hidden2 will be not null)
#     n_hidden_pairwise_heatmap(
#         dataframe=stat_test_res_df[stat_test_res_df.n_hidden.isnull()],
#         conf=conf,
#     )
#     # Volcano plot of hyperparameter significance in improving (reducing) rmse_val:
#     # subset to where n_hidden1 is null (for pairwise n_hidden comparisons above)
#     volcano_hyperparameter_significance(
#         dataframe=stat_test_res_df[stat_test_res_df.n_hidden1.isnull()],
#         conf=conf,
#     )
#
# # ########################################################################### #
# # ########################## Time-varying Response ########################## #
# # ########################################################################### #
# # Load measurement and observable dataframes
# df_meas, df_obs = get_measurements_and_obervables(conf)
#
# # Setup features_test for regressors - need to ensure all contexts and splits have the same number of features/columns
# features_test = {context: None for context in CONTEXT_SET}
#
# for dataset, context, split in itt.product(
#     ["train", "val"],
#     CONTEXT_SET,
#     sorted(SPLITS),  # ensure processing from 0of5 to 4of5
# ):
#     # Load petab base files and training/validation split
#     conf.samples = split
#     petab_base_files = load_petab_base_files(conf)
#     samples_dict = {
#         "train": training_samples(Wildcards(conf.data, split)),
#         "val": val_samples(Wildcards(conf.data, split)),
#     }
#
#     # Get per-sample simulation
#     problem = CytofProblem(conf.model)
#     per_sample_sim_dfs = []
#     for sample in samples_dict[dataset]:
#         output = process_per_sample_pretrain(
#             sample,
#             problem,
#             conf,
#             pretrain_dir / conf.model / conf.data,
#             petab_base_files,
#         )
#         if output is None:
#             # file not found
#             continue
#         _, simulation_df = output
#         per_sample_sim_dfs.append(simulation_df)
#     per_sample_sim_df = pd.concat(per_sample_sim_dfs)
#
#     # Get avg_model simulation
#     avg_model_sim_df = simulate_avg_model(
#         conf, pretrain_dir / conf.model / conf.data, petab_base_files, dataset
#     )
#     # Process and subset measurement dataset
#     avg_model_sim_df, df_meas_subset = process_avg_model_simulation(
#         avg_model_sim_df, df_meas, dataset, samples_dict
#     )
#
#     # Get best-regressor simulation
#     regressor_mode = best_regressors[
#         (best_regressors.context == context)
#         & (best_regressors.dataset == dataset)
#     ].ref.values[0]
#     trained_pipeline_file = REGR_TRAINED_PIPELINE.format(
#         model=conf.model,
#         data=conf.data,
#         samples=conf.samples,
#         mode=regressor_mode,
#         context=context,
#     )
#     features_train_file = REGR_FEATURES_TRAIN.format(
#         model=conf.model,
#         data=conf.data,
#         samples=conf.samples,
#         mode=regressor_mode,
#         context=context,
#     )
#     # load using joblib
#     trained_pipeline = load(trained_pipeline_file)
#     features_train = load(features_train_file)
#
#     # Load input and output data and fit regression pipeline to get simulation
#     input_data, _ = load_data(
#         contextualization=context,
#         samples=samples_dict[dataset],
#         features=features_train if dataset == "val" else None,
#         measurement_table=petab_base_files["measurement_table"],
#         observable_table=petab_base_files["observable_table"],
#         features_filepath=get_features_filepath(
#             replace(conf, context=context, features="all"),
#             FEATURES_OUTFILE,
#         )
#         if context == "MOSA"
#         else None,
#     )
#     output_data, test_columns = load_data(
#         contextualization="cytof_dynamic",
#         samples=samples_dict[dataset]
#         if context != "MOSA"
#         else input_data.index,  # restrict samples for MOSA (not all cell-lines available)
#         features=features_test[context] if dataset == "val" else None,
#         measurement_table=petab_base_files["measurement_table"],
#         observable_table=petab_base_files["observable_table"],
#     )
#     # Get features_test in "train" to later use with "test"
#     if features_test[context] is None:
#         features_test[context] = test_columns
#
#     best_regressor_sim_df = (
#         pd.DataFrame(
#             trained_pipeline.predict(input_data),
#             index=output_data.index,
#             columns=output_data.columns,
#         )
#         .T.stack()
#         .reset_index()
#         .sort_values(
#             by=[
#                 "preequilibrationConditionId",
#                 "observableId",
#                 "simulationConditionId",
#                 "time",
#             ]
#         )
#         .reset_index()
#         .drop(columns="index")
#         .rename(columns={0: "simulation"})
#     )
#     # TODO @GiacomoFabrini - both avg_model_sim_df and per_sample_sim_df have 698 rows, but
#     #  best_regressor_sim_df only has 608 - what are those 90 rows missing from the latter?
#     #  Is this related to the missing/inconsistent timepoints in some samples?
#
#     # BEST DMM -- chosen as best on validation set when considering performance from top 10 jobs on training set
#     # TODO @GiacomoFabrini - ensure this is cast to int when it is generated
#     best_hyperparam_dmm["sparse_threshold_perc"] = best_hyperparam_dmm[
#         "sparse_threshold_perc"
#     ].astype(int)
#     best_config_jobs = sorted(
#         best_hyperparam_dmm[
#             (best_hyperparam_dmm.context == context)
#             & (best_hyperparam_dmm.samples == split)
#         ].job.unique()
#     )
#     best_dmm_conf_obj = [
#         Conf(
#             model=conf.model,
#             data=conf.data,
#             **best_hyperparam_dmm[
#                 (best_hyperparam_dmm.context == context)
#                 & (best_hyperparam_dmm.samples == split)
#                 & (best_hyperparam_dmm.job == job)
#             ]
#             .drop(columns=["rmse_train", "rmse_test", "ref", "model"])
#             .to_dict(orient="records")[0],
#         )
#         for job in best_config_jobs
#     ]
#     # Compute features once (same across all jobs) - depend on SPLIT
#     # TODO @GiacomoFabrini - this needs to be adapted for multimodal concatenation
#     input_features = load_and_transform_features(
#         conf=best_dmm_conf_obj[0],
#         dataset=dataset,
#         features_filepath=FEATURES_OUTFILE.format(
#             **{**best_dmm_conf_obj[0].__dict__, "dataset": "{dataset}"}
#         ),
#     )
#     overall_best_dmm_sim_dfs = []
#
#     # temp_latent_embeddings, temp_parameter_medians, temp_parameter_deviations = [], [], []
#     for job, overall_best_conf in zip(best_config_jobs, best_dmm_conf_obj):
#         # simulation errors might result in missing files -> skip
#         try:
#             models, obj = load_model_and_obj(
#                 conf=overall_best_conf,
#                 petab_base_files=petab_base_files,
#                 dataset=dataset,
#                 num_ensemble_members=1,  # use the best ensemble member by default
#             )
#         except FileNotFoundError:
#             continue
#
#         # Simulate and append to growing pd.DataFrame list
#         overall_best_dmm_sim_dfs.append(
#             simulate_dmm(
#                 model=models[0],
#                 input_features=input_features,
#                 obj=obj,
#                 petab_problem=models[0].petab_importer.petab_problem,
#                 jit_fn=False,
#             ).assign(job=job)
#         )
#
#     # Concatenate simulations for time-varying response plot
#     overall_best_dmm_sim_df = pd.concat(overall_best_dmm_sim_dfs)
#     del overall_best_dmm_sim_dfs
#     # Add identifier column (to keep replicate datapoints)
#     overall_best_dmm_sim_df["unique_id"] = list(
#         range(int(len(overall_best_dmm_sim_df) / len(best_config_jobs)))
#     ) * int(len(best_config_jobs))
#     # Group by all necessary columns, compute mean for "simulation" column and drop unnecessary columns
#     overall_best_dmm_sim_df = (
#         overall_best_dmm_sim_df.groupby(
#             [
#                 "observableId",
#                 "preequilibrationConditionId",
#                 "time",
#                 "noiseParameters",
#                 "simulationConditionId",
#                 "measurementType",
#                 "observableParameters",
#                 "unique_id",
#             ]
#         )
#         .mean()
#         .reset_index()
#         .drop(columns=["job", "unique_id"])
#     )
#
#     # Plot the time-varying response
#     plot_cross_samples_multiple_simulations(
#         measurement_df=df_meas_subset,
#         simulation_dfs=[
#             per_sample_sim_df,
#             avg_model_sim_df,
#             best_regressor_sim_df,
#             overall_best_dmm_sim_df,
#         ],
#         labels=[  # TODO @GiacomoFabrini need to find a way to use these to produce secondary legend (currently unused)
#             "per_sample",
#             "avg_model",
#             "best_regressor",
#             "best_dmm_overall",
#         ],
#         linetypes=[
#             "dashed",
#             "dashdot",
#             "dotted",
#             "solid",
#         ],
#         linesizes=[
#             1,
#             1,
#             1,
#             1.25,  # slightly thicker lines for DMM models
#         ],
#         figdir=outdir / dataset,
#         prefix="__".join(
#             [
#                 dataset,
#                 split,
#                 context,
#                 "comparison_best",
#             ]
#         ),
#     )
#
# print("Done.")
