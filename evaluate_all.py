import fire
import itertools as itt
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import wandb

from common import (
    Conf,
    EVALUATION_REFERENCE,
    EVALUATION_REGRESSOR,
    EVALUATION_TRAINING,
    fig_dir,
    evaluations_dir,
    CONTEXT_SET
)
from dmm.analysis import plot_loss_vs_regularization
from evaluation_plotting import (group_plots,
                                  performance_barplot,
                                  n_hidden_pairwise_heatmap,
                                  volcano_hyperparameter_significance)
from training_configuration import (
    CONTEXTS_FEATURES, SPLITS, PRETRAIN,
    LATENT_DIMS, NETWORK_LAYOUT, USE_BIAS, NN_INIT_FN,
    RECONSTRUCT, ACTIVATION_FNS, OPTIMISERS,
    ORTH_REG_STRATEGIES, ALPHAS, BETAS, GAMMAS, DELTAS, EPSILONS, ZETAS,
    MAX_LEARNING_RATES, LEARNING_RATE_SPANS, LEARNING_RATE_DECAYS, WARMUP_FCTS, OPT_STEPS, OPT_MULT, LINEAR_SCHEDULE,
    USE_EARLY_STOP,
)
from stat_test import statistical_significance_test


def aggregate_and_log(df):
    # Define aggregation groups for DMM
    gbs = [
        "dataset",
        "context",
        "features",
        "samples",
        "ref",
        "orth_reg_strategy",
        "latent dim",
        "l1reg_inflate",
        "oreg_inflate",
        "l1reg_encode",
        "oreg_encode",
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
    #print("Overall evaluation DataFrame is now ready.")
    # cleanup
    del df, df_refs, data_dmm, data_refs

    # Prepare statistical test dataframe
    # Create pivot table for statistical testing
    cols = ['dataset', 'context', 'features', 'ref',
            'latent dim', 'orth_reg_strategy',
            'l1reg_inflate', 'oreg_inflate',
            'l1reg_encode', 'oreg_encode']
    # pivot table and create one column per cross-validation split and multistart/job
    pivot_data = data.pivot_table(index=cols, columns=['samples', 'job'], values='rmse')
    pivot_data = pivot_data.reset_index()
    # Create list of the MultiIndex RMSE columns created above
    multiindex_rmse_cols = [(sample, job) for sample in SPLITS for job in JOBS]
    # Create a single column 'rmse_list' listing all values from each of the MultiIndex columns (same order for all rows)
    pivot_data['rmse_list'] = pivot_data.apply(lambda row: np.array([row[col] for col in multiindex_rmse_cols]), axis=1)
    # Add the newly created column to the list of columns to be kept (cols)
    cols += ['rmse_list']
    # Subset the pivot table and reduce MultiIndex back to single-level index
    data_stat_tests = pivot_data[cols]
    data_stat_tests.columns = data_stat_tests.columns.droplevel(level=1)
    print("DataFrame for statistical testing is now ready.")

    stat_test_res_df = statistical_significance_test(data_stat_tests)

    # Log via W&B
    wandb.init(
        project=f"DeepMechanisticModels.{conf.data}.{conf.model}",
        config={
            **conf.__dict__,
        },
    )

    for evaluation_df, evaluation_tag in zip(
            [data, stat_test_res_df], ["evaluate_all", "stat_tests_all"]
    ):
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

    return data, stat_test_res_df


conf = fire.Fire(Conf)

outdir = fig_dir / conf.model / conf.data

# METHODS = ("pca embedding", "end-to-end")  # not used at the moment

JOBS = tuple([i for i in range(10)])  # need to change this - NO HARDCODING
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
                 use_simple_linear_schedule, use_early_stopping, job,
                 ) in itt.product(
                CONTEXTS_FEATURES,
                PRETRAIN,
                LATENT_DIMS,
                NETWORK_LAYOUT,
                USE_BIAS,
                NN_INIT_FN,
                RECONSTRUCT,
                ACTIVATION_FNS,
                OPTIMISERS,
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
            for mode in ["linreg", "lasso", "elasticnet"]
        }
        print(f'Finished processing regressors for {samples}, {dataset}')

        print(f'Starting to build hyperparam/job combination copies for references models - {samples}, {dataset}')
        missing_hyperparams = [
            "features",
            "encoder_layer_sizes", "inflater_layer_sizes", "linear_benchmark",
            "use_layer_bias", "nn_init_fn",
            "reconstruct", "activation_fn_name", "optimiser",
            "orth_reg_strategy",
            "l1reg_inflate", "oreg_inflate", "l1reg_encode", "oreg_encode", "recon_loss", "symm_reg",
            "max_lrate", "lrate_span", "lrate_decay", "warmup_fct", "opt_steps", "opt_mult",
            "use_simple_linear_schedule", "use_early_stopping",
            "job",
        ]
        avg_ps_dfs = []
        for context in CONTEXT_SET:
            # need to replicate info across contexts for "avg_model" and "sample"
            for rdf in [  # lack context
                # avg,
                avg_model,
                ps,
            ]:
                avg_ps_df = rdf.copy()
                # they have no hyperparams -- None
                avg_ps_df["context"] = context
                # avg_ps_df["type"] = method
                # avg_ps_df["pretrain"] = pretrain
                for col in missing_hyperparams:
                    avg_ps_df[col] = None
                avg_ps_dfs.append(avg_ps_df)
                # Once appended, this can be deleted
                del avg_ps_df

        # regression baselines already have context
        # but no hyperparameters
        for _, rdf in regressor_dfs.items():
            avg_ps_df = rdf.copy()
            for col in missing_hyperparams:
                avg_ps_df[col] = None
            # avg_ps_df["type"] = method
            # avg_ps_df["pretrain"] = pretrain
            avg_ps_dfs.append(avg_ps_df)
            # Once appended, this can be deleted
            del avg_ps_df
        print(f"Finished processing reference models for {samples}, {dataset}")

        # dfd = pd.concat([training, pretraining])
        dfd = pd.concat([training, *avg_ps_dfs])
        # Deleting DataFrames once concatenated into dfd
        del training, avg_ps_dfs, regressor_dfs
        dfd["dataset"] = dataset
        dfd["samples"] = samples
        dfs.append(dfd)
        # Deleting dfd once appended to dfs
        del dfd
        print(f"Finished concatenating training and reference models for {samples}, {dataset}")

df = pd.concat(dfs).reset_index()
# Now that dfs have been concatenated into df, delete them
del dfs
df.rename(
    columns={
        "layers": "latent dim",
        # "type": "method", #not used at the moment?!
    },
    inplace=True,
)

# Aggregate data into DataFrames for plotting, save the results as CSVs and log them
# as W&B artifacts
data, stat_test_res_df = aggregate_and_log(df)

# ########################################################################### #
# ############################ Performance Plots ############################ #
# ########################################################################### #

group_plots(
    dataframe=data,
    conf=conf
)

performance_barplot(
    dataframe=data,
    conf=conf
)

# ########################################################################## #
# ######################### Statistical Test Plots ######################### #
# ########################################################################## #
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
