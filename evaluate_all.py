import os

import fire
import pandas as pd

# import subprocess
# import wandb
from common import (
    EVALUATE_ALL_CSVS,
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
    CONTEXTS_FEATURES_BY_FIGURE,
    PARAMS_TO_SCAN,
    RETURN_STAT_TESTS,
    SELECT_CENTRAL_VALUES_BY_FIGURE,
    SPLITS_BY_FIGURE,
)


def process_reference(
    conf: Conf, samples: str, dataset: str, mode: str, ref_name: str
) -> pd.DataFrame:
    print(f"Processing {mode} model for {samples}, {dataset}")
    ref = pd.read_csv(
        EVALUATION_REFERENCE.format(
            **{
                **conf.to_dict(),
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
# Compute figure-specific set of unique contexts
CONTEXT_SET = sorted(
    {context for context, _ in CONTEXTS_FEATURES_BY_FIGURE[conf.figure]}
)

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
    contexts_features=CONTEXTS_FEATURES_BY_FIGURE[conf.figure],
    n_starts=conf.n_starts,
    select_central_values=SELECT_CENTRAL_VALUES_BY_FIGURE[conf.figure],
    params_to_scan=PARAMS_TO_SCAN[conf.figure],
    splits=SPLITS_BY_FIGURE[conf.figure],
)
hyperparam_configs = {
    samples: [
        hyperparam_config
        for hyperparam_config in hyperparam_configs
        if hyperparam_config["samples"] == samples
    ]
    for samples in SPLITS_BY_FIGURE[conf.figure]
}

# Load evaluations (DMMs, baselines, regressors), latent embeddings, parameters and parameter deviations
dfs, le_dfs, param_dev_dfs, param_dfs = [], [], [], []
for samples in sorted(SPLITS_BY_FIGURE[conf.figure]):
    for dataset in ["train", "val"]:
        dfs_sample_dataset = [
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
        ]
        if not len(dfs_sample_dataset):
            continue

        # DMM evaluations
        training = pd.concat(dfs_sample_dataset).assign(
            ref="DMM",  # previously called "meth"
            dataset=dataset,
        )
        print(
            f"Finished concatenating training evaluations for {samples}, {dataset}"
        )

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
                [
                    pd.read_csv(
                        EVALUATION_REGRESSOR.format(
                            **{
                                **conf.to_dict(),
                                "samples": samples,
                                "dataset": dataset,
                                "context": ctxt,
                                "features": features,
                            },
                            mode=mode,
                        ),
                        index_col=0,
                    ).assign(features=features)
                    for ctxt, features in CONTEXTS_FEATURES_BY_FIGURE[
                        conf.figure
                    ]
                ]
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
                    context=context,
                    samples=samples,
                    dataset=dataset,
                    features="None",
                )
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
le_df.to_csv(
    EVALUATE_ALL_CSVS.format(
        model=conf.model, data=conf.data, filename=f"embeddings_{conf.figure}"
    )
)
del le_dfs

param_dev_df = pd.concat(param_dev_dfs, ignore_index=True)
param_dev_df.to_csv(
    EVALUATE_ALL_CSVS.format(
        model=conf.model, data=conf.data, filename=f"param_devs_{conf.figure}"
    )
)
del param_dev_dfs

param_df = pd.concat(param_dfs, ignore_index=True)
del param_dfs

for results_df in (le_df, param_dev_df, param_df):
    results_df["job"] = results_df["job"].astype(int)

# ########################################################################### #
# ############################### Aggregation ############################### #
# ########################################################################### #

# Aggregate data, save CSVs and log W&B artifacts (currently disabled)
num_best = 10
aggregated_results = aggregate_and_log(
    df=df,
    conf=conf,
    return_stat_tests=RETURN_STAT_TESTS,
    num_best=num_best,
)
print("Done.")
