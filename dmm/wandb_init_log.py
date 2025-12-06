import os

import equinox as eqx
import git
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import wandb

from .config_options import (
    IO_SPARSITY,
    L1EREG,
    L1IREG,
    L1REG_IO,
    L2REG_IO,
    MEDIAN_REG,
    OEREG,
    OIREG,
    RECON_LOSS,
    SYMM_LOSS,
    Conf,
    EarlyStoppingParams,
)
from .dmm_autoencoder_eqx import DeepMechanisticModel


def log_performance(df, dataset, rmse):
    gbs = ["sample", "condition", "observable"]
    df_rmse = df[gbs + ["rmse"]].groupby(gbs).agg("mean").reset_index()
    tab_rmse = wandb.Table(data=df_rmse)
    wandb.log({f"rmse_{dataset}": rmse, f"rmses_{dataset}": tab_rmse})


def init_wandb_ltt(
    conf: Conf,
    type: str,
):
    """
    Initialise W&B run. Run name = chosen hyperparameters in configuration string representation.
    """
    repo = git.Repo(search_parent_directories=True)

    # default is "relu" but it is not applied unless there is at least 1 hidden layer in a given model module
    activation_fn_tag = "None" if conf.depth == 0 else conf.activation_fn_name
    group = f"{conf.model}_{conf.context}_{conf.features}"

    wandb.init(
        project="DeepMechanisticModels",
        group=group,
        config={
            "model": conf.model,
            "features": conf.features,
            "context": conf.context,
            "type": type,
            "samples": conf.samples,
            "commit": repo.head.object.hexsha,
        },
        name=conf.__str__(replace={"activation_fn_name": activation_fn_tag}),
        settings=wandb.Settings(
            git_commit=repo.head.object.hexsha,
            git_remote_url=repo.remotes.origin.url,
        ),
        mode="online",  # to run more jobs simultaneously on the cluster
    )

    for metric in ["rmse_train", "rmse_val"]:
        wandb.define_metric(metric, summary="min")


def init_wandb(
    model: DeepMechanisticModel,
    conf: Conf,
    early_stopping_params: EarlyStoppingParams,
):
    """
    Initialise W&B run. Run name = chosen hyperparameters in configuration string representation.
    """
    repo = git.Repo(search_parent_directories=True)

    # default is "relu" but it is not applied unless there is at least 1 hidden layer in a given model module
    activation_fn_tag = "None" if conf.depth == 0 else conf.activation_fn_name
    group = f"{conf.context}_{conf.features}"

    base_model = conf.model.split("__")[0]
    if "__" in conf.model:
        modifications = conf.model.split("__")[1].split("_")
    else:
        modifications = []

    job_id = os.environ.get("SLURM_JOB_ID", "0")
    node_name = os.environ.get("SLURMD_NODENAME", "local")

    wandb.init(
        # v2: Equinox
        # v3: Equinox, back to basics -- no decoder, simple decay learning rate schedule, first local attempts
        # v4: Equinox, basics - LinearScans
        # v5: Equinox, no network pretraining + leave-one-out cross-validation - Linear Scans
        # v6: same as v5, reduced, run locally due to wandb issues with cluster
        # v7: exploring various regularisation strategies, cluster, old wandb backend
        # v8: various tests on l1reg_inflater_output and epoch + median_reg
        # v11: l1reg scheduling
        # v12: MOSA 200 starts, no l1reg scheduling
        # v13: l1reg scheduling with fixed inflater_output_reg_epoch (500, half range), scanning sparsity percentage
        # v14: l1reg scheduling as above, but fixed best_models behaviour + updated sparse_threshold_perc behaviour
        # v15: l1reg scheduling, scanning optimal value for l1reg_inflater_output on CV 1of5
        # v16: updated feature selection (uniform across CV splits), regressors with feature selection, unregularised
        # v17: new/old mechanistic model, no reweighing, fixed Chi2 (MSE), no biases on last inflater layer (deviations)
        # v18: fixed mechanistic model, removed pretraining and relevant code, removed schedule-free optimisers
        # v19: same as v18 but with frozen kinetic param median
        # v20: updated metric to impose sparsity (median -> standard deviation); updated learning rate schedule
        # v21: force selection of pERBB2 features in cytof_init + n_hidde=3 for all contexts
        # v22: new CV split (cHCC2185 -> cUACC3199), new feature selection (per_cv vs across_cv)
        # v23: new CV split, per_cv selection, no l1reg inflater or sparsity, trying n_hidden, with/without frozen medians
        # v24: new feature selection, reduce depth to 0, explore l1reg_inflater_output, sparsity, l2reg_inflater_output, feature selection, constant schedule
        # v25: new feature selection
        # v26: new feature selection, optimal l1/l2reg_inflater_output, scanning l1reg & oreg on synced inflater/encoder
        # v27: drop l1reg/oreg for encoder/inflater; test decoder reconstruction
        # v28: standard scaling
        # v29: no bias, no standard scaling
        # v30: no bias, no standard scaling, scanning depth + testing multimodal
        # v31: no bias, with/without standard scaling, no depth, using last layer activation, incl. multimodal
        # v33: different mechanistic model architectures
        # v34: feature selection v2
        # v35: feature selection v2 with lower inflater reg and egfra model
        # v36: cytof_init only with different model variants incorporating variation in EGFR and ERBB2
        # v37: proteomics+transcriptomics, with different feature selection approaches
        # v38: cytof+px+tx linear scans over depth, width, and l1reg_inflater_output iteration
        # v39: cytof+px+tx linear scan over n_features
        # v40: re-running 'base' features with added regressor evaluations (with 'features' saved)
        # v40c: same as v40, but with input features centred around 0 (mean subtracted)
        # v41: added p90RSK to EGFR_MAPK, removed ERBB2 from freeeq model and fixed EGFR_MAPK_AKT model
        # v42: clean slate with p90RSK added, minimal setup with new directories
        # v43: faster initialisation for tegfr model, add baseline activation for EGFR and p90RSK, linear cytof observables
        # v44: revert faster initialisation (took too long, will revisit)
        # v45: model variants
        # v46: updated model, add HER2 signaling + inhibition by lapatinib
        # v47: features selection revisited
        # v49: refactored model
        # v50: code refactor, scan inflater output reg
        # v51: output reg + inflater bound scan
        # v52: figure 1a (snakemake figure logic, no ML scans, all contexts and splits, base __logobs model)
        # v53: figure 2a (cytof_init + tEGFR / pEGFR / both), __logobs, __fegfr and __fegfr_pobs (both logobs & aggavg)
        # v56_fig3: figure 3 (cytof_init), __logobs, __tegfr and __tegfr_pobs (both logobs & aggavg)
        # v57_fig4: figure 4 (cytof_init), __logobs_tegfr with or without mutations, growth factors and baselines
        # v52_fig1aext: figure 1a (snakemake figure logic, no ML scans, all contexts and splits, base __logobs model) - MDA-MB-468
        # v58_fig1b: scan over n_hidden, depth, l1/l2 inflater regularisation (fixed feature number = 10)
        # v58_fig1c: scan over features (without remaining ML scan)
        # v60_fig2a: fix up evaluate_all, new 1e-2 central value for l1reg_inflater_output, 2A runs with added subtype-augmented cytof_init
        # v61_fig3: fix up evaluate_all, new 1e-2 central value for l1reg_inflater_output, 3 runs with added subtype-augmented cytof_init
        # v62_fig2b: fix up evaluate_all, new 1e-2 central value for l1reg_inflater_output, 2B (ferbb2) runs with added subtype-augmented cytof_init
        # v63_fig1b: repeating 1B centred around new l1reg_inflater_output value (1e-2)
        # v64_fig1c: repeating 1C centred around new l1reg_inflater_output value (1e-2)
        # v65_fig4: repeating 4 centred around new l1reg_inflater_output value (1e-2), only base, tEGFR and mutations
        # v61_fig3missing: rerunning runs for which .eqx models were deleted
        # v64_fig1cext: extending 1C with curated feature sets
        # v65_fig1bext: extending 1B with ML scans around best features for proteomics/transcriptomics
        # v54: p38 model
        # v55: new model cv splits and l1 inflater scan, no recon loss
        # ff.v0: more contexts, updated feature selection, simplified model
        # ff.v1: add HER3
        # ff.v2: implement serum
        # v66_fig1a: repeating MAPK runs after merging with p38 and adding subchallenge I cell-lines + subchallenge II gold standard
        # v67_fig1adebug: rolled back data additions, debugging model
        # v68_fig1adebug: rolled back feature selection, fixed UACC893, debugging model
        # ffv3 rerun parameter scan
        # v69_fig1a_p38: repeating Fig 1A with whole data + multiheaded multimodal RFE
        # v69_fig1b_p38: same as above, Fig 1B ML scans (added dropout rate + reconstruction loss scans)
        # ffv4 repeat feature scan with new data
        # v70_fig1b_p38: same as above with updated central values + expanded ranges
        # ffv5: figure 3 with updated configuration
        # v71_fig1b_p38: same as above with updated central values + retuned ranges (n_hidden, depth, l2reg_inflater_output)
        # ffv6: figure 1c rerun
        # v72_fig1b_p38: same as above with updated dropout_rate; scanning more n_hidden and dropout_rate (only)
        # v73_fig3_p38: tEGFR fig3, updated n_hidden (6) and dropout (0.1)
        # ffv7: figure 1c rerun
        # v73_fig3c_p38:base & tEGFR, cytof & multimodal, scanning n_hidden and depth
        # v73_fig2a_p38: base vs fEGFR model, n_hidden=8, depth=0, cytof_init & augmentations + multimodal
        # ffv8: initialisation benchmark
        # ffv9: nn_init_scale scan
        # v74_loocv_p38_noher2: tEGFR variant, cytof_init & multiheaded multimodal, LOOCV across all cell-lines -- no HER2 in features
        # v75_fig3: clean repo, figure 3 runs (base / tEGFR / pEGFR / tERBB2 / pERBB2 model variants)
        # v76_fig3: clean repo, figure 3 runs (base / tEGFR / pEGFR / tERBB2 / pERBB2 model variants) - figures_ff merge
        # v76_fig2: clean repo, figure 2 runs (base vs fEGFR model variants) - post figures_ff merge
        project=f"DeepMechanisticModels.v76_fig2.{conf.data}",
        group=group,
        config={
            **conf.to_dict(),
            "use_early_stopping": conf.use_early_stopping,  # early-stopping enabled/disabled
            "patience": early_stopping_params.patience
            if conf.use_early_stopping
            else None,
            "min_improvement": early_stopping_params.min_improvement
            if conf.use_early_stopping
            else None,
            "scheduler": "linear"
            if conf.use_simple_linear_schedule
            else "custom",
            "commit": repo.head.object.hexsha,
            "base_model": base_model,
            **{mod: 1 for mod in modifications},
            "job_id": job_id,
            "node": node_name,
            "n_features": model.n_input_features,
        },
        name=conf.__str__(replace={"activation_fn_name": activation_fn_tag}),
        settings=wandb.Settings(
            git_commit=repo.head.object.hexsha,
            git_remote_url=repo.remotes.origin.url,
        ),
        tags=[
            "shallow_model" if conf.depth == 0 else "deep_model",
            "early_stop" if conf.use_early_stopping else "no_early_stop",
            conf.date_tag,  # label experiment with date of experiment start
        ],
        mode="online",
    )

    # Define W&B metrics
    metrics = {
        metric: "last"
        for metric in [
            "loss",
            "fval_train",
            "fval_val",
            "rmse_train",
            "rmse_val",
            "par_dev_max",
            "par_dev_norm",
            "log_parameter_std",
            "log_parameter_mean",
            L1EREG,
            L1IREG,
            L1REG_IO,
            L2REG_IO,
            MEDIAN_REG,
        ]
    }

    # common metrics - orthogonal regularisation + patience_counter
    for metric in [OEREG, OIREG]:
        metrics[metric] = "min"
    metrics["patience_counter"] = "none"
    metrics["start_rmse_val"] = "none"
    metrics["final_rmse_val"] = "none"
    metrics["integration_error"] = "none"
    # optional metrics depending on the presence of decoder head
    if conf.recon_loss:
        metrics[RECON_LOSS] = "last"
        metrics[SYMM_LOSS] = "last"

    for metric, summary in metrics.items():
        wandb.define_metric(metric, summary=summary)


def log_param_norms(
    model: DeepMechanisticModel,
    input_data: jnp.ndarray,
    epoch: int,
):
    # put in inference mode and use dummy key
    par_dev = eqx.nn.inference_mode(model).inflate_params(
        input_data, jr.PRNGKey(0)
    )
    stds = par_dev.std(axis=0)
    # compute the biggest gap in std values across parameters, this is a cheap
    # proxy for the separation of modes in a gmm model that we use for
    # sparsification
    log_std_diff = jnp.diff(jnp.log10(stds[stds > 0]).sort())
    if log_std_diff.size:
        par_dev_log_std_sep = log_std_diff.max()
    else:  # only one non-zero std
        par_dev_log_std_sep = 0
    wandb.log(
        {
            "par_dev_max": jnp.max(jnp.abs(par_dev)),
            "par_dev_norm": jnp.linalg.norm(x=par_dev, ord=None),
            "par_dev_log_std_sep": par_dev_log_std_sep,
        },
        step=epoch,
    )


def log_extra_loss_terms(
    model: DeepMechanisticModel,
    reg: dict,
    epoch: int,
):
    """
    Function to log extra loss terms (not fval nor loss itself) to W&B: regularisation terms, reconstruction loss.

    :param model:
        DeepMechanisticModel instance.
    :param reg:
        dictionary of regularisation terms
    :param epoch:
        training iteration/epoch.

    :return:
        n/a (simply logs to W&B).
    """
    # Log metrics defined above
    for key, val in reg.items():
        if val != 0:
            wandb.log({key: val}, step=epoch)

    wandb.log(
        {IO_SPARSITY: np.sum(model.output_sparsity_binary_mask)}, step=epoch
    )
