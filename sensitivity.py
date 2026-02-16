import re
import warnings
from dataclasses import replace
from typing import List, Optional, Tuple, Union

import equinox as eqx
import jax.numpy as jnp
import pandas as pd
from tqdm import tqdm

from common import FEATURES_OUTFILE
from dmm.config_options import Conf
from dmm.dmm_autoencoder_eqx import DeepMechanisticModel
from dmm.initialisation import (
    get_features_filepath,
    process_features_and_setup_models,
)
from dmm.training_helper_funcs import create_pypesto_problem, rmse
from evaluation_utils import load_model
from util import load_petab_base_files

warnings.filterwarnings("ignore")


def classify_param(name):
    # Match EGFR only when preceded and followed by non-letters or boundary
    egfr = re.search(r"(?<![A-Za-z])EGFR(?![A-Za-z])", name)
    erbb2 = re.search(r"(?<![A-Za-z])ERBB2(?![A-Za-z])", name)

    if egfr and not erbb2:
        return "EGFR"
    elif erbb2 and not egfr:
        return "ERBB2"
    elif egfr and erbb2:
        return "EGFR+ERBB2"
    else:
        return "Other"


def set_param_to_zero(
    dmm_model: DeepMechanisticModel, param: str
) -> DeepMechanisticModel:
    # Find parameter deviation index
    param_idx = dmm_model.parameter_deviation_names.index(param)
    # Extract binary mask and convert to JAX array
    new_output_sparsity_binary_mask = jnp.array(
        dmm_model.output_sparsity_binary_mask
    )
    # Set value to zero and convert back to tuple format
    new_output_sparsity_binary_mask = new_output_sparsity_binary_mask.at[
        param_idx
    ].set(0)
    new_output_sparsity_binary_mask = tuple(
        new_output_sparsity_binary_mask.tolist()
    )
    # Replace inside model and return new model instance
    return eqx.tree_at(
        lambda model: model.output_sparsity_binary_mask,
        dmm_model,
        new_output_sparsity_binary_mask,
    )


def activate_single_param(
    dmm_model: DeepMechanisticModel, param: str
) -> Union[DeepMechanisticModel, None]:
    # Find parameter deviation index and get sparsity mask value
    param_idx = dmm_model.parameter_deviation_names.index(param)
    param_mask_value = jnp.array(dmm_model.output_sparsity_binary_mask)[
        param_idx
    ]
    # If parameter has been removed from sparsity, return None to skip
    if param_mask_value == 0:
        # print("Parameter was set to zero by sparsity, skipping...")
        return None
    else:
        # Create a blank binary mask just like the one in the model
        new_output_sparsity_binary_mask = jnp.zeros_like(
            jnp.array(dmm_model.output_sparsity_binary_mask)
        )
        # Set parameter value to 1 and convert back to tuple format
        new_output_sparsity_binary_mask = new_output_sparsity_binary_mask.at[
            param_idx
        ].set(1)
        new_output_sparsity_binary_mask = tuple(
            new_output_sparsity_binary_mask.tolist()
        )
        # Replace inside model and return new model instance
        return eqx.tree_at(
            lambda model: model.output_sparsity_binary_mask,
            dmm_model,
            new_output_sparsity_binary_mask,
        )


def zero_latent_direction(
    dmm_model: DeepMechanisticModel, zero_idx: int, latent_dim: int = 2
):
    """
    Return a copy of `dmm_model` whose deep_encoder output has dim `zero_idx` forced to 0.
    Supports multiheaded multimodal DMMs.
    Does not currently support DMMs with more than 1 layer in encoder module.
    """
    assert (
        0 <= zero_idx < latent_dim
    ), f"zero_idx must be in [0, {latent_dim-1}]"
    if not dmm_model.multiheaded:
        encoder_weights = dmm_model.deep_encoder.layers[0].weight
        masked_weights = encoder_weights.at[zero_idx].set(0)
        return eqx.tree_at(
            lambda m: m.deep_encoder.layers[0].weight,
            dmm_model,
            masked_weights,
        )
    else:
        # Copy model
        updated_model = dmm_model
        for i, encoder in enumerate(dmm_model.deep_encoder):
            encoder_weights = encoder.layers[0].weight
            masked_weights = encoder_weights.at[zero_idx].set(0)
            updated_model = eqx.tree_at(
                lambda m: m.deep_encoder[i].layers[0].weight,  # noqa: B023
                updated_model,
                masked_weights,
            )
        return updated_model


def compute_sensitivities(
    conf: Conf,
    context_features: List[Tuple[str, str]],
    splits: List[str],
    latent_sensitivities: dict[str, bool],
    n_jobs: int = 10,
    parameters_interest: Optional[List[str]] = None,
    multiheaded_multimodal: bool = True,
) -> Tuple[pd.DataFrame, Union[pd.DataFrame, None]]:
    petab_base_files = load_petab_base_files(conf)

    res_params = []
    res_latent = []

    for context, feature_sel in context_features:
        subconf = replace(conf, context=context, features=feature_sel)
        if context == "multimodal":
            subconf = replace(subconf, multiheaded=multiheaded_multimodal)

        for samples in sorted(splits):
            subconf = replace(subconf, samples=samples)

            features_filepath = get_features_filepath(
                subconf, FEATURES_OUTFILE
            )

            # Get pypesto subproblems and feature sets
            (
                example_model,
                problem,
                pypesto_subproblems,
                features,
            ) = process_features_and_setup_models(
                conf=subconf,
                features_filepath=features_filepath,
                petab_base_files=petab_base_files,
                dataset="train+val",
            )

            # Create pypesto problems
            pypesto_problems = {
                dataset: create_pypesto_problem(pypesto_subproblems[dataset])
                for dataset in ["train", "val"]
            }

            # Get global list of parameter deviation names to perturb
            if parameters_interest is None:
                parameters_interest = example_model.parameter_deviation_names

            for job in range(n_jobs):
                subconf = replace(subconf, job=job)

                model = load_model(subconf, pypesto_subproblems["train"])

                # Precompute pristine RMSEs once per dataset
                rmse_pristine = {}
                for dataset in ["train", "val"]:
                    rmse_pristine[dataset] = float(
                        rmse(
                            pypesto_problems[dataset],
                            model,
                            features[dataset].values,
                        )
                    )

                # Parameter sensitivities
                for param in tqdm(
                    parameters_interest,
                    desc=f"context={context}, split={samples}, job={str(job)}, params",
                ):
                    # Zero the parameter at hand
                    zeroed_model = set_param_to_zero(model, param)

                    # Zero all parameters and activate only the parameter at hand
                    activated_model = activate_single_param(model, param)
                    for dataset in ["train", "val"]:
                        rmse_zeroed = rmse(
                            pypesto_problems[dataset],
                            zeroed_model,
                            features[dataset].values,
                        )
                        if activated_model is not None:
                            rmse_activated = rmse(
                                pypesto_problems[dataset],
                                activated_model,
                                features[dataset].values,
                            )
                        else:
                            # parameter affected by sparsity mask -> skip
                            rmse_activated = rmse_pristine[dataset]

                        res_params.append(
                            {
                                "context": context,
                                "samples": samples,
                                "job": job,
                                "param": param,
                                "dataset": dataset,
                                "rmse_pristine": rmse_pristine[dataset],
                                "rmse_zeroed": rmse_zeroed,
                                "rmse_activated": rmse_activated,
                                "rmse_diff_zeroed": rmse_zeroed
                                - rmse_pristine[dataset],
                                "rmse_diff_activated": rmse_activated
                                - rmse_pristine[dataset],
                            }
                        )

                # Latent dimension sensitivities (importances)
                if latent_sensitivities[context]:
                    for latent_dimension in tqdm(
                        range(subconf.n_hidden),
                        desc=f"context={context}, split={samples}, job={str(job)}, latents",
                    ):
                        latent_zeroed_model = zero_latent_direction(
                            model,
                            latent_dimension,
                            latent_dim=subconf.n_hidden,
                        )
                        for dataset in ["train", "val"]:
                            rmse_zeroed = rmse(
                                pypesto_problems[dataset],
                                latent_zeroed_model,
                                features[dataset].values,
                            )
                            res_latent.append(
                                {
                                    "context": context,
                                    "samples": samples,
                                    "job": job,
                                    "latent_dimension": latent_dimension,
                                    "dataset": dataset,
                                    "rmse_pristine": rmse_pristine[dataset],
                                    "rmse_zeroed": rmse_zeroed,
                                    "rmse_diff": rmse_zeroed
                                    - rmse_pristine[dataset],
                                }
                            )

    # Assemble output DataFrames
    results_params_df = pd.DataFrame(res_params)
    results_latent_df = pd.DataFrame(res_latent) if res_latent else None

    # Extract which receptor (EGFR/ERBB2, if any) each parameter belongs to
    results_params_df["receptor_group"] = results_params_df["param"].apply(
        classify_param
    )
    return results_params_df, results_latent_df
