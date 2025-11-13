import equinox as eqx
import jax.numpy as jnp
import pandas as pd
import re
import warnings

from common import FEATURES_OUTFILE
from dataclasses import replace
from dmm.dmm_autoencoder_eqx import DeepMechanisticModel
from dmm.config_options import Conf
from dmm.initialisation import get_features_filepath, process_features_and_setup_models
from dmm.training_helper_funcs import create_pypesto_problem, rmse
from evaluation_utils import load_model
from util import load_petab_base_files
from tqdm import tqdm
from typing import List, Optional, Tuple

warnings.filterwarnings("ignore")


def classify_param(name):
    # Match EGFR only when preceded and followed by non-letters or boundary
    egfr = re.search(r'(?<![A-Za-z])EGFR(?![A-Za-z])', name)
    erbb2 = re.search(r'(?<![A-Za-z])ERBB2(?![A-Za-z])', name)

    if egfr and not erbb2:
        return "EGFR"
    elif erbb2 and not egfr:
        return "ERBB2"
    elif egfr and erbb2:
        return "EGFR+ERBB2"
    else:
        return "Other"


def set_param_to_zero(
        dmm_model: DeepMechanisticModel,
        param: str
) -> DeepMechanisticModel:
    # Find parameter deviation index
    param_idx = dmm_model.parameter_deviation_names.index(param)
    # Extract binary mask and convert to JAX array
    new_output_sparsity_binary_mask = jnp.array(dmm_model.output_sparsity_binary_mask)
    # Set value to zero and convert back to tuple format
    new_output_sparsity_binary_mask = new_output_sparsity_binary_mask.at[param_idx].set(0)
    new_output_sparsity_binary_mask = tuple(new_output_sparsity_binary_mask.tolist())
    # Replace inside model and return new model instance
    return eqx.tree_at(
        lambda model: model.output_sparsity_binary_mask,
        dmm_model,
        new_output_sparsity_binary_mask,
    )


def activate_single_param(
        dmm_model: DeepMechanisticModel,
        param: str
) -> DeepMechanisticModel:
    # Find parameter deviation index
    param_idx = dmm_model.parameter_deviation_names.index(param)
    # Create a blank binary mask just like the one in the model
    new_output_sparsity_binary_mask = jnp.zeros_like(jnp.array(dmm_model.output_sparsity_binary_mask))
    # Set parameter value to 1 and convert back to tuple format
    new_output_sparsity_binary_mask = new_output_sparsity_binary_mask.at[param_idx].set(1)
    new_output_sparsity_binary_mask = tuple(new_output_sparsity_binary_mask.tolist())
    # Replace inside model and return new model instance
    return eqx.tree_at(
        lambda model: model.output_sparsity_binary_mask,
        dmm_model,
        new_output_sparsity_binary_mask,
    )


def compute_parameter_sensitivities(
        conf: Conf,
        context_features: List[Tuple[str, str]],
        splits: List[str],
        n_jobs: int = 10,
        parameters_interest: Optional[List[str]] = None,
        multiheaded_multimodal: bool = True,
) -> pd.DataFrame:
    petab_base_files = load_petab_base_files(conf)

    res = []
    for context, feature_sel in context_features:
        subconf = replace(conf, context=context, features=feature_sel)

        if context == "multimodal":
            subconf = replace(subconf, multiheaded=multiheaded_multimodal)

        for samples in sorted(splits):
            print(f"Now analysing {context}, {samples}...")
            subconf = replace(subconf, samples=samples)

            features_filepath = get_features_filepath(subconf, FEATURES_OUTFILE)

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
                    rmse_pristine[dataset] = float(rmse(pypesto_problems[dataset], model, features[dataset].values))

                for param in tqdm(parameters_interest, desc=f"job={str(job)}"):
                    # Zero the parameter at hand
                    zeroed_model = set_param_to_zero(
                        model, param
                    )

                    # Zero all parameters and activate only the parameter at hand
                    activated_model = activate_single_param(
                        model, param
                    )
                    for dataset in ["train", "val"]:
                        rmse_zeroed = rmse(pypesto_problems[dataset], zeroed_model, features[dataset].values)
                        rmse_activated = rmse(pypesto_problems[dataset], activated_model, features[dataset].values)

                        res.append(
                            {
                                "context": context,
                                "samples": samples,
                                "job": job,
                                "param": param,
                                "dataset": dataset,
                                "rmse_pristine": rmse_pristine[dataset],
                                "rmse_zeroed": rmse_zeroed,
                                "rmse_activated": rmse_activated,
                                "rmse_diff_zeroed": rmse_zeroed - rmse_pristine[dataset],
                                "rmse_diff_activated": rmse_activated - rmse_pristine[dataset],
                            }
                        )

    results_df = pd.DataFrame(res)
    # Extract which receptor (EGFR/ERBB2, if any) each parameter belongs to
    results_df["receptor_group"] = results_df["param"].apply(classify_param)
    return results_df
