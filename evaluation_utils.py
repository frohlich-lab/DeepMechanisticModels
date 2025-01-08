import jax.numpy as jnp
import numpy as np
import pandas as pd
import petab

from amici.petab_objective import rdatas_to_simulation_df
from common import (
    MEASUREMENTS_FILE,
    OBSERVABLES_FILE,
    TRAINED_BEST_MODELS,
    Wildcards,
    test_samples,
    training_samples,
)
from cytof.problem import CytofProblem
from dmm.config_options import Conf
from dmm.dmm_autoencoder_eqx import DeepMechanisticModel
from dmm.petab_subproblem import load_petab
from dmm.pretraining import generate_average_pretraining_problem, generate_per_sample_pretraining_problems
from dmm.training_helper_funcs import create_pypesto_problem
from jax import vmap
from pathlib import Path
from training_configuration import N_ENSEMBLE_MEMBERS
from typing import Dict, Tuple, Any, Union


def get_measurements_and_obervables(conf: Conf):
    df_meas = pd.read_csv(
        MEASUREMENTS_FILE.format(**conf.__dict__), sep="\t", index_col=0
    )
    df_obs = pd.read_csv(
        OBSERVABLES_FILE.format(**conf.__dict__), sep="\t", index_col=0
    )
    df_meas = df_meas[
        df_meas[petab.v1.OBSERVABLE_ID].apply(lambda x: x in df_obs.index)
    ]
    return df_meas, df_obs


def load_model_and_obj(
        conf: Conf, petab_base_files: Dict[str, pd.DataFrame], dataset: str, num_ensemble_members: int
) -> tuple[list[DeepMechanisticModel], Any]:
    # Get cytof problem
    cytof_problem = CytofProblem(conf.model)

    # Define filepaths for serialized models -- need to be formatted for ensemble_id
    trained_model_file = TRAINED_BEST_MODELS.format(
        **{**conf.__dict__, **dict(ensemble_id="{ensemble_id}")}
    )

    models = []
    for ensemble_id in range(
            min(num_ensemble_members, N_ENSEMBLE_MEMBERS)
    ):
        ensemble_member_file = Path(trained_model_file.format(ensemble_id=ensemble_id))

        # Load ensemble member model
        model = DeepMechanisticModel.load(
            filename=ensemble_member_file,
            problem=cytof_problem,
            dataset=dataset,
            petab_base_files=petab_base_files,
        )
        models.append(model)

    # Create pypesto problem from any of the loaded models to extract objective
    pypesto_problem = create_pypesto_problem(models[0])
    obj = pypesto_problem.objective.base_objective.base_objective
    return models, obj


def process_per_sample_pretrain(
        sample: str,
        problem,
        conf: Conf,
        indir,
        petab_base_files: Dict[str, pd.DataFrame]
):
    rfile = indir / f"{sample}.csv"
    if not rfile.exists():
        return None

    petab_base_importer = load_petab(
        problem,
        conf.data,
        **petab_base_files,
    )

    importer = generate_per_sample_pretraining_problems(
        petab_base_importer,
        problem,
        conf.data,
        sample,
    )

    problem_sample = importer.create_problem()
    df = pd.read_csv(rfile, index_col=[0])
    problem.apply_objective_settings(problem_sample.objective)

    ress = []
    fvals = []
    for ipar in range(len(df)):
        x = problem_sample.get_reduced_vector(
            df.values[ipar, :], problem_sample.x_free_indices
        )
        res = problem_sample.objective(x, return_dict=True)
        ress.append(res)
        fvals.append(res["fval"])

    # Convert the simulation to PEtab format.
    simulation_df = rdatas_to_simulation_df(
        ress[np.argmin(fvals)]["rdatas"],
        model=problem_sample.objective.amici_model,
        measurement_df=importer.petab_problem.measurement_df,
    )
    return importer, simulation_df


def simulate_avg_model(
        conf: Conf,
        indir,
        petab_base_files: Dict[str, pd.DataFrame],
        dataset: str,
) -> pd.DataFrame:
    problem = CytofProblem(conf.model)
    rfile = indir / f"model_average_{conf.samples}.csv"

    petab_base_importer = load_petab(
        problem,
        conf.data,
        **petab_base_files,
    )

    importer = generate_average_pretraining_problem(
        petab_base_importer,
        problem,
        conf.data,
        training_samples(Wildcards(conf.data, conf.samples))
        if dataset == "train"
        else test_samples(Wildcards(conf.data, conf.samples)),
    )
    problem_sample = importer.create_problem()
    df = pd.read_csv(rfile, index_col=[0])
    problem.apply_objective_settings(problem_sample.objective)

    ress = []
    fvals = []
    for ipar in range(len(df)):
        x = problem_sample.get_reduced_vector(
            df.values[0, :], problem_sample.x_free_indices
        )
        res = problem_sample.objective(x, return_dict=True)
        ress.append(res)
        fvals.append(res["fval"])

    # Convert the simulation to PEtab format.
    avg_model = rdatas_to_simulation_df(
        ress[np.argmin(fvals)]["rdatas"],
        model=problem_sample.objective.amici_model,
        measurement_df=importer.petab_problem.measurement_df,
    )
    return avg_model


def process_avg_model_simulation(
        avg_model: pd.DataFrame,
        df_meas: pd.DataFrame,
        dataset: str,
        samples: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    avg_model[petab.SIMULATION_CONDITION_ID] = df_meas[
        petab.SIMULATION_CONDITION_ID
    ]
    avg_model[petab.PREEQUILIBRATION_CONDITION_ID] = df_meas[
        petab.PREEQUILIBRATION_CONDITION_ID
    ]
    df_meas = df_meas[
        df_meas[petab.PREEQUILIBRATION_CONDITION_ID].isin(samples[dataset])
    ]
    avg_model = avg_model[
        avg_model[petab.PREEQUILIBRATION_CONDITION_ID].isin(samples[dataset])
    ]
    return avg_model, df_meas


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
            "cell_line": dmm_model.sample_name_list,
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
