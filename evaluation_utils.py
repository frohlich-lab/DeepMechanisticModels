import jax.random as jr
import numpy as np
import pandas as pd
import petab

from amici.petab_objective import rdatas_to_simulation_df
from common import (
    Conf,
    MEASUREMENTS_FILE,
    OBSERVABLES_FILE,
    TRAINED_BEST_MODELS,
    Wildcards,
    test_samples,
    training_samples,
)
from cytof.problem import CytofProblem
from dmm.initialisation import setup_models
from dmm.petab_subproblem import load_petab
from dmm.pretraining import generate_average_pretraining_problem, generate_per_sample_pretraining_problems
from dmm.training_helper_funcs import create_pypesto_problem
from typing import Dict


def get_measurements_and_obervables(conf: Conf):
    df_meas = pd.read_csv(
        MEASUREMENTS_FILE.format(**conf.__dict__), sep="\t", index_col=0
    )
    df_obs = pd.read_csv(
        OBSERVABLES_FILE.format(**conf.__dict__), sep="\t", index_col=0
    )
    df_meas = df_meas[
        df_meas[petab.OBSERVABLE_ID].apply(lambda x: x in df_obs.index)
    ]
    return df_meas, df_obs


def load_model_and_obj(conf: Conf, petab_base_files: Dict[str, pd.DataFrame], dataset: str):
    # Initialise model skeleton and get CytofProblem
    model, cytof_problem = setup_models(conf, petab_base_files, dataset)

    # Create pypesto problem
    pypesto_problem = create_pypesto_problem(model)
    # Extract base objective
    obj = pypesto_problem.objective.base_objective

    # Define filepaths for training results and serialized model - only the latter is needed
    # infile = TRAINING_OUTFILE_RESULTS.format(**conf.__dict__)
    trained_model_file = TRAINED_BEST_MODELS.format(**conf.__dict__)

    # Load training results - TODO @GiacomoFabrini - do we need these?
    # reader = OptimizationResultHDF5Reader(infile)
    # result = pypesto.Result(pypesto_problem)
    # result.optimize_result = reader.read().optimize_result

    # Load serialised best model
    model.load(
        trained_model_file,
        cytof_problem,
        petab_base_files['measurement_table'],
        petab_base_files['observable_table'],
        petab_base_files['condition_table'],
        jr.PRNGKey(conf.job)
    )
    return model, obj


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
