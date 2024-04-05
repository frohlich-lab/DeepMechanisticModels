import fire
import numpy as np
import petab

from amici.petab_objective import rdatas_to_simulation_df
from common import (
    Conf,
    MEASUREMENTS_FILE_RW,
    PER_SAMPLE_OUTFILE_RESULTS,
    Wildcards,
    training_samples,
)
from cytof.problem import CytofProblem
from dmm.petab_subproblem import load_petab
from dmm.pretraining import generate_per_sample_pretraining_problems
from pypesto.store import OptimizationResultHDF5Reader
from util import load_petab_base_files

conf = fire.Fire(Conf)

problem = CytofProblem(conf.model)

petab_base_files = load_petab_base_files(conf)

samples = training_samples(Wildcards(conf.data, conf.samples))

petab_base_importer = load_petab(
    problem=problem,
    dataset=conf.data,
    **petab_base_files,
)

sigmas = {}
for sample in samples:
    importer = generate_per_sample_pretraining_problems(
        importer=petab_base_importer,
        problem=problem,
        dataset=conf.data,
        sample=sample,
    )
    pypesto_problem = importer.create_problem()
    rfile = PER_SAMPLE_OUTFILE_RESULTS.format(
        model=conf.model, data=conf.data, sample=sample
    )
    result = OptimizationResultHDF5Reader(rfile).read()

    problem.apply_objective_settings(pypesto_problem.objective)
    x = pypesto_problem.get_reduced_vector(result.optimize_result.list[0].x)
    res = pypesto_problem.objective(x, return_dict=True)

    simulation_df = rdatas_to_simulation_df(
        res["rdatas"],
        model=pypesto_problem.objective.amici_model,
        measurement_df=importer.petab_problem.measurement_df,
    )

    residuals_df = importer.petab_problem.measurement_df.copy()
    residuals_df["residual"] = (
        importer.petab_problem.measurement_df[petab.MEASUREMENT]
        - simulation_df[petab.SIMULATION]
    )

    sigmas.update(
        {
            (sample, observable, condition): np.sqrt(
                np.mean(np.square(group_df["residual"]))
            )
            for (observable, condition), group_df in residuals_df.groupby(
                [petab.OBSERVABLE_ID, petab.SIMULATION_CONDITION_ID]
            )
        }
    )

measurement_df = petab_base_files["measurement_table"].copy()
for (sample, observable, condition), sigma in sigmas.items():
    measurement_df.loc[
        (measurement_df[petab.OBSERVABLE_ID] == observable)
        & (measurement_df[petab.SIMULATION_CONDITION_ID] == condition)
        & (measurement_df[petab.PREEQUILIBRATION_CONDITION_ID] == sample),
        petab.NOISE_PARAMETERS,
    ] = sigma

measurement_df.to_csv(
    MEASUREMENTS_FILE_RW.format(
        model=conf.model, data=conf.data, samples=conf.samples
    ),
    sep="\t",
)
