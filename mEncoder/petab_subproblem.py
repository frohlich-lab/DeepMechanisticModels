import petab

import pandas as pd
import numpy as np

from amici.petab_import import PysbPetabProblem
from pypesto.petab.pysb_importer import PetabImporterPysb

from . import parameter_boundaries_scales, MODEL_FEATURE_PREFIX, load_pathway, basedir

from typing import Tuple, Sequence
from pathlib import Path


def load_petab(
    datafiles: Tuple[Path, Path, Path],
    pathway_name: str,
    l1reg: float,
    samples: Sequence[str] = None,
) -> PetabImporterPysb:
    """
    Imports data from a csv and converts it to the petab format. This
    function is used to connect the mechanistic model to the specified data
    in order to defines the loss function of the autoencoder up to the
    inflated parameters

    :param datafiles:
        tuple of paths to measurements, conditions and observables files

    :param pathway_name:
        name of pathway to use for model

    :param l1reg:
        TBD
    """
    measurement_table = pd.read_csv(datafiles[0], index_col=0, sep="\t")
    condition_table = pd.read_csv(datafiles[1], index_col=0, sep="\t")
    observable_table = pd.read_csv(datafiles[2], index_col=0, sep="\t")

    # TEMPORARY: filter out baseline data
    measurement_table = measurement_table[
        measurement_table[petab.SIMULATION_CONDITION_ID]
        != measurement_table[petab.PREEQUILIBRATION_CONDITION_ID]
    ]

    if samples:
        measurement_table = measurement_table[
            measurement_table[petab.PREEQUILIBRATION_CONDITION_ID].apply(
                lambda x: x in samples
            )
        ]
        condition_table = condition_table.loc[
            [c for c in condition_table.index if c.split("__")[0] in samples], :
        ]

    model = load_pathway(pathway_name)

    features = [
        par for par in model.parameters if par.name.startswith(MODEL_FEATURE_PREFIX)
    ]

    # CONDITION TABLE
    # this defines the different samples. here we define the mapping from
    # input parameters to model parameters

    preeq_conds = dict()
    for cond in list(condition_table.index):
        candidates = measurement_table[
            measurement_table[petab.SIMULATION_CONDITION_ID] == cond
        ][petab.PREEQUILIBRATION_CONDITION_ID].unique()
        if len(candidates) > 1:
            raise RuntimeError(
                "Found multiple different preequilibration "
                f"conditions {candidates} for condition "
                f"{cond}, which is not supported."
            )
        if len(candidates) == 0:
            preeq_conds[cond] = cond
        else:
            preeq_conds[cond] = candidates[0]

    for feature in features:
        condition_table[feature.name] = [
            f"{feature.name}__{preeq_conds[s]}" for s in condition_table.index
        ]

    # PARAMETER TABLE
    # this defines the full set of parameters including boundaries, nominal
    # values, scale, priors and whether they will be estimated or not.
    params = [
        par.name for par in model.parameters if par.name not in condition_table.columns
    ]

    if petab.OBSERVABLE_PARAMETERS in measurement_table:
        params += sorted(
            list(
                set(
                    [
                        par
                        for pars in measurement_table[petab.OBSERVABLE_PARAMETERS]
                        for par in pars.split(";")
                        if par and any(obs in par for obs in observable_table.index)
                    ]
                )
            )
        )

    transforms = {"lin": lambda x: x, "log10": lambda x: np.power(10.0, x)}

    # base definition of id, upper and lower bounds, scale and value
    param_defs = [
        {
            petab.PARAMETER_ID: par,
            petab.LOWER_BOUND: transforms[
                parameter_boundaries_scales[par.split("_")[-1]][2]
            ](parameter_boundaries_scales[par.split("_")[-1]][0]),
            petab.UPPER_BOUND: transforms[
                parameter_boundaries_scales[par.split("_")[-1]][2]
            ](parameter_boundaries_scales[par.split("_")[-1]][1]),
            petab.PARAMETER_SCALE: parameter_boundaries_scales[par.split("_")[-1]][2],
            petab.NOMINAL_VALUE: model.parameters[par].value
            if par in model.parameters.keys()
            else 1e3
            if par.endswith("offset") and par.startswith("p")
            else 1.0
            if par.endswith("offset") and par.startswith("t")
            else 1.0,
        }
        for par in params
        if not par.startswith(MODEL_FEATURE_PREFIX)
    ]

    par_inputs = [par for par in params if par.startswith(MODEL_FEATURE_PREFIX)]

    # add additional input parameters for every base condition
    for cond in measurement_table[petab.PREEQUILIBRATION_CONDITION_ID].unique():
        param_defs.extend(
            [
                {
                    petab.PARAMETER_ID: f"{par.name}__{cond}"
                    if par in features
                    else par,
                    petab.LOWER_BOUND: 0.1,
                    petab.UPPER_BOUND: 10.0,
                    petab.PARAMETER_SCALE: petab.LOG10,
                    petab.NOMINAL_VALUE: 1.0 if par in features else 0.0,
                }
                for par in features + par_inputs
                if par in features or par.endswith(f"__{cond}")
            ]
        )

    parameter_table = pd.DataFrame(param_defs)

    input_pars = (
        parameter_table[petab.PARAMETER_ID]
        .apply(lambda x: x.startswith(MODEL_FEATURE_PREFIX))
        .values
    )

    parameter_table[petab.ESTIMATE] = 1
    # piece of codes allows disabling estimation for (non-input) parameters by
    # setting equal upper and lower bounds, primarily for debugging purposes
    parameter_table.loc[np.logical_not(input_pars), petab.ESTIMATE] = (
        parameter_table.loc[np.logical_not(input_pars), petab.LOWER_BOUND]
        != parameter_table.loc[np.logical_not(input_pars), petab.UPPER_BOUND]
    ).apply(lambda x: int(x))

    parameter_table.set_index(petab.PARAMETER_ID, inplace=True)

    # add l2 regularization to input parameters (only if estimating them)
    parameter_table[petab.OBJECTIVE_PRIOR_TYPE] = [
        petab.PARAMETER_SCALE_LAPLACE
        if name.startswith(MODEL_FEATURE_PREFIX) and l1reg > 0
        else np.NaN
        for name in parameter_table.index
    ]
    parameter_table[petab.OBJECTIVE_PRIOR_PARAMETERS] = [
        f"0.0;{1/l1reg}"
        if name.startswith(MODEL_FEATURE_PREFIX) and l1reg > 0
        else np.NaN
        for name in parameter_table.index
    ]

    data_name = "__".join(datafiles[0].stem.split("__")[:-1])

    problem = PysbPetabProblem(
        measurement_df=measurement_table,
        condition_df=condition_table,
        observable_df=observable_table,
        parameter_df=parameter_table,
        pysb_model=model,
    )

    filter_observables(problem)
    # petab.lint_problem(problem)

    return PetabImporterPysb(
        problem,
        output_folder=str(basedir / "amici_models" / f"{model.name}_{data_name}_petab"),
    )


def filter_observables(petab_problem: petab.Problem):
    petab_problem.measurement_df = petab_problem.measurement_df.loc[
        petab_problem.measurement_df[petab.OBSERVABLE_ID].apply(
            lambda x: x in petab_problem.observable_df.index
        ),
        :,
    ]
    # filter obsolete observable pars
    obs_pars = set(
        p
        for ir, r in petab_problem.measurement_df.iterrows()
        if petab.OBSERVABLE_PARAMETERS in r
        for p in r[petab.OBSERVABLE_PARAMETERS].split(";")
    )
    for par in list(petab_problem.parameter_df.index):
        if not par.endswith("_scale") and not par.endswith("_offset"):
            continue
        if par not in obs_pars:
            petab_problem.parameter_df.drop(index=par, inplace=True)
