from typing import List, Sequence

import numpy as np
import pandas as pd
import petab.v1 as petab
import pysb
from petab.v1.models.pysb_model import PySBModel
from pypesto.petab.importer import PetabImporter

from . import MODEL_FEATURE_PREFIX
from .problem import Problem


def generate_parameter_table(
    problem: Problem,
    model: pysb.Model,
    condition_table: pd.DataFrame,
    measurement_table: pd.DataFrame,
    observable_table: pd.DataFrame,
    features: List[pysb.Parameter],
) -> pd.DataFrame:

    if "__" in model.name:
        modifications = model.name.split("__")[1].split("_")
    else:
        modifications = []

    # this defines the full set of parameters including boundaries, nominal
    # values, scale, priors and whether they will be estimated or not.
    params = [
        par.name
        for par in model.parameters
        if (par.name not in condition_table.columns and par.name != "__k_t")
        # Avoid dropping mutation parameters for validation cell-lines (none of which exhibits those mutations)
        or (
            ("m_BRAF_kw" in par.name and "mbraf" in modifications) or
            ("m_KRAS_kw" in par.name and "mkras" in modifications)
        )
    ]

    if petab.OBSERVABLE_PARAMETERS in measurement_table:
        params += sorted(
            {
                par
                for pars in measurement_table[petab.OBSERVABLE_PARAMETERS]
                for par in pars.split(";")
                if par and any(obs in par for obs in observable_table.index)
            }
        )

        if "pobs" in modifications:
            for marker in ["EGFR", "ERBB2"]:
                if (f"f{marker.lower()}" in modifications) or (f"t{marker.lower()}" in modifications):
                    for param_type in ["offset", "scale"]:
                        param_check = f"t{marker}_obs_{param_type}"
                        if param_check not in params:
                            params.append(param_check)


    transforms = {"lin": lambda x: x, "log10": lambda x: np.power(10.0, x)}

    # base definition of id, upper and lower bounds, scale and value
    param_defs = [
        {
            petab.PARAMETER_ID: par,
            petab.LOWER_BOUND: transforms[
                problem.bounds[par.split("_")[-1]][2]
            ](problem.bounds[par.split("_")[-1]][0]),
            petab.UPPER_BOUND: transforms[
                problem.bounds[par.split("_")[-1]][2]
            ](problem.bounds[par.split("_")[-1]][1]),
            petab.PARAMETER_SCALE: problem.bounds[par.split("_")[-1]][2],
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

    par_inputs = [
        par for par in params if par.startswith(MODEL_FEATURE_PREFIX)
    ]

    # add additional input parameters for every base condition
    for cond in measurement_table[
        petab.PREEQUILIBRATION_CONDITION_ID
    ].unique():
        param_defs.extend(
            [
                {
                    petab.PARAMETER_ID: f"{par.name}__{cond}",
                    petab.LOWER_BOUND: 1e-3,
                    petab.UPPER_BOUND: 1e3,
                    petab.PARAMETER_SCALE: petab.LOG10,
                    petab.NOMINAL_VALUE: 1.0,
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

    return parameter_table


def load_petab(
    problem: Problem,
    dataset: str,
    measurement_table: pd.DataFrame,
    condition_table: pd.DataFrame,
    observable_table: pd.DataFrame,
    samples: Sequence[str] = None,
) -> PetabImporter:
    """Imports data from a csv and converts it to the petab format. This
    function is used to connect the mechanistic model to the specified data
    in order to define the loss function of the autoencoder up to the
    inflated parameters
    """
    # TEMPORARY: filter out baseline data
    measurement_table = measurement_table[
        np.logical_or(
            # dynamic cytof data
            measurement_table[petab.SIMULATION_CONDITION_ID]
            != measurement_table[petab.PREEQUILIBRATION_CONDITION_ID],
            # proteomics for pobs
            measurement_table[petab.OBSERVABLE_ID].str.startswith(
                ("tEGFR_obs",)
            )
            & (measurement_table["measurementType"] == "proteomics"),
        )
    ]

    if samples:
        measurement_table = measurement_table[
            measurement_table[petab.PREEQUILIBRATION_CONDITION_ID].isin(
                samples
            )
        ]
        condition_table = condition_table.loc[
            [c for c in condition_table.index if c.split("__")[0] in samples],
            :,
        ]

    model = problem.load_pysb()

    for col in condition_table.columns:
        if (condition_table[col] == 0).all():
            par_cols = [
                par for par in model.parameters.keys() if f"_{col}_" in par
            ]
            # disable estimation
            condition_table[par_cols] = 0

    features = [
        par
        for par in model.parameters
        if par.name.startswith(MODEL_FEATURE_PREFIX)
    ]

    # CONDITION TABLE
    # this defines the different samples. here we define the mapping from
    # input parameters to model parameters

    preeq_conds = {}
    for cond in list(condition_table.index):
        candidates = measurement_table[
            measurement_table[petab.SIMULATION_CONDITION_ID] == cond
        ][petab.PREEQUILIBRATION_CONDITION_ID].unique()
        if len(candidates) > 1:
            raise RuntimeError(
                f"Found multiple different preequilibration conditions {candidates} for condition {cond}, which is not "
                f"supported."
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
    parameter_table = generate_parameter_table(
        problem=problem,
        model=model,
        condition_table=condition_table,
        measurement_table=measurement_table,
        observable_table=observable_table,
        features=features,
    )

    petab_problem = petab.Problem(
        measurement_df=measurement_table,
        condition_df=condition_table,
        observable_df=observable_table,
        parameter_df=parameter_table,
        model=PySBModel(model, model.name),
    )

    filter_observables(petab_problem)
    # petab.lint_problem(problem)

    # general PetabImporter compared to old PetabImporterPysb
    return PetabImporter(
        petab_problem,
        model_name=model.name,
        output_folder=str(
            problem.amici_dir / f"{problem.model_name}_{dataset}_petab"
        ),
        validate_petab=False,
    )


def filter_observables(petab_problem: petab.Problem):
    petab_problem.measurement_df = petab_problem.measurement_df.loc[
        petab_problem.measurement_df[petab.OBSERVABLE_ID].apply(
            lambda x: x in petab_problem.observable_df.index
        ),
        :,
    ]
    # filter obsolete observable pars
    obs_pars = {
        p
        for ir, r in petab_problem.measurement_df.iterrows()
        if petab.OBSERVABLE_PARAMETERS in r
        for p in r[petab.OBSERVABLE_PARAMETERS].split(";")
    }

    if "__" in petab_problem.model.model.name:
        modifications = petab_problem.model.model.name.split("__")[1].split("_")
    else:
        modifications = []

    for par in list(petab_problem.parameter_df.index):
        if not par.endswith("_scale") and not par.endswith("_offset"):
            continue
        if par not in obs_pars:
            if "pobs" in modifications:
                marker = par.split("_")[0][1:]
                if (
                        (f"f{marker.lower()}" in modifications) or (f"t{marker.lower()}" in modifications)
                ) and (f"t{marker}" in par):
                    continue
            petab_problem.parameter_df.drop(index=par, inplace=True)
