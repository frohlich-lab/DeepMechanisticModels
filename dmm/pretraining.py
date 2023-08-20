from typing import List

import numpy as np
import pandas as pd
import petab
import pypesto
from petab.models.pysb_model import PySBModel
from pypesto.objective.jax import JaxObjective
from pypesto.petab.pysb_importer import PetabImporterPysb
from pysb import Model

from . import MODEL_FEATURE_PREFIX
from .autoencoder import DeepMechanisticModel
from .problem import Problem


def generate_per_sample_pretraining_problems(
    importer: PetabImporterPysb, problem: Problem, dataset: str, sample: str
) -> PetabImporterPysb:
    """
    Creates a pypesto problem that can be used to train the
    mechanistic model individually on every sample
    """
    # construct problem based on petab for pypesto subproblem
    pp = importer.petab_problem
    pp.parameter_df[petab.ESTIMATE] = [
        not x.startswith(MODEL_FEATURE_PREFIX)
        and pp.parameter_df[petab.ESTIMATE][x]
        for x in pp.parameter_df.index
    ]
    pp.parameter_df.loc[
        pp.parameter_df[petab.ESTIMATE] == 0,
        [petab.OBJECTIVE_PRIOR_TYPE, petab.OBJECTIVE_PRIOR_PARAMETERS],
    ] = np.NaN

    # create fresh model from scratch since the petab imported one already
    # has the observables added and this might lead to issues.
    clean_model = problem.load_pysb()

    # subset measurements, conditions and parameters for specified sample
    mdf = pp.measurement_df[
        pp.measurement_df[petab.PREEQUILIBRATION_CONDITION_ID] == sample
    ]
    cdf = pp.condition_df[
        [name.startswith(sample) for name in pp.condition_df.index]
    ]
    spars = (
        set(
            e
            for t in mdf[petab.OBSERVABLE_PARAMETERS].apply(
                lambda x: x.split(";")
            )
            for e in t
        )
        if petab.OBSERVABLE_PARAMETERS in mdf
        else {}
    )
    pdf = pp.parameter_df[
        [
            (
                not name.startswith(MODEL_FEATURE_PREFIX)
                and (
                    (not name.endswith(("_scale", "_offset"))) or name in spars
                )
            )
            or name.endswith(sample)
            for name in pp.parameter_df.index
        ]
    ]

    return PetabImporterPysb(
        petab.Problem(
            parameter_df=pdf,
            observable_df=pp.observable_df,
            measurement_df=mdf,
            condition_df=cdf,
            model=PySBModel(
                Model(base=clean_model, name=pp.model.model_id),
                pp.model.model_id,
            ),
        ),
        output_folder=str(
            problem.amici_dir / f"{pp.model.model_id}_{dataset}_petab"
        ),
    )


def generate_per_sample_reg_pretraining_problem(
    importer: PetabImporterPysb,
    problem: Problem,
    avg_pars: pd.DataFrame,
    dataset: str,
    sample: str,
    alpha: float = 0.0,
) -> PetabImporterPysb:
    """
    Creates a pypesto problem that can be used to train the
    mechanistic model individually on every sample
    """
    # construct problem based on petab for pypesto subproblem
    pp = importer.petab_problem
    pp.parameter_df[petab.ESTIMATE] = [
        not x.startswith(MODEL_FEATURE_PREFIX)
        and pp.parameter_df[petab.ESTIMATE][x]
        for x in pp.parameter_df.index
    ]
    pp.parameter_df.loc[
        pp.parameter_df[petab.ESTIMATE] == 0,
        [petab.OBJECTIVE_PRIOR_TYPE, petab.OBJECTIVE_PRIOR_PARAMETERS],
    ] = np.NaN

    # create fresh model from scratch since the petab imported one already
    # has the observables added and this might lead to issues.
    clean_model = problem.load_pysb()

    # subset measurements, conditions and parameters for specified sample
    mdf = pp.measurement_df[
        pp.measurement_df[petab.PREEQUILIBRATION_CONDITION_ID] == sample
    ]
    cdf = pp.condition_df[
        [name.startswith(sample) for name in pp.condition_df.index]
    ]
    spars = (
        set(
            e
            for t in mdf[petab.OBSERVABLE_PARAMETERS].apply(
                lambda x: x.split(";")
            )
            for e in t
        )
        if petab.OBSERVABLE_PARAMETERS in mdf
        else {}
    )
    pdf = pp.parameter_df[
        [
            (
                not name.startswith(MODEL_FEATURE_PREFIX)
                and (
                    (
                        not name.endswith("_scale")
                        and not name.endswith("offset")
                    )
                    or name in spars
                )
            )
            or name.endswith(sample)
            for name in pp.parameter_df.index
        ]
    ]
    for pname in pdf.index:
        if pname.startswith(MODEL_FEATURE_PREFIX):
            pdf.loc[pname, petab.ESTIMATE] = True
            if alpha > 0:
                pdf.loc[
                    pname, petab.OBJECTIVE_PRIOR_PARAMETERS
                ] = f"0.0;{1 / alpha}"
                pdf.loc[
                    pname, petab.OBJECTIVE_PRIOR_TYPE
                ] = petab.PARAMETER_SCALE_LAPLACE

        if pname.endswith(("_scale", "_offset")):
            continue

        if pname not in avg_pars.columns:  # includes inputs
            continue

        pdf.loc[pname, petab.NOMINAL_VALUE] = np.power(
            10, avg_pars.loc[0, pname]
        )
        pdf.loc[pname, petab.ESTIMATE] = False

    return PetabImporterPysb(
        petab.Problem(
            parameter_df=pdf,
            observable_df=pp.observable_df,
            measurement_df=mdf,
            condition_df=cdf,
            model=PySBModel(
                Model(base=clean_model, name=pp.model.model_id),
                pp.model.model_id,
            ),
        ),
        output_folder=str(
            problem.amici_dir / f"{pp.model.model_id}_{dataset}_petab"
        ),
    )


def generate_average_pretraining_problem(
    importer: PetabImporterPysb,
    problem: Problem,
    dataset: str,
    samples: List[str],
) -> PetabImporterPysb:
    """
    Creates a pypesto problem that can be used to train the mechanistic model on the average of all samples
    """
    # construct problem based on petab for pypesto subproblem
    pp = importer.petab_problem
    pp.parameter_df[petab.ESTIMATE] = [
        not x.startswith(MODEL_FEATURE_PREFIX)
        and pp.parameter_df[petab.ESTIMATE][x]
        for x in pp.parameter_df.index
    ]
    pp.parameter_df.loc[
        pp.parameter_df[petab.ESTIMATE] == 0,
        [petab.OBJECTIVE_PRIOR_TYPE, petab.OBJECTIVE_PRIOR_PARAMETERS],
    ] = np.NaN

    # create fresh model from scratch since the petab imported one already
    # has the observables added and this might lead to issues.
    clean_model = problem.load_pysb()

    # subset measurements, conditions and parameters for specified sample
    df_train = pp.measurement_df.loc[
        pp.measurement_df[petab.PREEQUILIBRATION_CONDITION_ID].isin(samples), :
    ]

    df_train[petab.SIMULATION_CONDITION_ID] = df_train[
        petab.SIMULATION_CONDITION_ID
    ].apply(lambda x: x.split("__")[1])

    df_train[petab.PREEQUILIBRATION_CONDITION_ID] = "baseline"

    cdf = pp.condition_df.loc[
        [name.startswith(samples[0]) for name in pp.condition_df.index], :
    ]
    cdf.index = [
        name.replace(samples[0] + "__", "").replace(samples[0], "baseline")
        for name in cdf.index
    ]
    cdf.drop(
        columns=[x for x in cdf.columns if x.startswith(MODEL_FEATURE_PREFIX)],
        inplace=True,
    )
    spars = (
        set(
            e
            for t in df_train[petab.OBSERVABLE_PARAMETERS].apply(
                lambda x: x.split(";")
            )
            for e in t
        )
        if petab.OBSERVABLE_PARAMETERS in df_train
        else {}
    )
    pdf = pp.parameter_df[
        [
            (
                not name.startswith(MODEL_FEATURE_PREFIX)
                and (
                    (
                        not name.endswith("_scale")
                        and not name.endswith("offset")
                    )
                    or name in spars
                )
            )
            or name.endswith(samples[0])
            for name in pp.parameter_df.index
        ]
    ]
    pdf.index = [name.replace("__" + samples[0], "") for name in pdf.index]

    return PetabImporterPysb(
        petab.Problem(
            parameter_df=pdf,
            observable_df=pp.observable_df,
            measurement_df=df_train,
            condition_df=cdf,
            model=PySBModel(
                Model(base=clean_model, name=pp.model.model_id),
                pp.model.model_id,
            ),
        ),
        output_folder=str(
            problem.amici_dir / f"{pp.model.model_id}_{dataset}_petab"
        ),
    )


def generate_cross_sample_pretraining_problem(
    model: DeepMechanisticModel, problem: Problem
) -> pypesto.Problem:
    """
    Creates a pypesto problem that can be used to train population
    parameters as well as individual sample specific parameters. This is
    effectively just the unconstrained petab subproblem.
    """
    x_names = model.x_names[model.n_encode_weights :]

    obj = JaxObjective(
        model.pypesto_subproblem.objective,
        model.inflate,
        x_names=x_names,
    )

    pypesto_problem = pypesto.Problem(
        objective=obj,
        x_names=x_names,
        lb=[
            problem.bounds[xname.split("_")[-1]][0] - 2.0 for xname in x_names
        ],
        ub=[
            problem.bounds[xname.split("_")[-1]][1] + 2.0 for xname in x_names
        ],
    )
    problem.apply_objective_settings(pypesto_problem.objective)
    return pypesto_problem
