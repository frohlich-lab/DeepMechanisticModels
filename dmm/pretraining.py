from pathlib import Path
from typing import Callable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import petab.v1 as petab
import pypesto
from petab.v1.models.pysb_model import PySBModel
from pypesto.optimize import OptimizeOptions, minimize
from pypesto.petab import (
    PetabImporter,  # general PetabImporter compared to old PetabImporterPysb
)
from pypesto.startpoint import UniformStartpoints
from pypesto.store import OptimizationResultHDF5Writer
from pypesto.visualize import parameters, waterfall
from pysb import Model

from . import MODEL_FEATURE_PREFIX
from .problem import Problem


def generate_per_sample_pretraining_problems(
    importer: PetabImporter, problem: Problem, dataset: str, sample: str
) -> PetabImporter:  # general PetabImporter compared to old PetabImporterPysb
    """Creates a pypesto problem that can be used to train the
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
        {
            e
            for t in mdf[petab.OBSERVABLE_PARAMETERS].apply(
                lambda x: x.split(";")
            )
            for e in t
        }
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

    model_name = pp.model.model_id

    # general PetabImporter compared to old PetabImporterPysb
    return PetabImporter(
        petab.Problem(
            parameter_df=pdf,
            observable_df=pp.observable_df,
            measurement_df=mdf,
            condition_df=cdf,
            model=PySBModel(
                Model(base=clean_model, name=model_name),
                pp.model.model_id,
            ),
        ),
        model_name=model_name,
        output_folder=str(
            problem.amici_dir / f"{pp.model.model_id}_{dataset}_petab"
        ),
    )


def generate_per_sample_reg_pretraining_problem(
    importer: PetabImporter,  # general PetabImporter compared to old PetabImporterPysb
    problem: Problem,
    avg_pars: pd.DataFrame,
    dataset: str,
    sample: str,
    alpha: float = 0.0,
) -> PetabImporter:  # general PetabImporter compared to old PetabImporterPysb
    """Creates a pypesto problem that can be used to train the
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
        {
            e
            for t in mdf[petab.OBSERVABLE_PARAMETERS].apply(
                lambda x: x.split(";")
            )
            for e in t
        }
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

    model_name = pp.model.model_id

    # general PetabImporter compared to old PetabImporterPysb
    return PetabImporter(
        petab.Problem(
            parameter_df=pdf,
            observable_df=pp.observable_df,
            measurement_df=mdf,
            condition_df=cdf,
            model=PySBModel(
                Model(base=clean_model, name=model_name),
                pp.model.model_id,
            ),
        ),
        model_name=model_name,
        output_folder=str(
            problem.amici_dir / f"{pp.model.model_id}_{dataset}_petab"
        ),
    )


def generate_average_pretraining_problem(
    importer: PetabImporter,  # general PetabImporter compared to old PetabImporterPysb
    problem: Problem,
    dataset: str,
    samples: List[str],
) -> PetabImporter:  # general PetabImporter compared to old PetabImporterPysb
    """Creates a pypesto problem that can be used to train the mechanistic model on the average of all samples"""
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
    ].copy()

    can_be_aggregated = not any(
        pp.condition_df[s].nunique() > 1
        for s in pp.condition_df
        if s.endswith("_eq")
    )

    if can_be_aggregated:
        df_train[petab.SIMULATION_CONDITION_ID] = df_train[
            petab.SIMULATION_CONDITION_ID
        ].apply(lambda x: x.replace(x.split("__")[0], ""))
        df_train.loc[
            df_train[petab.SIMULATION_CONDITION_ID] == "",
            petab.SIMULATION_CONDITION_ID,
        ] = "baseline"

        df_train[petab.PREEQUILIBRATION_CONDITION_ID] = "baseline"

        cdf = pp.condition_df.loc[
            [name.startswith(samples[0]) for name in pp.condition_df.index], :
        ].copy()
        cdf.index = [
            name.replace(samples[0] + "__", "__").replace(
                samples[0], "baseline"
            )
            for name in cdf.index
        ]
    else:
        cdf = pp.condition_df.copy()

    cdf.drop(
        columns=[x for x in cdf.columns if x.startswith(MODEL_FEATURE_PREFIX)],
        inplace=True,
    )
    cdf.index.name = petab.CONDITION_ID
    spars = (
        {
            e
            for t in df_train[petab.OBSERVABLE_PARAMETERS].apply(
                lambda x: x.split(";")
            )
            for e in t
        }
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
    pdf.index.name = petab.PARAMETER_ID

    model_name = pp.model.model_id

    # general PetabImporter compared to old PetabImporterPysb
    return PetabImporter(
        petab.Problem(
            parameter_df=pdf,
            observable_df=pp.observable_df,
            measurement_df=df_train,
            condition_df=cdf,
            model=PySBModel(
                Model(base=clean_model, name=model_name),
                pp.model.model_id,
            ),
        ),
        model_name=model_name,
        output_folder=str(
            problem.amici_dir / f"{pp.model.model_id}_{dataset}_petab"
        ),
    )


# NOT IN USE (uses old model).
# def generate_cross_sample_pretraining_problem(
#     model: DeepMechanisticModel, problem: Problem
# ) -> pypesto.Problem:
#     """
#     Creates a pypesto problem that can be used to train population
#     parameters as well as individual sample specific parameters. This is
#     effectively just the unconstrained petab subproblem.
#     """
#     x_names = model.x_names[model.n_encode_weights :]
#
#     obj = JaxObjective(
#         model.pypesto_subproblem.objective,
#         model.inflate,
#         x_names=x_names,
#     )
#
#     pypesto_problem = pypesto.Problem(
#         objective=obj,
#         x_names=x_names,
#         lb=[
#             problem.bounds[xname.split("_")[-1]][0] - 2.0 for xname in x_names
#         ],
#         ub=[
#             problem.bounds[xname.split("_")[-1]][1] + 2.0 for xname in x_names
#         ],
#     )
#     problem.apply_objective_settings(pypesto_problem.objective)
#     return pypesto_problem


def pretrain(
    problem: Problem,
    nstarts: int,
    optimizer,
    startpoint_method: Optional[Callable] = None,
    engine=None,
) -> pypesto.Result:
    """Pretrain the provided problem via optimization.
    :param problem:
        problem that defines the pretraining optimization problem
    :param startpoint_method:
        function that generates the initial points for optimization. In most
        cases this uses results from previous pretraining steps.
    :param nstarts:
        number of local optimizations to perform
    """
    if startpoint_method is None:
        startpoint_method = UniformStartpoints(
            check_fval=True, check_grad=True
        )
    optimize_options = OptimizeOptions(allow_failed_starts=False)
    return minimize(
        problem,
        optimizer,
        n_starts=nstarts,
        options=optimize_options,
        startpoint_method=startpoint_method,
        engine=engine,
        filename=None,
    )


def store_and_plot_pretraining(
    result: pypesto.Result,
    rfile: Path,
    pfile: Path,
    plot_waterfall: bool = True,
):
    """Store optimiziation results in HDF5 as well as csv for later reuse. Also
    saves some visualization for debugging purposes.
    """
    # store full results as hdf5
    assert rfile.parent == pfile.parent
    outdir = rfile.parent
    outdir.mkdir(parents=True, exist_ok=True)
    run_name = rfile.stem
    writer = OptimizationResultHDF5Writer(str(rfile))
    writer.write(result, overwrite=True)
    # store parameter values, this will be used in subsequent steps
    parameter_df = pd.DataFrame(
        [r for r in result.optimize_result.get_for_key("x") if r is not None],
        columns=result.problem.x_names,
    )
    parameter_df.to_csv(pfile)
    # do plotting
    if plot_waterfall:
        waterfall(result, scale_y="log10", offset_y=0.0)
        plt.tight_layout()
        plt.savefig(outdir / f"{run_name}_waterfall.pdf")
        parameters(result)
        plt.tight_layout()
        plt.savefig(outdir / f"{run_name}_parameters.pdf")
