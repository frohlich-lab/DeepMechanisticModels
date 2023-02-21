import pypesto
from amici.petab_import import PysbPetabProblem
from pypesto.petab.pysb_importer import PetabImporterPysb
from pypesto.optimize import OptimizeOptions, minimize
from pypesto.history import HistoryOptions
from pypesto.store import OptimizationResultHDF5Writer
from pypesto.visualize import waterfall, parameters
from pypesto.objective.jax import JaxObjective

import petab
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import Callable, List
from pathlib import Path
from pysb import Model

from .autoencoder import MechanisticAutoEncoder
from .problem import Problem
from . import MODEL_FEATURE_PREFIX


def generate_per_sample_pretraining_problems(
    importer: PetabImporterPysb,
    problem: Problem,
    dataset: str,
    sample: str
) -> PetabImporterPysb:
    """
    Creates a pypesto problem that can be used to train the
    mechanistic model individually on every sample
    """
    # construct problem based on petab for pypesto subproblem
    pp = importer.petab_problem
    pp.parameter_df[petab.ESTIMATE] = [
        not x.startswith(MODEL_FEATURE_PREFIX) and pp.parameter_df[petab.ESTIMATE][x]
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
    cdf = pp.condition_df[[name.startswith(sample) for name in pp.condition_df.index]]
    spars = (
        set(
            e
            for t in mdf[petab.OBSERVABLE_PARAMETERS].apply(lambda x: x.split(";"))
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
                    (not name.endswith(("_scale", "_offset")))
                    or name in spars
                )
            )
            or name.endswith(sample)
            for name in pp.parameter_df.index
        ]
    ]

    return PetabImporterPysb(
        PysbPetabProblem(
            parameter_df=pdf,
            observable_df=pp.observable_df,
            measurement_df=mdf,
            condition_df=cdf,
            pysb_model=Model(base=clean_model, name=pp.pysb_model.name),
        ),
        output_folder=str(
            problem.amici_dir / f"{pp.pysb_model.name}_{dataset}_petab"
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
        not x.startswith(MODEL_FEATURE_PREFIX) and pp.parameter_df[petab.ESTIMATE][x]
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
    cdf = pp.condition_df[[name.startswith(sample) for name in pp.condition_df.index]]
    spars = (
        set(
            e
            for t in mdf[petab.OBSERVABLE_PARAMETERS].apply(lambda x: x.split(";"))
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
                    (not name.endswith("_scale") and not name.endswith("offset"))
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
                pdf.loc[pname, petab.OBJECTIVE_PRIOR_PARAMETERS] = f"0.0;{1 / alpha}"
                pdf.loc[pname, petab.OBJECTIVE_PRIOR_TYPE] = petab.PARAMETER_SCALE_LAPLACE

        if pname.endswith(("_scale", "_offset")):
            continue

        if pname not in avg_pars.columns:  # includes inputs
            continue

        pdf.loc[pname, petab.NOMINAL_VALUE] = np.power(10, avg_pars.loc[0, pname])
        pdf.loc[pname, petab.ESTIMATE] = False

    return PetabImporterPysb(
        PysbPetabProblem(
            parameter_df=pdf,
            observable_df=pp.observable_df,
            measurement_df=mdf,
            condition_df=cdf,
            pysb_model=Model(base=clean_model, name=pp.pysb_model.name),
        ),
        output_folder=str(
            problem.amici_dir / f"{pp.pysb_model.name}_{dataset}_petab"
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
        not x.startswith(MODEL_FEATURE_PREFIX) and pp.parameter_df[petab.ESTIMATE][x]
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

    df_train[petab.SIMULATION_CONDITION_ID] = df_train[petab.SIMULATION_CONDITION_ID].apply(
        lambda x: x.split("__")[1]
    )

    df_train[petab.PREEQUILIBRATION_CONDITION_ID] = 'baseline'

    cdf = pp.condition_df.loc[[name.startswith(samples[0]) for name in pp.condition_df.index], :]
    cdf.index = [name.replace(samples[0] + '__', '').replace(samples[0], 'baseline') for name in cdf.index]
    cdf.drop(columns=[x for x in cdf.columns if x.startswith(MODEL_FEATURE_PREFIX)], inplace=True)
    spars = (
        set(
            e
            for t in df_train[petab.OBSERVABLE_PARAMETERS].apply(lambda x: x.split(";"))
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
                    (not name.endswith("_scale") and not name.endswith("offset"))
                    or name in spars
                )
            )
            or name.endswith(samples[0])
            for name in pp.parameter_df.index
        ]
    ]
    pdf.index = [name.replace('__' + samples[0], '') for name in pdf.index]

    return PetabImporterPysb(
        PysbPetabProblem(
            parameter_df=pdf,
            observable_df=pp.observable_df,
            measurement_df=df_train,
            condition_df=cdf,
            pysb_model=Model(base=clean_model, name=pp.pysb_model.name),
        ),
        output_folder=str(
            problem.amici_dir / f"{pp.pysb_model.name}_{dataset}_petab"
        ),
    )


def generate_cross_sample_pretraining_problem(ae: MechanisticAutoEncoder, problem: Problem) -> pypesto.Problem:
    """
    Creates a pypesto problem that can be used to train population
    parameters as well as individual sample specific parameters. This is
    effectively just the unconstrained petab subproblem.
    """
    x_names = ae.x_names[ae.n_encode_weights:]

    obj = JaxObjective(
        ae.pypesto_subproblem.objective,
        ae.inflate,
        x_names=x_names,
    )

    pypesto_problem = pypesto.Problem(
        objective=obj,
        x_names=x_names,
        lb=[
            problem.bounds[xname.split("_")[-1]][0] - 2.0
            for xname in x_names
        ],
        ub=[
            problem.bounds[xname.split("_")[-1]][1] + 2.0
            for xname in x_names
        ],
    )
    problem.apply_objective_settings(pypesto_problem.objective)
    return pypesto_problem


def pretrain(
    problem: Problem, startpoint_method: Callable, nstarts: int, optimizer, hfile=None, engine=None
) -> pypesto.Result:
    """
    Pretrain the provided problem via optimization.

    :param problem:
        problem that defines the pretraining optimization problem

    :param startpoint_method:
        function that generates the initial points for optimization. In most
        cases this uses results from previous pretraining steps.

    :param nstarts:
        number of local optimizations to perform
    """

    optimize_options = OptimizeOptions(allow_failed_starts=False)
    if hfile is not None:
        Path.unlink(hfile, missing_ok=True)
        history_options = HistoryOptions(
            trace_record=True,
            trace_record_grad=False,
            trace_record_hess=False,
            trace_record_res=False,
            trace_record_sres=False,
            storage_file=str(hfile),
        )
    else:
        history_options = None

    return minimize(
        problem,
        optimizer,
        n_starts=nstarts,
        options=optimize_options,
        startpoint_method=startpoint_method,
        history_options=history_options,
        engine=engine,
        filename=None,
    )


def store_and_plot_pretraining(
    result: pypesto.Result, rfile: Path, pfile: Path, plot_waterfall: bool = True
):
    """
    Store optimiziation results in HDF5 as well as csv for later reuse. Also
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
        plt.savefig(outdir / f'{run_name}_waterfall.pdf')
