from pypesto import Problem, Result
from amici.petab_import import PysbPetabProblem
from pypesto.petab.pysb_importer import PetabImporterPysb
from pypesto.optimize import OptimizeOptions, minimize
from pypesto.store import OptimizationResultHDF5Writer
from pypesto.visualize import waterfall, parameters
from pypesto.objective.aesara import AesaraObjective

import petab
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import Callable
from pathlib import Path
from pysb import Model

from .autoencoder import MechanisticAutoEncoder
from . import (
    MODEL_FEATURE_PREFIX, load_pathway, parameter_boundaries_scales,
    basedir, apply_objective_settings
)


def generate_per_sample_pretraining_problems(
    importer: PetabImporterPysb,
    model: str,
    dataset: str,
    sample: str
) -> PetabImporterPysb:
    """
    Creates a pypesto problem that can be used to train the
    mechanistic model individually on every sample

    :param ae:
        Mechanistic autoencoder that will be pretrained

    :returns:
        Dict of pypesto problems. Keys are sample names.
    """
    # construct problem based on petab for pypesto subproblem
    pp = importer.petab_problem
    pp.parameter_df[petab.ESTIMATE] = [
        not x.startswith(MODEL_FEATURE_PREFIX) and
        pp.parameter_df[petab.ESTIMATE][x]
        for x in pp.parameter_df.index
    ]
    pp.parameter_df.loc[
        pp.parameter_df[petab.ESTIMATE] == 0,
        [petab.OBJECTIVE_PRIOR_TYPE, petab.OBJECTIVE_PRIOR_PARAMETERS]
    ] = np.NaN

    # create fresh model from scratch since the petab imported one already
    # has the observables added and this might lead to issues.
    clean_model = load_pathway('pw_' + model)

    # subset measurements, conditions and parameters for specified sample
    mdf = pp.measurement_df[
        pp.measurement_df[petab.PREEQUILIBRATION_CONDITION_ID]
        == sample
    ]
    cdf = pp.condition_df[[
        name.startswith(sample)
        for name in pp.condition_df.index
    ]]
    spars = set(
        e
        for t in mdf[petab.OBSERVABLE_PARAMETERS].apply(lambda x: x.split(';'))
        for e in t
    ) if petab.OBSERVABLE_PARAMETERS in mdf else {}
    pdf = pp.parameter_df[[
        (not name.startswith(MODEL_FEATURE_PREFIX) and (
            (not name.endswith('_scale') and not name.endswith('offset'))
            or name in spars
         ))
        or name.endswith(sample)
        for name in pp.parameter_df.index
    ]]

    return PetabImporterPysb(PysbPetabProblem(
        parameter_df=pdf,
        observable_df=pp.observable_df,
        measurement_df=mdf,
        condition_df=cdf,
        pysb_model=Model(base=clean_model, name=pp.pysb_model.name),
    ), output_folder=str(
        basedir / 'amici_models' / f'{pp.pysb_model.name}_{dataset}_petab'
    ))


def generate_cross_sample_pretraining_problem(
        ae: MechanisticAutoEncoder
) -> Problem:
    """
    Creates a pypesto problem that can be used to train population
    parameters as well as individual sample specific parameters. This is
    effectively just the unconstrained petab subproblem.

    :param ae:
        Mechanistic autoencoder that will be pretrained

    :returns:
        pypesto Problem
    """
    x_names = ae.x_names[ae.n_encode_weights:]

    obj = AesaraObjective(
        ae.pypesto_subproblem.objective, ae.x_embedding,
        ae.embedding_model_pars, x_names=x_names
    )

    problem = Problem(
        objective=obj,
        x_names=x_names,
        lb=[parameter_boundaries_scales[xname.split('_')[-1]][0] - 2.0
            for xname in x_names],
        ub=[parameter_boundaries_scales[xname.split('_')[-1]][1] + 2.0
            for xname in x_names],
    )
    apply_objective_settings(problem, ae.pathway_name)
    return problem


def pretrain(problem: Problem, startpoint_method: Callable, nstarts: int,
             optimizer, engine=None) -> Result:
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

    return minimize(
        problem, optimizer, n_starts=nstarts, options=optimize_options,
        startpoint_method=startpoint_method,
        engine=engine, filename=None
    )


def store_and_plot_pretraining(result: Result, outdir: Path, prefix: str,
                               plot_waterfall: bool = True):
    """
    Store optimziation results in HDF5 as well as csv for later reuse. Also
    saves some visualization for debugging purposes.

    :param result:
        result from pretraining

    :param prefix:
        prefix for file names that can be used to differentiate between
        different pretraining stages as well as models/datasets.
    """
    # store full results as hdf5
    rfile = outdir / (prefix + '.hdf5')
    writer = OptimizationResultHDF5Writer(str(rfile))
    writer.write(result, overwrite=True)

    # store parameter values, this will be used in subsequent steps
    parameter_df = pd.DataFrame(
        [r for r in result.optimize_result.get_for_key('x')
         if r is not None],
        columns=result.problem.x_names
    )
    parameter_df.to_csv(outdir / (prefix + '.csv'))

    # do plotting
    if plot_waterfall:
        waterfall(result, scale_y='log10', offset_y=0.0)
        plt.tight_layout()
        plt.savefig(outdir / (prefix + '_waterfall.pdf'))

    if result.problem.dim_full < 2e3:
        parameters(result)
        plt.tight_layout()
        plt.savefig(outdir / (prefix + '_parameters.pdf'))
