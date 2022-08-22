import amici.pysb_import
import logging
import re
import pysb
import sys
import sympy as sp
import amici
import pypesto
import pysb.export
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from pathlib import Path

basedir: Path = Path(__file__).resolve().parents[1]
fig_dir = basedir / 'figures'
results_dir = basedir / 'results'
data_dir = basedir / 'data'
pretrain_dir = basedir / 'pretraining'


def load_pathway(pathway_name: str) -> pysb.Model:
    model_file = basedir / 'pathways' / (pathway_name + '.py')
    sys.path.insert(0, str(basedir / 'pathways'))
    model = amici.pysb_import.pysb_model_from_path(model_file)

    with open(basedir / 'pysb_models' / (model.name + '.py'), 'w') as file:
        file.write(pysb.export.export(model, 'pysb_flat'))

    return model


def load_model(pathway_name: str,
               force_compile: bool = True,
               add_observables: bool = False) -> Tuple[amici.AmiciModel,
                                                       amici.AmiciSolver]:

    model = load_pathway(pathway_name)
    outdir = basedir / 'amici_models' / model.name

    # extend observables
    if add_observables:
        for obs in model.observables:
            if re.match(r'[p|t][A-Z0-9]+[SYT0-9_]*', obs.name):
                offset = pysb.Parameter(obs.name + '_offset', 0.0)
                scale = pysb.Parameter(obs.name + '_scale', 1.0)
                pysb.Expression(obs.name + '_obs',
                                sp.log(scale * obs + offset))

    if force_compile or \
            not (outdir / model.name / (model.name + '.py')).exists():
        outdir.makedir(exist_ok=True, parents=True)
        amici.pysb_import.pysb2amici(model,
                                     outdir,
                                     verbose=logging.DEBUG,
                                     observables=[
                                         expr.name
                                         for expr in model.expressions
                                         if expr.name.endswith('_obs')
                                     ],
                                     constant_parameters=[
                                         par.name
                                         for par in model.parameters
                                         if par.name.endswith('_0')
                                     ])

    model_module = amici.import_model_module(model.name, outdir)

    amici_model = model_module.getModel()
    solver = amici_model.getSolver()

    apply_solver_settings(solver)

    return amici_model, solver


def plot_and_save_fig(filename: str, figdir: Optional[Path] = None):
    if figdir is None:
        figdir = figdir
    plt.tight_layout()
    figdir.mkdir(exist_ok=True, parents=True)
    if filename is not None:
        plt.savefig(figdir / filename)


def apply_solver_settings(solver):
    solver.setMaxSteps(int(1e5))
    solver.setAbsoluteTolerance(1e-12)
    solver.setRelativeTolerance(1e-12)
    solver.setAbsoluteToleranceSteadyState(1e-8)
    solver.setRelativeToleranceSteadyState(1e-8)


def apply_objective_settings(problem, pathway_name):
    if isinstance(problem.objective, pypesto.objective.AmiciObjective):
        amiobjective = problem.objective
    elif isinstance(problem.objective, pypesto.objective.AggregatedObjective):
        amiobjective = problem.objective._objectives[0]
    elif isinstance(problem.objective,
                    pypesto.objective.aesara.AesaraObjective):
        base_objective = problem.objective.base_objective
        if isinstance(base_objective, pypesto.objective.AggregatedObjective):
            amiobjective = base_objective._objectives[0]
        elif isinstance(base_objective, pypesto.objective.AmiciObjective):
            amiobjective = base_objective

    amiobjective.guess_steadystate = False
    apply_solver_settings(amiobjective.amici_solver)
    for e in amiobjective.edatas:
        e.reinitializeFixedParameterInitialStates = True
        if pathway_name.startswith('EGFR'):
            fp = list(e.fixedParameters)
            fp[amiobjective.amici_model.getFixedParameterIds().index(
               'EGF_0')] \
                = 0
            e.fixedParametersPresimulation = tuple(fp)
            e.t_presim = 15


parameter_boundaries_scales = {
    'kdeg': (-6, -1, 'log10'),       # [1/[t]]
    'eq': (-4, 4, 'log10'),          # [[c]]
    'kcat': (-4, 4, 'log10'),        # [1/([t]*[c])]
    'kr': (-3, 3, 'log10'),          # [-]
    'scale': (0, 10, 'log10'),         # [1/[c]]
    'offset': (-5, 5, 'log10'),        # [[c]]
    'weight': (-10, 10, 'lin'),        # [-]
    'koff': (-3, 2, 'log10'),        # [1/[t]]
    'kd':   (-3, 3, 'log10'),       # [[c]]
    'kw':   (-4, 3, 'log10'),        # [1/[c]]
}

MODEL_FEATURE_PREFIX = 'INPUT_'

PER_SAMPLE_OUTFILE_TEMP = '{sample}' + '.csv'

COLLECTED_ESTIMATION_OUTFILE_TEMP = '__'.join(['{samples}', '{n_hidden}',
                                               '{alpha}', 'full']) + '.hdf5'

ESTIMATION_OUTFILE_TEMP = '__'.join(['{samples}', '{n_hidden}',
                                     '{alpha}', '{job}']) + '.hdf5'
