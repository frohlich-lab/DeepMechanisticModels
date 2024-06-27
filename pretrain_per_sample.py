"""
Per sample pretraining.
"""

import fides
import fire
import numpy as np
import pypesto

from common import (
    PER_SAMPLE_OUTFILE_PARS,
    PER_SAMPLE_OUTFILE_RESULTS,
    fig_dir,
    pretrain_dir,
)
from cytof.problem import CytofProblem
from dmm.config_options import Conf
from dmm.petab_subproblem import load_petab
from dmm.pretraining import (
    generate_per_sample_pretraining_problems,
    pretrain,
    store_and_plot_pretraining,
)
from pathlib import Path
from pypesto.optimize import FidesOptimizer
from util import load_petab_base_files

np.random.seed(0)

conf = fire.Fire(Conf)

problem = CytofProblem(conf.model)

petab_base_importer = load_petab(
    problem=problem, dataset=conf.data, **load_petab_base_files(conf)
)

importer = generate_per_sample_pretraining_problems(
    importer=petab_base_importer,
    problem=problem,
    dataset=conf.data,
    sample=conf.sample,
)

outdir = pretrain_dir / conf.model / conf.data
figdir = fig_dir / conf.model / conf.data / "pretraining_sample"
pypesto_problem = importer.create_problem()

problem.apply_objective_settings(pypesto_problem.objective)

optimizer = FidesOptimizer(
    options={
        fides.Options.FATOL: 0.0,
        fides.Options.XTOL: 1e-8,
        fides.Options.MAXTIME: 7200,
        fides.Options.MAXITER: 100,
    }
)
result = pretrain(
    problem=pypesto_problem,
    startpoint_method=pypesto.startpoint.UniformStartpoints(
        check_fval=True, check_grad=True
    ),
    nstarts=10, # multistarts for pretraining (hard-coded)
    optimizer=optimizer,
)
results_file = Path(PER_SAMPLE_OUTFILE_RESULTS.format(**conf.__dict__))
pars_file = Path(PER_SAMPLE_OUTFILE_PARS.format(**conf.__dict__))
store_and_plot_pretraining(result, pfile=pars_file, rfile=results_file)
