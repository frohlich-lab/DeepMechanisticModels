"""
Per sample pretraining.
"""

from pathlib import Path

import fides
import fire
import numpy as np
import pypesto
from pypesto.optimize import FidesOptimizer

from common import (
    PER_SAMPLE_OUTFILE_PARS,
    PER_SAMPLE_OUTFILE_RESULTS,
    fig_dir,
    pretrain_dir,
)
from cytof.problem import CytofProblem
from dmm.petab_subproblem import load_petab
from dmm.pretraining import (
    generate_per_sample_pretraining_problems,
    pretrain,
    store_and_plot_pretraining,
)
from util import Conf, load_petab_base_files

np.random.seed(0)

conf = fire.Fire(Conf)

problem = CytofProblem(conf.model)

petab_base_importer = load_petab(
    problem, conf.data, 0.0, **load_petab_base_files(conf)
)

importer = generate_per_sample_pretraining_problems(
    petab_base_importer,
    problem,
    conf.data,
    conf.sample,
)

outdir = pretrain_dir / conf.model / conf.data
figdir = fig_dir / conf.model / conf.data / "pretraining_sample"
pypesto_problem = importer.create_problem()
model = importer.create_model()

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
    pypesto_problem,
    pypesto.startpoint.UniformStartpoints(check_fval=True, check_grad=True),
    10,
    optimizer,
)
results_file = Path(PER_SAMPLE_OUTFILE_RESULTS.format(**conf.__dict__))
pars_file = Path(PER_SAMPLE_OUTFILE_PARS.format(**conf.__dict__))
store_and_plot_pretraining(result, pfile=pars_file, rfile=results_file)
