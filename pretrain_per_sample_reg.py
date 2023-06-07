"""
Per sample pretraining.
"""

import sys
from pathlib import Path

import fides
import numpy as np
import pandas as pd
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
    generate_per_sample_reg_pretraining_problem,
    pretrain,
    store_and_plot_pretraining,
)
from util import load_petab_base_files

np.random.seed(0)

MODEL = sys.argv[1]
DATA = sys.argv[2]
SAMPLES = sys.argv[3]
ALPHA = float(sys.argv[4])
SAMPLE = sys.argv[5]

problem = CytofProblem(MODEL)

petab_base_importer = load_petab(
    problem,
    DATA,
    0.0,
    **load_petab_base_files(MODEL, DATA),
)

pars_file = Path(
    PER_SAMPLE_OUTFILE_PARS.format(model=MODEL, data=DATA, sample="average")
)
avg_pars = pd.read_csv(pars_file)

importer = generate_per_sample_reg_pretraining_problem(
    petab_base_importer,
    problem,
    avg_pars,
    DATA,
    SAMPLE,
    ALPHA,
)

outdir = pretrain_dir / MODEL / DATA
figdir = fig_dir / MODEL / DATA / "pretraining_sample"
pypesto_problem = importer.create_problem()
model = importer.create_model()

problem.apply_objective_settings(pypesto_problem.objective)

optimizer = FidesOptimizer(
    options={
        fides.Options.FATOL: 1e-8,
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
results_file = Path(
    PER_SAMPLE_OUTFILE_RESULTS.format(
        model=MODEL, data=DATA, sample=SAMPLE + f"__{SAMPLES}__{ALPHA}"
    )
)
pars_file = Path(
    PER_SAMPLE_OUTFILE_PARS.format(
        model=MODEL, data=DATA, sample=SAMPLE + f"__{SAMPLES}__{ALPHA}"
    )
)
store_and_plot_pretraining(result, pfile=pars_file, rfile=results_file)
