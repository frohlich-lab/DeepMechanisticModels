"""
Per sample pretraining.
"""

import sys
from pathlib import Path

import fides
import matplotlib.pyplot as plt
import numpy as np
import pypesto
from pypesto.optimize import FidesOptimizer
from pypesto.visualize import parameters, waterfall

from common import (
    PER_SAMPLE_OUTFILE_PARS,
    PER_SAMPLE_OUTFILE_RESULTS,
    Wildcards,
    fig_dir,
    pretrain_dir,
    training_samples,
)
from cytof.problem import CytofProblem
from mEncoder.petab_subproblem import load_petab
from mEncoder.pretraining import (
    generate_average_pretraining_problem,
    pretrain,
    store_and_plot_pretraining,
)
from util import load_petab_base_files

np.random.seed(0)

MODEL = sys.argv[1]
DATA = sys.argv[2]
SAMPLES = sys.argv[3]

problem = CytofProblem(MODEL)

petab_base_importer = load_petab(
    problem,
    DATA,
    0.0,
    **load_petab_base_files(MODEL, DATA),
)

importer = generate_average_pretraining_problem(
    petab_base_importer,
    problem,
    DATA,
    training_samples(Wildcards(DATA, SAMPLES)),
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
        model=MODEL, data=DATA, sample=f"model_average_{SAMPLES}"
    )
)
pars_file = Path(
    PER_SAMPLE_OUTFILE_PARS.format(
        model=MODEL, data=DATA, sample=f"model_average_{SAMPLES}"
    )
)
store_and_plot_pretraining(result, pfile=pars_file, rfile=results_file)
parameters(result)
plt.tight_layout()
plt.savefig(f"parameters_avg.pdf")

waterfall(result)
plt.tight_layout()
plt.savefig(f"waterfall_avg.pdf")
