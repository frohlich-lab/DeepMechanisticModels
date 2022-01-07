"""
Per sample pretraining.
"""

import sys
import os
import fides
import pypesto
import numpy as np

import amici.petab_objective

from pypesto.optimize import FidesOptimizer

from mEncoder.petab_subproblem import load_petab
from mEncoder.pretraining import (
    generate_per_sample_pretraining_problems, pretrain,
    store_and_plot_pretraining
)
from mEncoder.plotting import plot_single_sample
from mEncoder import (
    apply_objective_settings, pretrain_dir, PER_SAMPLE_OUTFILE_TEMP
)

np.random.seed(0)

MODEL = sys.argv[1]
DATA = sys.argv[2]
SAMPLE = sys.argv[3]

datafiles = (
    os.path.join('data', f'{DATA}__{MODEL}__measurements.tsv'),
    os.path.join('data', f'{DATA}__{MODEL}__conditions.tsv'),
    os.path.join('data', f'{DATA}__{MODEL}__observables.tsv'),
)

importer = generate_per_sample_pretraining_problems(
    load_petab(datafiles, 'pw_' + MODEL, 0.0, [SAMPLE]),
    MODEL, f'{DATA}__{MODEL}', SAMPLE
)
outdir = os.path.join(pretrain_dir, MODEL, DATA)
output_prefix = os.path.splitext(
    PER_SAMPLE_OUTFILE_TEMP.format(sample=SAMPLE)
)[0]
problem = importer.create_problem()
model = importer.create_model()
apply_objective_settings(problem, MODEL)

optimizer = FidesOptimizer(
    hessian_update=fides.HybridFixed(switch_iteration=75),
    options={
        fides.Options.FATOL: 1e-6,
        fides.Options.XTOL: 1e-8,
        fides.Options.MAXTIME: 7200,
        fides.Options.MAXITER: 1e3,
        fides.Options.SUBSPACE_DIM: fides.SubSpaceDim.TWO,
    }
)
result = pretrain(problem, pypesto.startpoint.uniform, 5,
                  optimizer, pypesto.engine.MultiThreadEngine(1))
store_and_plot_pretraining(result, outdir=outdir, prefix=output_prefix)

x = problem.get_reduced_vector(result.optimize_result.list[0]['x'],
                               problem.x_free_indices)
simulation = problem.objective(x, return_dict=True)
# Convert the simulation to PEtab format.
simulation_df = amici.petab_objective.rdatas_to_simulation_df(
    simulation['rdatas'],
    model=model,
    measurement_df=importer.petab_problem.measurement_df,
)
plot_single_sample(importer.petab_problem.measurement_df,
                   simulation_df,
                   outdir,
                   output_prefix)
