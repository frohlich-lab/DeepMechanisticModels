"""
Per sample pretraining.
"""
import fides
import fire
import matplotlib.pyplot as plt
import numpy as np
from pypesto.optimize import FidesOptimizer
from pypesto.visualize import parameters, waterfall

from common import (
    Conf,
    PER_SAMPLE_OUTFILE_PARS,
    PER_SAMPLE_OUTFILE_RESULTS,
    Wildcards,
    fig_dir,
    pretrain_dir,
    training_samples,
)
from cytof.problem import CytofProblem
from dmm.petab_subproblem import load_petab
from dmm.pretraining import (
    generate_average_pretraining_problem,
    pretrain,
    store_and_plot_pretraining,
)
from pathlib import Path
from util import load_petab_base_files

np.random.seed(0)

conf = fire.Fire(Conf)

problem = CytofProblem(conf.model)

petab_base_importer = load_petab(
    problem=problem,
    dataset=conf.data,
    **load_petab_base_files(conf),
)

importer = generate_average_pretraining_problem(
    importer=petab_base_importer,
    problem=problem,
    dataset=conf.data,
    samples=training_samples(Wildcards(conf.data, conf.samples)),
)

outdir = pretrain_dir / conf.model / conf.data
figdir = fig_dir / conf.model / conf.data / "pretraining_sample"
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
    10,
    optimizer,
)

results_file = Path(
    PER_SAMPLE_OUTFILE_RESULTS.format(
        model=conf.model,
        data=conf.data,
        sample=f"model_average_{conf.samples}",
    )
)
pars_file = Path(
    PER_SAMPLE_OUTFILE_PARS.format(
        model=conf.model,
        data=conf.data,
        sample=f"model_average_{conf.samples}",
    )
)
store_and_plot_pretraining(result, pfile=pars_file, rfile=results_file)
parameters(result)
plt.tight_layout()
plt.savefig("parameters_avg.pdf")

waterfall(result)
plt.tight_layout()
plt.savefig("waterfall_avg.pdf")
