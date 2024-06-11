import fire
import matplotlib.pyplot as plt
import os
import pypesto.visualize
import re

from common import Conf, COLLECTED_TRAINING_RESULTS, TRAINING_OUTFILE_RESULTS
from dmm.initialisation import setup_models
from dmm.training_helper_funcs import create_pypesto_problem
from pathlib import Path
from pypesto.store import (
    OptimizationResultHDF5Reader,
    OptimizationResultHDF5Writer,
)
from pypesto.visualize import waterfall


conf = fire.Fire(Conf)

model, problem = setup_models(conf, "train")
pypesto_problem = create_pypesto_problem(model)

outfile = COLLECTED_TRAINING_RESULTS.format(**conf.__dict__)
indir = Path(TRAINING_OUTFILE_RESULTS.format(**conf.__dict__)).parent
inpattern = (
    str(
        Path(
            TRAINING_OUTFILE_RESULTS.replace("{job}", "([0-9]+)").format(
                **conf.__dict__
            )
        ).stem
    )
    + ".hdf5"
)

optimizer_results = []
for file in os.listdir(indir):
    if not str(file).endswith(".hdf5"):
        continue
    m = re.match(inpattern, str(file))
    if not m:
        continue

    # ignore previous results with higher n_starts
    if int(str(m.group(1))) >= conf.n_starts:
        print(f"ignoring old results from {file} (njobs={conf.n_starts})")
        continue

    print(f"loading results from {file}")
    reader = OptimizationResultHDF5Reader(str(indir / str(file)))
    starts = reader.read().optimize_result.list
    for start in starts:
        start["hess"] = None

    optimizer_results.extend(starts)

print(
    sorted([r["fval"] for r in optimizer_results])[
        0: min(5, len(optimizer_results))
    ]
)

for istart, start in enumerate(optimizer_results):
    start["id"] = str(istart)

result = pypesto.Result(problem=pypesto_problem)
optimize_result = pypesto.OptimizeResult()
optimize_result.list = optimizer_results
optimize_result.sort()

result.optimize_result = optimize_result

writer = OptimizationResultHDF5Writer(outfile)
writer.write(result, overwrite=True)


of = Path(outfile)
outdir = of.parent
run_name = of.stem

waterfall(result, scale_y="log10", offset_y=0.0)
plt.tight_layout()
plt.savefig(outdir / f"{run_name}_waterfall.pdf")

# parameters(result)
# plt.tight_layout()
# plt.savefig(outdir / f'{run_name}_parameters.pdf')
