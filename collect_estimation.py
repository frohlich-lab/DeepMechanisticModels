import sys
import os
import re
from pathlib import Path

from pypesto.store import OptimizationResultHDF5Reader, OptimizationResultHDF5Writer
from mEncoder.training import create_pypesto_problem
from common import COLLECTED_TRAINING_RESULTS, TRAINING_OUTFILE_RESULTS
from util import load_from_argv

import pypesto.visualize

conf, mae, problem = load_from_argv(sys.argv)
pypesto_problem = create_pypesto_problem(mae, problem)

outfile = COLLECTED_TRAINING_RESULTS.format(**conf.__dict__)
indir = Path(TRAINING_OUTFILE_RESULTS.format(**conf.__dict__)).parent
inpattern = TRAINING_OUTFILE_RESULTS.format(job='[0-9]+').format(**conf.__dict__).replace('.', '\\.')

optimizer_results = []
for file in os.listdir(indir):
    if not re.match(inpattern, file):
        continue
    reader = OptimizationResultHDF5Reader(str(indir / str(file)))
    starts = reader.read().optimize_result.list
    for start in starts:
        start["hess"] = None

    optimizer_results.extend(starts)

print(
    sorted([r["fval"] for r in optimizer_results])[0: min(5, len(optimizer_results))]
)

for istart, start in enumerate(optimizer_results):
    start["id"] = str(istart)

result = pypesto.Result(problem=problem)
optimize_result = pypesto.OptimizeResult()
optimize_result.list = optimizer_results
optimize_result.sort()

result.optimize_result = optimize_result

writer = OptimizationResultHDF5Writer(outfile)
writer.write(result, overwrite=True)
