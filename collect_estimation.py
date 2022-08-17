import sys
import os

from pypesto.store import (
    OptimizationResultHDF5Reader, OptimizationResultHDF5Writer
)
from mEncoder.autoencoder import MechanisticAutoEncoder
from mEncoder.training import create_pypesto_problem
from mEncoder import (
    results_dir, data_dir, COLLECTED_ESTIMATION_OUTFILE_TEMP,
    ESTIMATION_OUTFILE_TEMP
)
from process_data import training_samples, Wildcards

import pypesto.visualize

MODEL = sys.argv[1]
DATA = sys.argv[2]
SAMPLES = sys.argv[3]
N_HIDDEN = int(sys.argv[4])
ALPHA = float(sys.argv[5])

mae = MechanisticAutoEncoder(
    N_HIDDEN, (
        data_dir / f'{DATA}__{MODEL}__measurements.tsv',
        data_dir / f'{DATA}__{MODEL}__conditions.tsv',
        data_dir / f'{DATA}__{MODEL}__observables.tsv',
    ),
    pathway_name=MODEL, samples=training_samples(Wildcards(DATA, SAMPLES)),
    par_modulation_scale=ALPHA
)

problem = create_pypesto_problem(mae)

optimizer_results = []

result_path = results_dir / MODEL / DATA
result_files = os.listdir(result_path)

outfile = result_path / COLLECTED_ESTIMATION_OUTFILE_TEMP.format(
    samples=SAMPLES, n_hidden=N_HIDDEN, alpha=ALPHA
)

prefix = '__'.join(ESTIMATION_OUTFILE_TEMP.format(
   samples=SAMPLES, n_hidden=N_HIDDEN, alpha=ALPHA, job='JOB'
).split('__')[:-1])

for file in result_files:
    if not file.startswith(prefix) or \
            not file.endswith('.hdf5') or file == outfile:
        continue
    reader = OptimizationResultHDF5Reader(str(result_path / file))
    starts = reader.read().optimize_result.list
    for start in starts:
        start['hess'] = None

    optimizer_results.extend(starts)

print(sorted([
    r['fval']
    for r in optimizer_results
])[0:min(5, len(optimizer_results))])

for istart, start in enumerate(optimizer_results):
    start['id'] = str(istart)

result = pypesto.Result(
    problem=problem
)
optimize_result = pypesto.OptimizeResult()
optimize_result.list = optimizer_results
optimize_result.sort()

result.optimize_result = optimize_result

writer = OptimizationResultHDF5Writer(outfile)
writer.write(result, overwrite=True)
