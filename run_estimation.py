import sys

from mEncoder.autoencoder import MechanisticAutoEncoder
from mEncoder.training import train
from process_data import training_samples, Wildcards
from mEncoder import results_dir, data_dir, ESTIMATION_OUTFILE_TEMP

from pypesto.store import OptimizationResultHDF5Writer

MODEL = sys.argv[1]
DATA = sys.argv[2]
CONTEXT = sys.argv[3]
SAMPLES = sys.argv[4]
N_HIDDEN = int(sys.argv[5])
ALPHA = float(sys.argv[6])
JOB = int(sys.argv[7])

mae = MechanisticAutoEncoder(
    N_HIDDEN,
    (
        data_dir / f"{DATA}__{MODEL}__measurements.tsv",
        data_dir / f"{DATA}__{MODEL}__conditions.tsv",
        data_dir / f"{DATA}__{MODEL}__observables.tsv",
    ),
    pathway_name=MODEL,
    samples=training_samples(Wildcards(DATA, SAMPLES)),
    contextualization=CONTEXT,
    l1reg=ALPHA,
    n_threads=4,
)

result = train(mae, SAMPLES, n_starts=1, seed=JOB, context=CONTEXT)
outdir = results_dir / MODEL / DATA
outfile = outdir / ESTIMATION_OUTFILE_TEMP.format(
    context=CONTEXT, samples=SAMPLES, n_hidden=N_HIDDEN, alpha=ALPHA, job=JOB
)
outdir.mkdir(exist_ok=True, parents=True)
writer = OptimizationResultHDF5Writer(outfile)
writer.write(result)
