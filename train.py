import sys

from mEncoder.training import train
from common import CROSS_SAMPLE_OUTFILE_PARS, TRAINING_OUTFILE_RESULTS
from util import load_from_argv

from pypesto.store import OptimizationResultHDF5Writer
from pathlib import Path

conf, mae, problem = load_from_argv(sys.argv, dataset='train')

pretraining_file = Path(CROSS_SAMPLE_OUTFILE_PARS.format(**conf.__dict__))

result = train(mae, problem, pretraining_file,  n_starts=1, seed=conf.job)
rfile = Path(TRAINING_OUTFILE_RESULTS.format(**conf.__dict__))
rfile.parent.mkdir(exist_ok=True, parents=True)
writer = OptimizationResultHDF5Writer(str(rfile))
writer.write(result)
