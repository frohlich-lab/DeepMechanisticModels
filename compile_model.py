import sys

from mEncoder.petab_subproblem import load_petab
from mEncoder import data_dir

MODEL = sys.argv[1]
DATA = sys.argv[2]

importer = load_petab((
    data_dir / f'{DATA}__{MODEL}__measurements.tsv',
    data_dir / f'{DATA}__{MODEL}__conditions.tsv',
    data_dir / f'{DATA}__{MODEL}__observables.tsv',
), 'pw_' + MODEL, 1.0)
importer.create_model(force_compile=True)

