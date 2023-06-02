import sys

from cytof.problem import CytofProblem
from mEncoder.petab_subproblem import load_petab
from util import load_petab_base_files

MODEL = sys.argv[1]
DATA = sys.argv[2]

problem = CytofProblem(MODEL)

importer = load_petab(
    problem,
    DATA,
    0.0,
    **load_petab_base_files(MODEL, DATA),
)

importer.create_model(force_compile=True)
