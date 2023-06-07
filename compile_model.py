import fire

from cytof.problem import CytofProblem
from dmm.petab_subproblem import load_petab
from util import Conf, load_petab_base_files

conf = fire.Fire(Conf)

problem = CytofProblem(conf.model)

importer = load_petab(
    problem,
    conf.data,
    0.0,
    **load_petab_base_files(conf),
)

importer.create_model(force_compile=True)
