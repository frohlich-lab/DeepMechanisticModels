import fire

from cytof.problem import CytofProblem
from dmm.config_options import Conf
from dmm.petab_subproblem import load_petab
from util import load_petab_base_files

conf = fire.Fire(Conf)

problem = CytofProblem(conf.model)

importer = load_petab(
    problem,
    conf.data,
    **load_petab_base_files(conf),
)
creator = importer.create_objective_creator()
creator.create_model(force_compile=True)
