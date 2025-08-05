import logging
import re
from pathlib import Path
from typing import Tuple

import amici
import amici.petab.conditions
import amici.pysb_import
import pandas as pd
import pypesto.objective
import pypesto.objective.jax
import pysb
import pysb.export
import sympy as sp

from dmm.problem import ParameterBounds, Problem
from dmm.training_helper_funcs import Chi2Objective

from .data import load_dream_data

base_dir = Path(__file__).parents[0]
pysb_dir = base_dir / "pysb"
pathway_dir = base_dir

logger = logging.getLogger("cytof_problem")

BOUNDS = ParameterBounds(
    kdeg=(-4, 2, "log10"),  # [1/[t]]
    eq=(-4, 4, "log10"),  # [[c]]
    kcat=(-3, 3, "log10"),  # [1/([t]*[c])]
    kr=(-6, 6, "log10"),  # [-]
    scale=(-2, 4, "log10"),  # [1/[c]]
    offset=(-4, 4, "log10"),  # [[c]]
    koff=(-3, 2, "log10"),  # [1/[t]]
    kd=(-10, 3, "log10"),  # [[c]]
    kw=(-4, 4, "log10"),  # [1/[c]]
    bact=(-4, 4, "log10"),
)


class CytofProblem(Problem):
    @property
    def bounds(self) -> ParameterBounds:
        return BOUNDS

    def load_amici(
        self,
        model: pysb.Model,
        amici_dir: Path,
        force_compile: bool = True,
        add_observables: bool = False,
        name_suffix: str = "",
    ) -> Tuple[amici.AmiciModel, amici.AmiciSolver]:
        outdir = amici_dir / (model.name + name_suffix)

        # extend observables
        if add_observables:
            for obs in model.observables:
                if re.match(r"[p|t][A-Z0-9]+[SYT0-9_]*", obs.name):
                    offset = pysb.Parameter(obs.name + "_offset", 0.0)
                    scale = pysb.Parameter(obs.name + "_scale", 1.0)
                    pysb.Expression(
                        obs.name + "_obs", sp.log(scale * obs + offset)
                    )

        if (
            force_compile
            or not (outdir / model.name / (model.name + ".py")).exists()
        ):
            outdir.mkdir(exist_ok=True, parents=True)
            amici.pysb_import.pysb2amici(
                model,
                outdir,
                verbose=logging.DEBUG,
                observables=[
                    expr.name
                    for expr in model.expressions
                    if expr.name.endswith("_obs")
                    and not expr.name.startswith("free_")
                ],
                constant_parameters=[
                    par.name
                    for par in model.parameters
                    if par.name.endswith("_0")
                ],
            )

        model_module = amici.import_model_module(model.name, outdir)

        amici_model = model_module.getModel()
        solver = amici_model.getSolver()

        self.apply_solver_settings(solver)

        return amici_model, solver

    def load_pysb(self) -> pysb.Model:
        pathway = self.model_name.split("__")[0]
        model_file = pathway_dir / f"pw_{pathway}.py"
        if not model_file.exists():
            raise ValueError(
                f"{pathway} is not a valid pathway name for this problem class. Please specify"
                f" a valid name via the `pathway_name` keyword argument when instantiating the problem."
            )
        logger.debug(f"loading pathway from {model_file}")
        model = amici.pysb_import.pysb_model_from_path(model_file)

        pysb_dir.mkdir(exist_ok=True, parents=True)
        pysb_file = pysb_dir / f"{model.name}.py"
        with open(pysb_file, "w") as file:
            logger.debug(f"writing pysb model to {pysb_file}")
            file.write(pysb.export.export(model, "pysb_flat"))

        model.name = self.model_name

        return model

    def apply_solver_settings(self, solver):
        solver.setMaxSteps(int(2e4))
        solver.setNewtonMaxSteps(int(100))
        solver.setAbsoluteTolerance(1e-10)
        solver.setRelativeTolerance(1e-10)
        solver.setAbsoluteToleranceSteadyState(1e-6)
        solver.setRelativeToleranceSteadyState(1e-6)
        solver.setNewtonStepSteadyStateCheck(True)

    def apply_objective_settings(self, objective, n_threads: int = 1):
        amiobjective = None
        if isinstance(objective, pypesto.objective.AmiciObjective):
            amiobjective = objective
        elif isinstance(objective, pypesto.objective.AggregatedObjective):
            amiobjective = next(
                (
                    obj
                    for obj in objective._objectives
                    if isinstance(obj, pypesto.objective.AmiciObjective)
                ),
                None,
            )
        elif isinstance(objective, pypesto.objective.jax.JaxObjective):
            base_objective = objective.base_objective
            if isinstance(
                base_objective, pypesto.objective.AggregatedObjective
            ):
                amiobjective = next(
                    (
                        obj
                        for obj in base_objective._objectives
                        if isinstance(obj, pypesto.objective.AmiciObjective)
                    ),
                    None,
                )
            elif isinstance(base_objective, pypesto.objective.AmiciObjective):
                amiobjective = base_objective
        elif isinstance(objective, Chi2Objective):
            amiobjective = objective.base_objective

        if amiobjective is None:
            logger.warning(
                "could not identify suitable objective function, settings were not applied."
            )
            return

        amiobjective.guess_steadystate = False
        amiobjective.n_threads = n_threads
        self.apply_solver_settings(amiobjective.amici_solver)
        amiobjective.amici_model.setSteadyStateSensitivityMode(
            amici.SteadyStateSensitivityMode.newtonOnly
        )
        amiobjective.amici_model.setSteadyStateComputationMode(
            amici.SteadyStateComputationMode.integrationOnly
        )

        for e in amiobjective.edatas:
            e.reinitializeFixedParameterInitialStates = True
            fp = list(e.fixedParameters)
            if "EGF_0" in amiobjective.amici_model.getFixedParameterIds():
                fp[
                    amiobjective.amici_model.getFixedParameterIds().index(
                        "EGF_0"
                    )
                ] = 0
            e.fixedParametersPresimulation = tuple(fp)
            e.t_presim = 15

    @staticmethod
    def load_preprocess_petab_tables(
        model,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return load_dream_data(model)

    @property
    def base_dir(self) -> Path:
        return base_dir
