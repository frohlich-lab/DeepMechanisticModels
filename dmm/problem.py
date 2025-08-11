from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import amici
import pypesto
import pysb


@dataclass
class ParameterBounds:
    kdeg: Tuple[float, float, str]  # degradation rate
    eq: Tuple[float, float, str]  # equilibrium level
    kcat: Tuple[
        float, float, str
    ]  # baseline catalytic rate of the reverse reaction
    scale: Tuple[float, float, str]  # scaling factor for observables
    offset: Tuple[float, float, str]  # offset for observables
    kw: Tuple[
        float, float, str
    ]  # scale factor for effectors in the forward reaction
    bact: Tuple[
        float, float, str
    ]  # baseline activation rate of the forward reaction

    def __getitem__(self, item):
        return getattr(self, item)


@dataclass
class Problem(object):
    model_name: str

    @abstractmethod
    def load_pysb(self) -> pysb.Model:
        ...

    @abstractmethod
    def load_amici(
        self,
        model: pysb.Model,
        amici_dir: Path,
        force_compile: bool = True,
        add_observables: bool = False,
        name_suffix: str = "",
    ) -> amici.Model:
        ...

    @abstractmethod
    def apply_solver_settings(self, solver: amici.AmiciSolver):
        ...

    @abstractmethod
    def apply_objective_settings(
        self, objective: pypesto.ObjectiveBase, n_threads: int = 1
    ):
        ...

    @property
    @abstractmethod
    def bounds(self) -> ParameterBounds:
        ...

    @property
    @abstractmethod
    def base_dir(self) -> Path:
        ...

    @property
    def amici_dir(self) -> Path:
        return self.base_dir / "amici_models"
