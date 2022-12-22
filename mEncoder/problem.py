import pypesto
import pysb
import amici

from abc import abstractmethod
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple


@dataclass
class ParameterBounds:
    kdeg: Tuple[float, float, str]
    eq: Tuple[float, float, str]
    kcat: Tuple[float, float, str]
    kr: Tuple[float, float, str]
    scale: Tuple[float, float, str]
    offset: Tuple[float, float, str]
    weight: Tuple[float, float, str]
    koff: Tuple[float, float, str]
    kd: Tuple[float, float, str]
    kw: Tuple[float, float, str]

    def __getitem__(self, item):
        return getattr(self, item)


@dataclass
class Problem(object):
    pathway_name: str

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
    def apply_objective_settings(self, objective: pypesto.ObjectiveBase, n_threads: int = 1):
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
        return self.base_dir / 'amici_models'
