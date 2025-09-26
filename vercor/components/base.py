import abc
from typing import Dict
from dataclasses import dataclass, field
import numpy as np
from vercor.grid import RectilinearGrid


@dataclass
class Component(abc.ABC):
    name: str
    grid: RectilinearGrid
    state: Dict[str, np.ndarray] = field(default_factory=dict)

    @abc.abstractmethod
    def initialize(self, coupler):
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, dt, time, coupler):
        raise NotImplementedError

    def export_fields(self) -> Dict[str, np.ndarray]:
        return {k: v for k, v in self.state.items()}

    def import_fields(self, fields: Dict[str, np.ndarray]) -> None:
        # simplistic merge/overwrite
        for name, fld in fields.items():
            self.state[name] = fld
