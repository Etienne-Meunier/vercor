import abc
from typing import Dict, List
from dataclasses import dataclass, field
from verec.grid import RectilinearGrid
from verec.fields import Field


@dataclass
class Component(abc.ABC):
    name: str
    grid: RectilinearGrid
    # inputs: List[str] = field(default_factory=list)
    # outputs: List[str] = field(default_factory=list)
    state: Dict[str, Field] = field(default_factory=dict)

    @abc.abstractmethod
    def initialize(self, coupler):
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, dt, time, coupler):
        raise NotImplementedError

    def export_fields(self) -> Dict[str, Field]:
        return {k: v for k, v in self.state.items()}

    def import_fields(self, fields: Dict[str, Field]) -> None:
        # simplistic merge/overwrite
        for name, fld in fields.items():
            self.state[name] = fld
