import abc
from typing import Dict
from dataclasses import dataclass, field
from numpy.typing import NDArray
from vercor.grid import RectilinearGrid


@dataclass
class Component(abc.ABC):
    name: str
    grid: RectilinearGrid
    state: Dict[str, NDArray] = field(default_factory=dict)

    @abc.abstractmethod
    def initialize(self, coupler):
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, dt, time, coupler):
        raise NotImplementedError

    def export_fields(self) -> Dict[str, NDArray]:
        return {k: v for k, v in self.state.items()}

    def import_fields(self, fields: Dict[str, NDArray]) -> None:
        # TODO: implement more sophisticated merging with dimensions checks for every array
        # simplistic merge/overwrite
        for name, fld in fields.items():
            self.state[name] = fld
