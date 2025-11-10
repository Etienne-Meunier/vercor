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
    """A component's default grid dimensions are (nTime, nLev, nLon, nLat)

    Some components may have different dimensions, e.g., sea-ice (nTime, nLon, nLat) or
    JCM atmospheric model (nTime, nLev, nLon, nLat). 

    One must implement necessary dimensions check and reshaping of fields
    during import/export if needed.
    """

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
