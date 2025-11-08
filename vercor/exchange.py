from dataclasses import dataclass
from typing import Callable, List, Tuple, Union
from vercor.grid import Grid
from vercor.regridders.base import Regridder
from vercor.regridders.bilinear import BilinearRectilinear


@dataclass
class Exchange:
    source: str
    destination: str
    field_names: List[Union[str, Tuple[str, str]]]
    regridder_factory: Callable[..., Regridder]
    when: str = "pre"
    """
    Exchange definition between two components

        source, destination: component names
        field_names: list of scalar field names and
                     tuples of vectors (u-component, v-component)
        regridder_factory: list of callables that return Regridder instances
        when: specifies when to perform the exchange, i.e, 
              before (pre) or after (post) component stepping
    """

    def __post_init__(self) -> None:
        self.name = f"{self.source}2{self.destination}"

    def create(
        self,
        source_grid: Grid,
        destination_grid: Grid,
    ) -> BilinearRectilinear:
        regridder = self.regridder_factory(source_grid, destination_grid)
        return regridder.setup()
