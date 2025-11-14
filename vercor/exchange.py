from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Union

from vercor.grid import Grid
from vercor.regridders.bilinear import BilinearRectilinearRegridder


@dataclass
class Exchange:
    source: str
    destination: str
    name: str = field(init=False)
    field_names: List[Union[str, Tuple[str, str]]]
    regridder_factory: Callable[..., BilinearRectilinearRegridder]
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

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}:\n"
            f"|----Name: {self.name}\n"
            f"|----Source component: {self.source}\n"
            f"|----Destination component: {self.destination}\n"
        )

    def create(
        self,
        source_grid: Grid,
        destination_grid: Grid,
    ) -> BilinearRectilinearRegridder:
        return self.regridder_factory(source_grid, destination_grid)
