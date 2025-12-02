from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Union

from vercor.grid import Grid
from vercor.regridders.bilinear import BilinearRectilinearRegridder
from vercor.regridders.conservative import ConservativeRectilinearRegridder


@dataclass
class Exchange:
    source: str
    destination: str
    name: str = field(init=False)
    field_names: List[Union[str, Tuple[str, str]]]
    regridder_factory: Callable[
        ..., BilinearRectilinearRegridder | ConservativeRectilinearRegridder
    ]
    """
    Exchange definition between two components

        source, destination: component names
        name: exchange name (automatically set to "SOURCE2DESTINATION")
        field_names: list of scalar field names and
                     tuples of vectors (u-component, v-component)
        regridder_factory: list of callables that return Regridder instances
    """

    def __post_init__(self) -> None:
        self.name = f"{self.source}2{self.destination}"

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}:\n"
            f"├── Name: {self.name}\n"
            f"├── Source component: {self.source}\n"
            f"└── Destination component: {self.destination}\n"
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name={self.name}, source={self.source},"
            f" destination={self.destination}, fields={self.field_names})"
        )

    def create(
        self,
        source_grid: Grid,
        destination_grid: Grid,
    ) -> BilinearRectilinearRegridder | ConservativeRectilinearRegridder:
        """
        Create and return a Regridder instance using the provided factory.

        Arguments:
            source_grid: Grid of the source component
            destination_grid: Grid of the destination component

        Returns:
            Regridder instance created by the factory
        """
        return self.regridder_factory(source_grid, destination_grid)
