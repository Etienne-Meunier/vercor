from dataclasses import dataclass
from typing import Callable, List, Tuple, Union

from verec.regridders.base import Regridder


@dataclass
class Exchange:
    name: str
    source: str  # component name
    destination: str  # component name
    field_names: List[
        Union[str, Tuple[str, str]]
    ]  # list of scalar field names and (u-vector-component, v-vector-component)
    regridder_factory: Callable[..., Regridder]
    when: str = "pre"  # "pre" or "post" component stepping

    def build(
        self,
        source_grid,
        source_mask,
        destination_grid,
        destination_mask,
    ) -> Regridder:
        regridder = self.regridder_factory(
            source_grid, source_mask, destination_grid, destination_mask
        )
        return regridder.prepare()
