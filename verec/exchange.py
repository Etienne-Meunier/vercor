from dataclasses import dataclass
from typing import Any, Callable, List, Tuple, Union


@dataclass
class Exchange:
    name: str
    source: str  # component name
    destination: str  # component name
    field_names: List[
        Union[str, Tuple[str, str]]
    ]  # list of scalar field names or (u-vector-component, v-vector-component)
    regridder_factory: Callable  # (src_grid, src_mask, dst_grid, dst_mask) -> Regridder
    when: str = "pre"  # "pre" or "post" component stepping

    def build(
        self,
        src_grid,
        src_mask,
        dst_grid,
        dst_mask,
    ) -> Any:
        regridder = self.regridder_factory(src_grid, src_mask, dst_grid, dst_mask)
        return regridder.prepare()
