from dataclasses import dataclass
from typing import Any, Callable, List


@dataclass
class Exchange:
    name: str
    source: str  # component name
    destination: str  # component name
    field_names: List[str]
    regridder_factory: Callable  # (src_grid, dst_grid) -> Regridder
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
