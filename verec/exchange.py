from dataclasses import dataclass
from typing import Callable, List, Union

from .regridders.bilinear import XESMFBilinear
from .regridders.conservative import XESMFConservative_normed


@dataclass
class Exchange:
    name: str
    src: str  # component name
    dst: str  # component name
    field_names: List[str]
    regridder_factory: Callable  # (src_grid, dst_grid) -> Regridder
    when: str = "pre"  # "pre" or "post" component stepping

    def build(
        self, src_grid, dst_grid, src_mask=None, dst_mask=None
    ) -> Union[XESMFBilinear, XESMFConservative_normed]:
        regridder = self.regridder_factory(src_grid, src_mask, dst_grid, dst_mask)
        return regridder.prepare()
