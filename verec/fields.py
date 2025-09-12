from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import numpy as np


@dataclass
class Field:
    name: str
    data: np.ndarray
    grid: Any  # forward-declared; typically a Grid
    units: str = ""
    attrs: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[float] = None  # seconds since start or UNIX ts ???

    def like(
        self,
        name: Optional[str] = None,
        data: Optional[np.ndarray] = None,
        units: Optional[str] = None,
    ) -> "Field":
        return Field(
            name=name or self.name,
            data=data if data is not None else self.data.copy(),
            grid=self.grid,
            units=units or self.units,
            attrs=dict(self.attrs),
            timestamp=self.timestamp,
        )
