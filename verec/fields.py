from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import numpy as np


@dataclass
class Field:
    name: str
    data: np.ndarray
    grid: Any
    units: str = ""
    attrs: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[float] = None
