from typing import Any, TypeAlias

import jax
from numpy.typing import NDArray

RuntimeArray: TypeAlias = NDArray[Any] | jax.Array

__all__ = ["RuntimeArray"]
