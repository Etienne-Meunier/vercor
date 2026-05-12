from __future__ import annotations

import h5netcdf
import jax.numpy as jnp
import numpy as np

from vercor.dtypes import as_jax_real_array
from vercor.types import RuntimeArray


class ComponentForcingData:
    """Read named forcing variables from configured NetCDF files."""

    def __init__(self) -> None:
        self.DATA_FILES: dict[str, str] = {}

    def _read_forcing(
        self, variable: str, where: str, flip_y: bool = False
    ) -> RuntimeArray:
        """Read a variable from one configured forcing file as a JAX array."""

        try:
            with h5netcdf.File(self.DATA_FILES[where], "r") as infile:
                var_obj = as_jax_real_array(np.array(infile.variables[variable]).T)
                if flip_y:
                    return jnp.flip(var_obj, axis=1)
                return var_obj
        except KeyError as e:
            raise KeyError(
                f"Provided 'where' key '{where}' not found in DATA_FILES"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Error reading variable '{variable}' from forcing file '{self.DATA_FILES[where]}'"
            ) from e

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}:\n"
            f"└── Forcing files: {self.DATA_FILES if self.DATA_FILES else 'No files assigned'}"
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(DATA_FILES={self.DATA_FILES})"
