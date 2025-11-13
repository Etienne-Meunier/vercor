import abc
from dataclasses import dataclass, field
from typing import Dict

import h5netcdf
import numpy as np
from numpy.typing import NDArray

from vercor.grid import RectilinearGrid


@dataclass
class Component(abc.ABC):
    name: str
    grid: RectilinearGrid
    shared_fields: Dict[str, NDArray] = field(default_factory=dict)
    """A component's default grid dimensions are (nTime, nLev, nLon, nLat)

    Some components may have different dimensions, e.g., sea-ice (nTime, nLon, nLat) or
    JCM atmospheric model (nTime, nLev, nLon, nLat). 

    One must implement necessary dimensions check and reshaping of fields
    during import/export if needed.
    """

    @abc.abstractmethod
    def initialize(self, coupler):
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, dt, time, coupler):
        raise NotImplementedError

    @abc.abstractmethod
    def finalize(self, coupler):
        raise NotImplementedError

    def export_fields(self) -> Dict[str, NDArray]:
        return {k: v for k, v in self.shared_fields.items()}

    def import_fields(self, fields: Dict[str, NDArray]) -> None:
        # TODO: implement more sophisticated merging with dimensions checks for every array
        # simplistic merge/overwrite
        for name, fld in fields.items():
            self.shared_fields[name] = fld

    def __repr__(self) -> str:
        shared_fields_list = []
        shared_fields_string = ""
        if self.shared_fields:
            shared_fields_list = list(self.shared_fields.keys())
            shared_fields_string = ", ".join(shared_fields_list)
        return (
            f"{self.__class__.__name__}:\n"
            f"|----Name: {self.name}\n"
            f"|----Shared fields: {shared_fields_string if len(shared_fields_list) > 0 else 'Not provided'}\n"
            f"|----Grid name: {self.grid.name}\n"
            f"     |---Shape: {self.grid.shape}\n"
        )


class ForcingData:
    def __init__(self):
        self.DATA_FILES = {}

    def _read_forcing(self, variable: str, where: str, flip_y: bool = False):
        """Read a variable from the specified forcing file.

        Arguments:
            variable (str): variable name to read from a file
            where (str): key to identify which file to read from DATA_FILES
            flip_y (bool): whether to flip the variable along the latitude axis

        Returns:
            (`ndarray`): the requested variable data
        """

        try:
            with h5netcdf.File(self.DATA_FILES[where], "r") as infile:
                var_obj = np.array(infile.variables[variable]).T
                if flip_y:
                    return np.flip(var_obj, axis=1)
                else:
                    return var_obj
        except KeyError as e:
            raise KeyError(
                f"Provided 'where' key '{where}' not found in DATA_FILES"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Error reading variable '{variable}' from forcing file '{self.DATA_FILES[where]}'"
            ) from e
