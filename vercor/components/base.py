import abc
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict

import h5netcdf
import numpy as np
from numpy.typing import NDArray

from vercor.grid import RectilinearGrid


@dataclass
class TimedNamedArray:
    """Container for a field (array), its timestamp, and its component name."""
    data: np.ndarray
    timestamp: datetime
    component_name: str

    def __array__(self, dtype=None):
        """Let NumPy see this as an array transparently."""
        return np.asarray(self.data, dtype=dtype)

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}:\n"
            f"|----Component name: {self.component_name!r}\n"
            f"|----Shape: {self.data.shape}\n"
            f"|----Timestamp: {self.timestamp!r}"
        )

@dataclass
class Shared:
    _fields: Dict[str, TimedNamedArray] = field(default_factory=dict, init=False)

    def __setattr__(self, name: str, value: Any) -> None:
        # internal attributes
        if name.startswith("_"):
            return super().__setattr__(name, value)

        if isinstance(value, TimedNamedArray):
            self._fields[name] = value
            return

        if isinstance(value, tuple):
            if len(value) == 3:
                data, timestamp, component_name = value
            else:
                raise ValueError(
                f"Expected tuple of length 3 for field assignment, got length {len(value)}"
                )

            if not isinstance(timestamp, datetime):
                raise TypeError(
                    f"When assigning a tuple, the second element must be a datetime, got {type(timestamp)}"
                )

        else:
            raise TypeError(
                "When assigning a field, provide a tuple (data, timestamp, component name)"
            )
        
        data = np.asarray(data)
        self._fields[name] = TimedNamedArray(
            data=data,
            timestamp=timestamp,
            component_name=component_name,
        )

    def __getattr__(self, name: str) -> TimedNamedArray:
        try:
            return self._fields[name]
        except KeyError:
            raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def __str__(self) -> str:
        field_descriptions = ", ".join(
            f"{name}({value.component_name})" for name, value in self._fields.items()
        )
        return (
            f"{self.__class__.__name__}:\n"
            f"|----Fields: {field_descriptions if field_descriptions else 'No fields assigned'}"
        )

    def fields(self) -> Dict[str, np.ndarray]:
        return {k: v.data for k, v in self._fields.items()}

    def timestamps(self) -> Dict[str, datetime]:
        return {k: v.timestamp for k, v in self._fields.items()}

    def component_names(self) -> Dict[str, str]:
        return {k: v.component_name for k, v in self._fields.items()}


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

    Attributes:
        name: component name
        grid: component grid
        shared_fields: dictionary of shared fields from the current component 
            to be exchanged with another component(s)

            N.B! Do not overwrite this directly; use import_fields and export_fields methods
            to keep the data from all exchanged components.
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

    def __str__(self) -> str:
        shared_fields_list = []
        shared_fields_string = ""

        if self.shared_fields:
            shared_fields_list = list(self.shared_fields.keys())
            shared_fields_string = ", ".join(shared_fields_list)

        return (
            f"{self.__class__.__name__}:\n"
            f" |----Name: {self.name}\n"
            f" |----Shared fields: {shared_fields_string if len(shared_fields_list) > 0 else 'Not provided'}\n"
            f" |----Grid name: {self.grid.name}\n"
            f"      |---Shape: {self.grid.shape}\n"
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
