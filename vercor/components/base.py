import abc
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import h5netcdf
import numpy as np
import xarray as xr
from numpy.typing import NDArray

from vercor.grid import RectilinearGrid
from vercor.tools import get_field_at_specific_time

if TYPE_CHECKING:
    from vercor.coupler import Coupler


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
            f"├── Component name: {self.component_name!r}\n"
            f"├── Shape: {self.data.shape}\n"
            f"└── Timestamp: {self.timestamp!r}"
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
            f"└── Fields: {field_descriptions if field_descriptions else 'No fields assigned'}"
        )

    @property
    def is_empty(self) -> bool:
        return len(self._fields) == 0

    @property
    def field_names(self) -> List[str]:
        return list(self._fields.keys())

    def fields(self) -> Dict[str, NDArray]:
        return {k: v.data for k, v in self._fields.items()}

    def timestamps(self) -> Dict[str, datetime]:
        return {k: v.timestamp for k, v in self._fields.items()}

    def component_names(self) -> Dict[str, str]:
        return {k: v.component_name for k, v in self._fields.items()}


@dataclass
class Component(abc.ABC):
    name: str
    grid: RectilinearGrid
    incoming_fields: Shared = field(default_factory=Shared)
    outgoing_fields: Shared = field(default_factory=Shared)
    _cdata: Dict[str, NDArray] = field(default_factory=dict)
    _fields2import: List[str] = field(default_factory=list)
    _fields2export: List[str] = field(default_factory=list)
    _settings: Dict[str, Any] = field(default_factory=dict)
    """A component's default grid dimensions are (nTime, nLev, nLon, nLat)

    Some components may have different dimensions, e.g., sea-ice (nTime, nLon, nLat) or
    JCM atmospheric model (nTime, nLev, nLon, nLat). 

    One must implement necessary dimensions check and reshaping of fields
    during import/export if needed.

    Attributes:
        name: component name
        grid: component grid
        incoming_fields, outgoing_fields: shared fields from the current component
            received from another component(s)
        _settings: component-specific settings
        _fields2import: list of field names to import from other components
        _fields2export: list of field names to export to other components
        _cdata: internal storage for component data arrays
    """

    @abc.abstractmethod
    def initialize(self, coupler: "Coupler") -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def step(
        self,
        dt: timedelta,
        time: datetime,
        coupler: "Coupler",
    ) -> None:
        raise NotImplementedError

    def finalize(self, output_file_mask: Optional[Path] = None) -> None:
        if output_file_mask is None:
            filepath = Path(f"{self.name.lower()}_shared.nc")
        else:
            filepath = Path(f"{self.name.lower()}_{output_file_mask}.nc")

        merged_fields = self.merge_incoming_outgoing_fields()
        write_shared_to_netcdf(merged_fields, self.grid, filepath)

    def check_not_empty_import_export_lists(self) -> None:
        if not self._fields2import:
            raise ValueError(
                f"Component '{self.name}' has no fields to import defined."
            )
        if not self._fields2export:
            raise ValueError(
                f"Component '{self.name}' has no fields to export defined."
            )

        all_fields = set(self._fields2import + self._fields2export)
        if len(all_fields) < len(self._fields2import) + len(self._fields2export):
            raise ValueError(
                f"Component '{self.name}' has overlapping fields in import/export lists."
            )

    def export_fields(self) -> Shared:
        # TODO: export only component related fields
        return self.outgoing_fields

    def import_fields(self, fields: Shared) -> None:
        # TODO: import only component related fields
        incoming_fields = fields.field_names
        for name in incoming_fields:
            setattr(self.incoming_fields, name, getattr(fields, name))

    def receive_fields(self, time: datetime) -> None:
        # check that all required fields are present
        for field in self._fields2import:
            if field not in self.incoming_fields.field_names:
                raise KeyError(
                    f"Field '{field}' required by component '{self.name}' not found in incoming fields."
                )

        # check if every imported field's timestamp matches the current time
        for field in self._fields2import:
            tna = getattr(self.incoming_fields, field)
            if tna.timestamp != time:
                raise ValueError(
                    f"Receive field '{field}' timestamp {tna.timestamp} does not match current time {time} in component '{self.name}'."
                )

        self._cdata.update(self.incoming_fields.fields())

    def send_fields(self, time: datetime, coupler: "Coupler") -> None:
        for field in self._fields2export:
            if self._settings.get("apply_time_interpolation", False):
                # for data models with monthly means
                field2send = get_field_at_specific_time(field, self._cdata, coupler)
            else:
                field2send = self._cdata[field]

            setattr(
                self.outgoing_fields,
                field,
                TimedNamedArray(field2send, time, self.name),
            )

    def get(self, field_name: str) -> NDArray:
        in_fields = self.incoming_fields.fields()
        out_fields = self.outgoing_fields.fields()
        in_fieldnames = in_fields.keys()
        out_fieldnames = out_fields.keys()

        if field_name in in_fieldnames and field_name in out_fieldnames:
            raise KeyError(
                f"Field name '{field_name}' found in both incoming and outgoing fields."
            )

        if field_name in in_fieldnames:
            return in_fields[field_name]

        if field_name in out_fieldnames:
            return out_fields[field_name]

        raise KeyError(
            f"Field name '{field_name}' not found in incoming or outgoing fields"
        )

    def merge_incoming_outgoing_fields(self) -> Shared:
        output_fields = Shared()

        for name, tna in self.incoming_fields._fields.items():
            setattr(output_fields, name, tna)
        for name, tna in self.outgoing_fields._fields.items():
            setattr(output_fields, name, tna)

        return output_fields

    def __str__(self) -> str:
        shared_fields_list = []
        shared_fields_string = ""

        if self.incoming_fields or self.outgoing_fields:
            shared_fields_list = list(self.incoming_fields.fields().keys()) + list(
                self.outgoing_fields.fields().keys()
            )
            shared_fields_string = ", ".join(shared_fields_list)

        return (
            f"{self.__class__.__name__}:\n"
            f" ├── Name: {self.name}\n"
            f" ├── Shared fields: {shared_fields_string if len(shared_fields_list) > 0 else 'Not provided'}\n"
            f" └── Grid name: {self.grid.name}\n"
            f"     └── Shape: {self.grid.shape}\n"
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


def write_shared_to_netcdf(
    shared: Shared, grid: RectilinearGrid, filename: Path
) -> None:
    lat = xr.DataArray(grid.latitude, dims=("nlat",), name="latitude")
    lon = xr.DataArray(grid.longitude, dims=("nlon",), name="longitude")

    data_vars = {}
    for name, tna in shared._fields.items():
        data_vars[name] = xr.DataArray(
            data=tna.data,
            dims=("nlat", "nlon"),
            coords={"latitude": lat, "longitude": lon},
            attrs={
                "timestamp": tna.timestamp.isoformat(),
                "component": tna.component_name,
            },
        )

    xr.Dataset(
        data_vars=data_vars,
        coords={"latitude": lat, "longitude": lon},
    ).to_netcdf(filename)


if __name__ == "__main__":
    shared = Shared()
    if not shared.is_empty:
        print("Shared is not empty initially, something is wrong!")

    t_model = datetime(2025, 11, 14, 12, 0, 0)
    shared.temperature = (np.array([[1.0, 2.0], [3.0, 4.0]]), t_model, "ocean")
    shared.humidity = (np.array([[0.5, 0.6], [0.7, 0.8]]), t_model, "atmosphere")
    shared.temperature.data += 10.0

    if shared.is_empty:
        print("Shared is not empty!")

    print(shared)

    temp_array = shared.temperature
    print(temp_array)
    print("Temperature data:\n", temp_array.data)
    print("Temperature timestamp:", temp_array.timestamp)
    print("Temperature component name:", temp_array.component_name)
