from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import h5netcdf
import jax.numpy as jnp
import numpy as np
import xarray as xr
from numpy.typing import DTypeLike, NDArray

from vercor.clock import ModelDateTime, CustomDateTime
from vercor.exceptions import ComponentError, CouplerError
from vercor.grid import RectilinearGrid
from vercor.settings import ComponentSettings
from vercor.tools import _runtime_array_to_host
from vercor.exchange import VALID_EXCHANGE_FIELD_NAMES
from vercor.types import RuntimeArray

if TYPE_CHECKING:
    from vercor.coupler import Coupler
    from vercor.runtime import RuntimeComponentState, RuntimeStepInfo


@dataclass
class TimedNamedArray:
    """Container class for a field (array), its timestamp, and its component name."""

    data: RuntimeArray
    timestamp: datetime | ModelDateTime
    component_name: str

    def __array__(self, dtype: Optional[DTypeLike] = None) -> NDArray[Any]:
        """Let NumPy see this as an array transparently."""
        return np.asarray(_runtime_array_to_host(self.data), dtype=dtype)

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}:\n"
            f"├── Component name: {self.component_name!r}\n"
            f"├── Shape: {self.data.shape}\n"
            f"└── Timestamp: {self.timestamp!r}"
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(component_name={self.component_name!r}, "
            f"shape={self.data.shape}, timestamp={self.timestamp!r})"
        )


@dataclass
class Shared:
    _fields: dict[str, TimedNamedArray] = field(default_factory=dict, init=False)

    def _assign_field(self, name: str, value: Any) -> None:
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

            if not isinstance(timestamp, datetime | ModelDateTime):
                raise TypeError(
                    f"When assigning a tuple, the second element must be a datetime, got {type(timestamp)}"
                )

        else:
            raise TypeError(
                "When assigning a field, provide a tuple (data, timestamp, component name)"
            )

        self._fields[name] = TimedNamedArray(
            data=data,
            timestamp=timestamp,
            component_name=component_name,
        )

    def __setattr__(self, name: str, value: Any) -> None:
        self._assign_field(name, value)

    def __setitem__(self, name: str, value: Any) -> None:
        self._assign_field(name, value)

    def __getattr__(self, name: str) -> TimedNamedArray:
        try:
            return self._fields[name]
        except KeyError:
            raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def __getitem__(self, name: str) -> TimedNamedArray | None:
        try:
            return self._fields[name]
        except KeyError:
            print(f"{type(self).__name__!s} has no item {name!r}")
            return None

    def __str__(self) -> str:
        field_descriptions = ", ".join(
            f"{name}({value.component_name})" for name, value in self._fields.items()
        )
        return (
            f"{self.__class__.__name__}:\n"
            f"└── Fields: {field_descriptions if field_descriptions else 'No fields assigned'}"
        )

    def __repr__(self) -> str:
        field_reprs = ", ".join(
            f"{name}={repr(value)}" for name, value in self._fields.items()
        )
        return f"{self.__class__.__name__}({field_reprs})"

    @property
    def is_empty(self) -> bool:
        """Check if the Shared object has no fields."""
        return len(self._fields) == 0

    @property
    def field_names(self) -> list[str]:
        """Return a list of all field names in the Shared object."""
        return list(self._fields.keys())

    def fields(self) -> dict[str, RuntimeArray]:
        """Return a dictionary of all fields' data arrays."""
        return {k: v.data for k, v in self._fields.items()}

    def timestamps(self) -> dict[str, datetime | CustomDateTime]:
        """Return a dictionary of all fields' timestamps."""
        return {k: v.timestamp for k, v in self._fields.items()}

    def component_names(self) -> dict[str, str]:
        """Return a dictionary of all fields' component names."""
        return {k: v.component_name for k, v in self._fields.items()}


@dataclass
class Component:
    name: str
    grid: RectilinearGrid
    incoming_fields: Shared = field(default_factory=Shared)
    outgoing_fields: Shared = field(default_factory=Shared)
    data: dict[str, RuntimeArray] = field(default_factory=dict)
    settings: ComponentSettings = field(default_factory=ComponentSettings)
    _fields2import: list[str] = field(default_factory=list)
    _fields2export: list[str] = field(default_factory=list)
    """A component's default grid dimensions are (nTime, nLev, nLon, nLat)

    Some components may have different dimensions, e.g., sea-ice (nTime, nLon, nLat) or
    JCM atmospheric model (nTime, nLev, nLon, nLat).

    One must implement necessary dimensions check and reshaping of fields
    during import/export if needed.

    Common conventions for exchange fields:
        - All fields must have SI units.
        - Surface fluxes are positive downward and negative upward.

    Attributes:
        name: component name
        grid: component grid
        incoming_fields: shared fields received by the current component
                         from another component(s)
        outgoing_fields: shared fields to be sent from the current component
                         to another component(s)
        data: internal storage for component data arrays to/from which fields
                        are imported/exported
        settings: component-specific settings
        _fields2import: list of field names to import from other components to data
        _fields2export: list of field names to export to other components from data
    """

    def initialize(self, coupler: "Coupler") -> None:
        """Initialize component-owned runtime data before coupling."""

        _ = coupler

    def step(
        self,
        dt: timedelta,
        time: datetime | ModelDateTime,
        coupler: "Coupler",
    ) -> None:
        """Compatibility wrapper around the runtime-state step implementation."""

        component_state = self.step_runtime_state(
            self.to_runtime_component_state(prefill_missing=True),
            float(dt.total_seconds()),
            getattr(coupler, "settings", None),
            time=time,
            coupler=coupler,
        )
        self.commit_runtime_state(component_state, time)

    def create_runtime_payload(self) -> Any | None:
        """Return optional immutable payload carried by runtime component state."""

        return None

    def prefill_runtime_state_fields(
        self,
        data: dict[str, RuntimeArray],
        incoming: dict[str, RuntimeArray],
        outgoing: dict[str, RuntimeArray],
    ) -> None:
        """Add missing fields required for stable runtime-state execution."""

        zeros = jnp.zeros(self.grid.shape, dtype=jnp.float_)
        for field_name in self._fields2import:
            incoming.setdefault(field_name, zeros)
            data.setdefault(field_name, zeros)
        for field_name in self._fields2export:
            outgoing.setdefault(field_name, data.get(field_name, zeros))
            data.setdefault(field_name, zeros)

    def _validate_runtime_store_field(
        self,
        store: Any,
        field_name: str,
        store_description: str,
        expected_shape: tuple[int, int],
    ) -> None:
        if field_name not in store.field_names:
            raise CouplerError(
                "Runtime missing "
                f"{store_description} field '{field_name}' for component '{self.name}'"
            )

        field_shape = jnp.asarray(store.get(field_name)).shape
        if field_shape != expected_shape:
            raise CouplerError(
                "Runtime "
                f"{store_description} field '{field_name}' for component '{self.name}' "
                f"has shape {field_shape}, expected {expected_shape}"
            )

    def _validate_runtime_data_field_exists(
        self,
        component_state: "RuntimeComponentState",
        field_name: str,
    ) -> None:
        if field_name not in component_state.data.field_names:
            raise CouplerError(
                "Runtime missing required data field "
                f"'{field_name}' for component '{self.name}'"
            )

    def _validate_runtime_grid_data_field(
        self,
        component_state: "RuntimeComponentState",
        field_name: str,
        expected_shape: tuple[int, int],
    ) -> None:
        self._validate_runtime_data_field_exists(component_state, field_name)
        self._validate_runtime_store_field(
            component_state.data,
            field_name,
            "required data",
            expected_shape,
        )

    def validate_runtime_state(
        self,
        component_state: "RuntimeComponentState",
        expected_shape: tuple[int, int],
    ) -> None:
        """Validate generic runtime fields before execution."""

        for field_name in component_state.fields_to_import:
            self._validate_runtime_store_field(
                component_state.incoming,
                field_name,
                "imported incoming",
                expected_shape,
            )
            self._validate_runtime_grid_data_field(
                component_state,
                field_name,
                expected_shape,
            )
        for field_name in component_state.fields_to_export:
            self._validate_runtime_data_field_exists(component_state, field_name)
            self._validate_runtime_store_field(
                component_state.outgoing,
                field_name,
                "exported source",
                expected_shape,
            )
        for field_name in component_state.incoming.field_names:
            self._validate_runtime_store_field(
                component_state.incoming,
                field_name,
                "incoming",
                expected_shape,
            )

    def to_runtime_component_state(
        self, *, prefill_missing: bool = False
    ) -> "RuntimeComponentState":
        """Create a runtime component state from this wrapper's current fields."""

        from vercor.runtime import RuntimeComponentState, RuntimeFieldStore

        data = dict(self.data)
        incoming = self.incoming_fields.fields()
        outgoing = self.outgoing_fields.fields()
        if prefill_missing:
            self.prefill_runtime_state_fields(data, incoming, outgoing)

        return RuntimeComponentState(
            name=self.name,
            data=RuntimeFieldStore.from_mapping(data),
            incoming=RuntimeFieldStore.from_mapping(incoming),
            outgoing=RuntimeFieldStore.from_mapping(outgoing),
            fields_to_import=tuple(self._fields2import),
            fields_to_export=tuple(self._fields2export),
            runtime_payload=self.create_runtime_payload(),
        )

    def commit_runtime_state(
        self,
        component_state: "RuntimeComponentState",
        timestamp: datetime | ModelDateTime | None = None,
    ) -> None:
        """Copy runtime state fields back to this compatibility wrapper."""

        self.data = component_state.data.to_mapping()
        if timestamp is None:
            return

        incoming_fields = Shared()
        for field_name, field_value in component_state.incoming.to_mapping().items():
            incoming_fields[field_name] = (field_value, timestamp, self.name)
        self.incoming_fields = incoming_fields

        outgoing_fields = Shared()
        for field_name, field_value in component_state.outgoing.to_mapping().items():
            outgoing_fields[field_name] = (field_value, timestamp, self.name)
        self.outgoing_fields = outgoing_fields

    def receive_runtime_fields(
        self,
        component_state: "RuntimeComponentState",
    ) -> "RuntimeComponentState":
        """Move imported incoming runtime fields into component data."""

        data = component_state.data
        for field_name in component_state.fields_to_import:
            data = data.set(field_name, component_state.incoming.get(field_name))
        return component_state.with_data(data)

    def _select_runtime_field_for_send(
        self,
        component_state: "RuntimeComponentState",
        field_name: str,
        step_info: "RuntimeStepInfo | None",
    ) -> RuntimeArray:
        field = component_state.data.get(field_name)
        if step_info is None:
            return field

        if self.settings.apply_time_interpolation:
            arr = jnp.asarray(field)
            left = jnp.take(arr, step_info.monthly_index_left, axis=-1)
            right = jnp.take(arr, step_info.monthly_index_right, axis=-1)
            return (
                step_info.monthly_weight_left * left
                + step_info.monthly_weight_right * right
            ).swapaxes(-2, -1)

        if self.settings.get_field_time_slice:
            return jnp.take(jnp.asarray(field), step_info.daily_index, axis=0)

        return field

    def send_runtime_fields(
        self,
        component_state: "RuntimeComponentState",
        step_info: "RuntimeStepInfo | None" = None,
    ) -> "RuntimeComponentState":
        """Move exported component data into outgoing runtime fields."""

        outgoing = component_state.outgoing
        for field_name in component_state.fields_to_export:
            outgoing = outgoing.set(
                field_name,
                self._select_runtime_field_for_send(
                    component_state,
                    field_name,
                    step_info,
                ),
            )
        return component_state.with_outgoing(outgoing)

    def step_runtime_state(
        self,
        component_state: "RuntimeComponentState",
        dt_seconds: float,
        runtime_settings: Any | None = None,
        *,
        time: datetime | ModelDateTime | None = None,
        coupler: "Coupler | None" = None,
    ) -> "RuntimeComponentState":
        """Return this component advanced by one runtime step."""

        _ = dt_seconds, runtime_settings, time, coupler
        return component_state

    def finalize(
        self, coupler: "Coupler", output_file_mask: Optional[Path] = None
    ) -> None:
        """Finalize the component by writing its all shared fields (incoming and outgoing)
        to a netCDF file.

        Arguments:
            output_file_mask: optional mask to include in the output filename
        """

        if output_file_mask is None:
            filepath = Path(f"{self.name.lower()}_component_shared_fields.nc")
        else:
            filepath = Path(f"{self.name.lower()}_{output_file_mask}.nc")

        merged_fields = self.merge_incoming_outgoing_fields()
        coupler.append_masks_to_output(self.name, merged_fields)

        write_shared_to_netcdf(merged_fields, self.grid, filepath)

    def check_not_empty_import_export_lists(self) -> None:
        """Check that the component has non-empty and non-overlapping
        import and export fields.
        """

        if not self._fields2import:
            raise ComponentError(
                f"Component '{self.name}' has no fields to import defined."
            )
        if not self._fields2export:
            raise ComponentError(
                f"Component '{self.name}' has no fields to export defined."
            )

        all_fields = set(self._fields2import + self._fields2export)
        if len(all_fields) < len(self._fields2import) + len(self._fields2export):
            raise ComponentError(
                f"Component '{self.name}' has overlapping fields in import/export lists."
            )

    def check_valid_exchange_field_names(self) -> None:
        for fld in set(self._fields2import + self._fields2export):
            if fld not in VALID_EXCHANGE_FIELD_NAMES:
                raise ComponentError(
                    f"Field name '{fld}' in component '{self.name}' is not a recognized exchange variable.\n"
                    f"Replace field name '{fld}' with one of the supported names: {VALID_EXCHANGE_FIELD_NAMES}"
                )

    def get(self, field_name: str) -> RuntimeArray:
        """
        Returns the data array of the specified field from either
        incoming_fields or outgoing_fields.

        Arguments:
            field_name (str): name of the field to retrieve
        """

        in_fields = self.incoming_fields.fields()
        out_fields = self.outgoing_fields.fields()

        if field_name in in_fields and field_name in out_fields:
            raise ComponentError(
                f"Field name '{field_name}' found in both incoming and outgoing fields."
            )

        if field_name in in_fields:
            return in_fields[field_name]

        if field_name in out_fields:
            return out_fields[field_name]

        if field_name in self.data:
            return self.data[field_name]

        raise ComponentError(
            f"Field name '{field_name}' not found in incoming, outgoing or internal pool of fields"
        )

    def merge_incoming_outgoing_fields(self) -> Shared:
        """
        Merge incoming_fields and outgoing_fields into a single Shared object for further output.
        """

        output_fields = Shared()

        for name, tna in self.incoming_fields._fields.items():
            output_fields[name] = tna
        for name, tna in self.outgoing_fields._fields.items():
            output_fields[name] = tna

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

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name={self.name!r}, grid={repr(self.grid)},"
            f" incoming_fields={repr(self.incoming_fields)}, outgoing_fields={repr(self.outgoing_fields)})"
        )


class ComponentForcingData:
    def __init__(self) -> None:
        self.DATA_FILES: dict[str, str] = {}

    def _read_forcing(
        self, variable: str, where: str, flip_y: bool = False
    ) -> RuntimeArray:
        """Read a variable from the specified forcing file.

        Arguments:
            variable (str): variable name to read from a file
            where (str): key to identify which file to read from DATA_FILES
            flip_y (bool): whether to flip the variable along the latitude axis

        Returns:
            RuntimeArray: the requested variable data as a JAX-backed array.
        """

        try:
            with h5netcdf.File(self.DATA_FILES[where], "r") as infile:
                var_obj = jnp.asarray(np.array(infile.variables[variable]).T)
                if flip_y:
                    return jnp.flip(var_obj, axis=1)
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

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}:\n"
            f"└── Forcing files: {self.DATA_FILES if self.DATA_FILES else 'No files assigned'}"
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(DATA_FILES={self.DATA_FILES})"


def write_shared_to_netcdf(
    shared: Shared, grid: RectilinearGrid, filename: Path
) -> None:
    """Write the contents of a Shared object to a netCDF file.
    Arguments:
        shared: Shared object containing fields to write
        grid: Grid object defining the grid
        filename: path to the output netCDF file
    """

    lat = xr.DataArray(
        _runtime_array_to_host(grid.latitude), dims=("nlat",), name="latitude"
    )
    lon = xr.DataArray(
        _runtime_array_to_host(grid.longitude), dims=("nlon",), name="longitude"
    )

    data_vars = {}
    for name, tna in shared._fields.items():
        data_vars[name] = xr.DataArray(
            data=_runtime_array_to_host(tna.data),
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
