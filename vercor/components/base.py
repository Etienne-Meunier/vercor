from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5netcdf
import jax.numpy as jnp
import numpy as np
import xarray as xr

from vercor.clock import ModelDateTime
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
class Component:
    name: str
    grid: RectilinearGrid
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
        data: internal storage for component data arrays to/from which fields
                        are imported/exported
        settings: component-specific settings
        _fields2import: list of field names to import from other components to data
        _fields2export: list of field names to export to other components from data
    """

    def initialize(self, coupler: "Coupler") -> None:
        """Initialize component-owned runtime data before coupling."""

        _ = coupler

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
    ) -> None:
        expected_shape = self.grid.shape
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
    ) -> None:
        self._validate_runtime_data_field_exists(component_state, field_name)
        self._validate_runtime_store_field(
            component_state.data,
            field_name,
            "required data",
        )

    def validate_runtime_state(
        self,
        component_state: "RuntimeComponentState",
    ) -> None:
        """Validate generic runtime fields before execution.

        Validation derives expected 2D field shapes from ``self.grid.shape`` so
        callers do not need to pass shape metadata already owned by the
        component.
        """

        for field_name in self._fields2import:
            self._validate_runtime_store_field(
                component_state.incoming,
                field_name,
                "imported incoming",
            )
            self._validate_runtime_grid_data_field(
                component_state,
                field_name,
            )
        for field_name in self._fields2export:
            self._validate_runtime_data_field_exists(component_state, field_name)
            self._validate_runtime_store_field(
                component_state.outgoing,
                field_name,
                "exported source",
            )
        for field_name in component_state.incoming.field_names:
            self._validate_runtime_store_field(
                component_state.incoming,
                field_name,
                "incoming",
            )

    def to_runtime_component_state(
        self, *, prefill_missing: bool = False
    ) -> "RuntimeComponentState":
        """Create a runtime component state from this component's data."""

        from vercor.runtime import RuntimeComponentState, RuntimeFieldStore

        data = dict(self.data)
        incoming: dict[str, RuntimeArray] = {}
        outgoing: dict[str, RuntimeArray] = {}
        if prefill_missing:
            self.prefill_runtime_state_fields(data, incoming, outgoing)

        return RuntimeComponentState(
            data=RuntimeFieldStore.from_mapping(data),
            incoming=RuntimeFieldStore.from_mapping(incoming),
            outgoing=RuntimeFieldStore.from_mapping(outgoing),
            runtime_payload=self.create_runtime_payload(),
        )

    def _sync_data_from_runtime_state(
        self,
        component_state: "RuntimeComponentState",
    ) -> None:
        """Synchronize mutable host-adapter storage from runtime state data."""

        self.data = component_state.data.to_mapping()

    def receive_runtime_fields(
        self,
        component_state: "RuntimeComponentState",
    ) -> "RuntimeComponentState":
        """Move imported incoming runtime fields into component data."""

        data = component_state.data
        for field_name in self._fields2import:
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
        for field_name in self._fields2export:
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

    def __str__(self) -> str:
        field_names = sorted(set(self._fields2import + self._fields2export))
        field_names_string = ", ".join(field_names)

        return (
            f"{self.__class__.__name__}:\n"
            f" ├── Name: {self.name}\n"
            f" ├── Runtime fields: {field_names_string if field_names else 'Not provided'}\n"
            f" └── Grid name: {self.grid.name}\n"
            f"     └── Shape: {self.grid.shape}\n"
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, grid={repr(self.grid)})"


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


def write_runtime_component_to_netcdf(
    component_name: str,
    component_state: "RuntimeComponentState",
    grid: RectilinearGrid,
    filename: Path,
    *,
    masks: dict[str, RuntimeArray] | None = None,
) -> None:
    """Write final runtime component fields to a netCDF file.

    Arguments:
        component_name: component name to write as output metadata
        component_state: runtime component state containing fields to write
        grid: Grid object defining the grid
        filename: path to the output netCDF file
        masks: optional mask fields to include in the same output
    """

    lat = xr.DataArray(
        _runtime_array_to_host(grid.latitude), dims=("nlat",), name="latitude"
    )
    lon = xr.DataArray(
        _runtime_array_to_host(grid.longitude), dims=("nlon",), name="longitude"
    )

    data_vars = {}
    for store_name, store in (
        ("incoming", component_state.incoming),
        ("outgoing", component_state.outgoing),
    ):
        for name, value in store.to_mapping().items():
            data_vars[f"{store_name}_{name}"] = xr.DataArray(
                data=_runtime_array_to_host(value),
                dims=("nlat", "nlon"),
                coords={"latitude": lat, "longitude": lon},
                attrs={
                    "component": component_name,
                    "runtime_store": store_name,
                    "field_name": name,
                },
            )

    for name, value in (masks or {}).items():
        data_vars[name] = xr.DataArray(
            data=_runtime_array_to_host(value),
            dims=("nlat", "nlon"),
            coords={"latitude": lat, "longitude": lon},
            attrs={
                "component": component_name,
                "runtime_store": "mask",
                "field_name": name,
            },
        )

    xr.Dataset(
        data_vars=data_vars,
        coords={"latitude": lat, "longitude": lon},
    ).to_netcdf(filename)
