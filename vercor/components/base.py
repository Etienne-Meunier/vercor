from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from logging import Logger
from typing import TYPE_CHECKING, Any

import h5netcdf
import jax.numpy as jnp
import numpy as np

from vercor.clock import ModelDateTime
from vercor.exceptions import ComponentError, CouplerError
from vercor.grid import RectilinearGrid
from vercor.settings import ComponentSettings
from vercor.exchange import VALID_EXCHANGE_FIELD_NAMES
from vercor.types import RuntimeArray

if TYPE_CHECKING:
    from vercor.coupler import Coupler
    from vercor.runtime import (
        RuntimeComponentContract,
        RuntimeComponentState,
        RuntimeStepInfo,
    )


def _runtime_contract(
    contract: RuntimeComponentContract | None,
) -> RuntimeComponentContract:
    """Return an explicit runtime contract, defaulting to no import/export fields."""

    from vercor.runtime import RuntimeComponentContract

    return RuntimeComponentContract.empty() if contract is None else contract


@dataclass
class Component:
    name: str
    grid: RectilinearGrid
    data: dict[str, RuntimeArray] = field(default_factory=dict)
    settings: ComponentSettings = field(default_factory=ComponentSettings)
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
                        seed the runtime state during initialization
        settings: component-specific settings
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
        contract: RuntimeComponentContract | None = None,
    ) -> None:
        """Add missing fields required for stable runtime-state execution."""

        runtime_contract = _runtime_contract(contract)
        zeros = jnp.zeros(self.grid.shape, dtype=jnp.float_)
        for field_name in runtime_contract.imports:
            incoming.setdefault(field_name, zeros)
            data.setdefault(field_name, zeros)
        for field_name in runtime_contract.exports:
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
        contract: RuntimeComponentContract | None = None,
    ) -> None:
        """Validate generic runtime fields before execution.

        Validation derives expected 2D field shapes from ``self.grid.shape`` so
        callers do not need to pass shape metadata already owned by the
        component.
        """

        runtime_contract = _runtime_contract(contract)
        for field_name in runtime_contract.imports:
            self._validate_runtime_store_field(
                component_state.incoming,
                field_name,
                "imported incoming",
            )
            self._validate_runtime_grid_data_field(
                component_state,
                field_name,
            )
        for field_name in runtime_contract.exports:
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
        self,
        *,
        prefill_missing: bool = False,
        contract: RuntimeComponentContract | None = None,
    ) -> "RuntimeComponentState":
        """Create a runtime component state from this component's data."""

        from vercor.runtime import RuntimeComponentState, RuntimeFieldStore

        data = dict(self.data)
        incoming: dict[str, RuntimeArray] = {}
        outgoing: dict[str, RuntimeArray] = {}
        if prefill_missing:
            self.prefill_runtime_state_fields(data, incoming, outgoing, contract)

        return RuntimeComponentState(
            data=RuntimeFieldStore.from_mapping(data),
            incoming=RuntimeFieldStore.from_mapping(incoming),
            outgoing=RuntimeFieldStore.from_mapping(outgoing),
            runtime_payload=self.create_runtime_payload(),
        )

    def receive_runtime_fields(
        self,
        component_state: "RuntimeComponentState",
        contract: RuntimeComponentContract | None = None,
    ) -> "RuntimeComponentState":
        """Move imported incoming runtime fields into component data."""

        runtime_contract = _runtime_contract(contract)
        data = component_state.data
        for field_name in runtime_contract.imports:
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
        *,
        contract: RuntimeComponentContract | None = None,
    ) -> "RuntimeComponentState":
        """Move exported component data into outgoing runtime fields."""

        runtime_contract = _runtime_contract(contract)
        outgoing = component_state.outgoing
        for field_name in runtime_contract.exports:
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
        logger: Logger | None = None,
    ) -> "RuntimeComponentState":
        """Return this component advanced by one runtime step."""

        _ = dt_seconds, runtime_settings, time, logger
        return component_state

    def check_not_empty_import_export_lists(
        self,
        contract: RuntimeComponentContract | None = None,
    ) -> None:
        """Check that the component has non-empty and non-overlapping
        import and export fields.
        """

        runtime_contract = _runtime_contract(contract)
        if not runtime_contract.imports:
            raise ComponentError(
                f"Component '{self.name}' has no fields to import defined."
            )
        if not runtime_contract.exports:
            raise ComponentError(
                f"Component '{self.name}' has no fields to export defined."
            )

        all_fields = set(runtime_contract.all_fields)
        if len(all_fields) < len(runtime_contract.all_fields):
            raise ComponentError(
                f"Component '{self.name}' has overlapping fields in import/export lists."
            )

    def check_valid_exchange_field_names(
        self,
        contract: RuntimeComponentContract | None = None,
    ) -> None:
        runtime_contract = _runtime_contract(contract)
        for fld in set(runtime_contract.all_fields):
            if fld not in VALID_EXCHANGE_FIELD_NAMES:
                raise ComponentError(
                    f"Field name '{fld}' in component '{self.name}' is not a recognized exchange variable.\n"
                    f"Replace field name '{fld}' with one of the supported names: {VALID_EXCHANGE_FIELD_NAMES}"
                )

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}:\n"
            f" ├── Name: {self.name}\n"
            f" ├── Runtime fields: Configured by Coupler runtime contract\n"
            f" └── Grid name: {self.grid.name}\n"
            f"     └── Shape: {self.grid.shape}\n"
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, grid={repr(self.grid)})"


class HostRuntimeComponent(Component):
    """Base class for host-backed adapters that cannot run inside JAX scan."""

    def _step_host_runtime_state(
        self,
        component_state: "RuntimeComponentState",
        dt_seconds: float,
        runtime_settings: Any | None = None,
        *,
        time: datetime | ModelDateTime | None = None,
        logger: Logger | None = None,
    ) -> "RuntimeComponentState":
        """Advance this non-differentiable host adapter by one runtime step."""

        raise NotImplementedError


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
