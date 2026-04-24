from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp

from vercor.components.slab.atmosphere import (
    _bulk_flux_step,
    _default_sea_surface_temperature,
    _surface_wind_10m,
)
from vercor.components.slab.land import _update_soil_moisture
from vercor.components.slab.ocean import (
    _REFERENCE_SEA_SURFACE_TEMPERATURE,
    _advance_sea_surface_temperature,
)
from vercor.components.slab.seaice import _diagnose_ice_fraction
from vercor.exceptions import ExchangerError
from vercor.exchange import Exchange
from vercor.types import RuntimeArray

_SUPPORTED_DATA_COMPONENTS = {
    ("vercor.components.data.era5_atmosphere", "ERA5Atmosphere"),
    ("vercor.components.data.era5_ocean", "ERA5Ocean"),
    ("vercor.components.data.era5_land", "ERA5Land"),
    ("vercor.components.data.erainterim_ocean", "ERAInterimOcean"),
    ("vercor.components.data.jcm_land", "JCMLand"),
}


def exchange_key_name(source: str, destination: str, interpolation_type: str) -> str:
    """Return a stable field-store key for exchange metadata arrays."""

    return f"{source}|{destination}|{interpolation_type}"


def is_supported_differentiable_component(component: Any) -> bool:
    """Return whether ``component`` can run in the pure differentiable runtime."""

    return _is_slab_component(component) or _is_supported_data_component(component)


def _is_slab_component(component: Any) -> bool:
    module_name: str = component.__class__.__module__
    return module_name.startswith("vercor.components.slab.")


def _is_supported_data_component(component: Any) -> bool:
    return (
        component.__class__.__module__,
        component.__class__.__name__,
    ) in _SUPPORTED_DATA_COMPONENTS


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class RuntimeStepInfo:
    """Precomputed time-selection metadata for one differentiable runtime step."""

    monthly_index_left: RuntimeArray
    monthly_index_right: RuntimeArray
    monthly_weight_left: RuntimeArray
    monthly_weight_right: RuntimeArray
    daily_index: RuntimeArray

    @classmethod
    def from_sequences(
        cls,
        monthly_index_left: Sequence[int],
        monthly_index_right: Sequence[int],
        monthly_weight_left: Sequence[float],
        monthly_weight_right: Sequence[float],
        daily_index: Sequence[int],
    ) -> "RuntimeStepInfo":
        """Create scan metadata from host-precomputed index and weight arrays."""

        return cls(
            monthly_index_left=jnp.asarray(monthly_index_left, dtype=jnp.int32),
            monthly_index_right=jnp.asarray(monthly_index_right, dtype=jnp.int32),
            monthly_weight_left=jnp.asarray(monthly_weight_left, dtype=jnp.float_),
            monthly_weight_right=jnp.asarray(monthly_weight_right, dtype=jnp.float_),
            daily_index=jnp.asarray(daily_index, dtype=jnp.int32),
        )

    def tree_flatten(self) -> tuple[tuple[RuntimeArray, ...], None]:
        return (
            (
                self.monthly_index_left,
                self.monthly_index_right,
                self.monthly_weight_left,
                self.monthly_weight_right,
                self.daily_index,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(
        cls, aux_data: None, children: tuple[RuntimeArray, ...]
    ) -> "RuntimeStepInfo":
        _ = aux_data
        (
            monthly_index_left,
            monthly_index_right,
            monthly_weight_left,
            monthly_weight_right,
            daily_index,
        ) = children
        return cls(
            monthly_index_left=monthly_index_left,
            monthly_index_right=monthly_index_right,
            monthly_weight_left=monthly_weight_left,
            monthly_weight_right=monthly_weight_right,
            daily_index=daily_index,
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class RuntimeFieldStore:
    """Immutable named array store used by the differentiable runtime."""

    field_names: tuple[str, ...]
    values: tuple[RuntimeArray, ...]

    @classmethod
    def empty(cls) -> "RuntimeFieldStore":
        """Create an empty field store."""

        return cls(field_names=(), values=())

    @classmethod
    def from_mapping(cls, fields: Mapping[str, RuntimeArray]) -> "RuntimeFieldStore":
        """Create a field store from a mapping while preserving insertion order."""

        return cls(
            field_names=tuple(fields.keys()),
            values=tuple(jnp.asarray(value) for value in fields.values()),
        )

    def tree_flatten(self) -> tuple[tuple[RuntimeArray, ...], tuple[str, ...]]:
        return self.values, self.field_names

    @classmethod
    def tree_unflatten(
        cls, aux_data: tuple[str, ...], children: tuple[RuntimeArray, ...]
    ) -> "RuntimeFieldStore":
        return cls(field_names=aux_data, values=children)

    def get(self, name: str) -> RuntimeArray:
        """Return a field by name."""

        try:
            index = self.field_names.index(name)
        except ValueError as exc:
            raise KeyError(f"Runtime field {name!r} not found") from exc
        return self.values[index]

    def set(self, name: str, value: RuntimeArray) -> "RuntimeFieldStore":
        """Return a new store with ``name`` replaced or appended."""

        value_array = jnp.asarray(value)
        if name not in self.field_names:
            return RuntimeFieldStore(
                field_names=(*self.field_names, name),
                values=(*self.values, value_array),
            )

        values = tuple(
            value_array if field_name == name else current
            for field_name, current in zip(self.field_names, self.values)
        )
        return RuntimeFieldStore(field_names=self.field_names, values=values)

    def merge(self, other: "RuntimeFieldStore") -> "RuntimeFieldStore":
        """Return a new store with fields from ``other`` replacing this store."""

        out = self
        for name, value in zip(other.field_names, other.values):
            out = out.set(name, value)
        return out

    def subset(self, names: Sequence[str]) -> "RuntimeFieldStore":
        """Return a new store containing fields in ``names`` order."""

        return RuntimeFieldStore.from_mapping({name: self.get(name) for name in names})

    def to_mapping(self) -> dict[str, RuntimeArray]:
        """Return a dictionary view of the store values."""

        return dict(zip(self.field_names, self.values))


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class RuntimeComponentState:
    """Immutable differentiable state for one component."""

    name: str
    data: RuntimeFieldStore
    incoming: RuntimeFieldStore
    outgoing: RuntimeFieldStore
    fields_to_import: tuple[str, ...]
    fields_to_export: tuple[str, ...]

    def tree_flatten(
        self,
    ) -> tuple[
        tuple[RuntimeFieldStore, RuntimeFieldStore, RuntimeFieldStore],
        tuple[str, tuple[str, ...], tuple[str, ...]],
    ]:
        children = (self.data, self.incoming, self.outgoing)
        aux_data = (self.name, self.fields_to_import, self.fields_to_export)
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: tuple[str, tuple[str, ...], tuple[str, ...]],
        children: tuple[RuntimeFieldStore, RuntimeFieldStore, RuntimeFieldStore],
    ) -> "RuntimeComponentState":
        name, fields_to_import, fields_to_export = aux_data
        data, incoming, outgoing = children
        return cls(
            name=name,
            data=data,
            incoming=incoming,
            outgoing=outgoing,
            fields_to_import=fields_to_import,
            fields_to_export=fields_to_export,
        )

    def with_data(self, data: RuntimeFieldStore) -> "RuntimeComponentState":
        """Return this component state with replaced data."""

        return RuntimeComponentState(
            name=self.name,
            data=data,
            incoming=self.incoming,
            outgoing=self.outgoing,
            fields_to_import=self.fields_to_import,
            fields_to_export=self.fields_to_export,
        )

    def with_incoming(self, incoming: RuntimeFieldStore) -> "RuntimeComponentState":
        """Return this component state with replaced incoming fields."""

        return RuntimeComponentState(
            name=self.name,
            data=self.data,
            incoming=incoming,
            outgoing=self.outgoing,
            fields_to_import=self.fields_to_import,
            fields_to_export=self.fields_to_export,
        )

    def with_outgoing(self, outgoing: RuntimeFieldStore) -> "RuntimeComponentState":
        """Return this component state with replaced outgoing fields."""

        return RuntimeComponentState(
            name=self.name,
            data=self.data,
            incoming=self.incoming,
            outgoing=outgoing,
            fields_to_import=self.fields_to_import,
            fields_to_export=self.fields_to_export,
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class RuntimeCouplerState:
    """Immutable differentiable state for the VerCOR runtime core."""

    components: tuple[RuntimeComponentState, ...]
    fractional_masks: RuntimeFieldStore
    binary_masks: RuntimeFieldStore

    @property
    def component_names(self) -> tuple[str, ...]:
        """Return component names in runtime-state order."""

        return tuple(component.name for component in self.components)

    def tree_flatten(
        self,
    ) -> tuple[
        tuple[tuple[RuntimeComponentState, ...], RuntimeFieldStore, RuntimeFieldStore],
        None,
    ]:
        return (self.components, self.fractional_masks, self.binary_masks), None

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: None,
        children: tuple[
            tuple[RuntimeComponentState, ...], RuntimeFieldStore, RuntimeFieldStore
        ],
    ) -> "RuntimeCouplerState":
        _ = aux_data
        components, fractional_masks, binary_masks = children
        return cls(
            components=components,
            fractional_masks=fractional_masks,
            binary_masks=binary_masks,
        )

    def get_component_state(self, name: str) -> RuntimeComponentState:
        """Return one component state by name."""

        for component in self.components:
            if component.name == name:
                return component
        raise KeyError(f"Runtime component {name!r} not found")

    def set_component_state(
        self, component_state: RuntimeComponentState
    ) -> "RuntimeCouplerState":
        """Return a new coupler state with one component replaced."""

        components = tuple(
            component_state if component.name == component_state.name else component
            for component in self.components
        )
        return RuntimeCouplerState(
            components=components,
            fractional_masks=self.fractional_masks,
            binary_masks=self.binary_masks,
        )

    def get_fractional_mask(
        self, source: str, destination: str, interpolation_type: str
    ) -> RuntimeArray:
        """Return the fractional mask for an exchange."""

        return self.fractional_masks.get(
            exchange_key_name(source, destination, interpolation_type)
        )


def dispatch_component_exchanges(
    state: RuntimeCouplerState,
    destination_name: str,
    exchanges: Sequence[Exchange],
    regridders: Mapping[tuple[str, str, str], Any],
) -> RuntimeCouplerState:
    """Dispatch all exchanges targeting one destination component."""

    destination_component = state.get_component_state(destination_name)
    destination_incoming = destination_component.incoming

    for exchange in exchanges:
        if exchange.destination != destination_name:
            continue

        source_component = state.get_component_state(exchange.source)
        source_fields = source_component.outgoing
        key = (exchange.source, exchange.destination, exchange.interpolation_type)
        regrid = regridders[key]
        fractional_mask = state.get_fractional_mask(*key)

        for field_name in exchange.field_names:
            if isinstance(field_name, tuple):
                if not all(name in source_fields.field_names for name in field_name):
                    raise ExchangerError(
                        f"Not all fields in vector {field_name} are present in source fields"
                    )
                u_vector, v_vector = regrid(
                    source_fields.get(field_name[0]),
                    source_fields.get(field_name[1]),
                )
                destination_incoming = destination_incoming.set(field_name[0], u_vector)
                destination_incoming = destination_incoming.set(field_name[1], v_vector)
            else:
                if field_name not in source_fields.field_names:
                    raise ExchangerError(
                        f"Field {field_name} not present in source fields"
                    )
                scalar = regrid(source_fields.get(field_name)) * fractional_mask
                destination_incoming = destination_incoming.set(field_name, scalar)

    destination_component = destination_component.with_incoming(destination_incoming)
    return state.set_component_state(destination_component)


def receive_component_fields(
    component_state: RuntimeComponentState,
) -> RuntimeComponentState:
    """Move imported incoming fields into component data."""

    data = component_state.data
    for field_name in component_state.fields_to_import:
        data = data.set(field_name, component_state.incoming.get(field_name))
    return component_state.with_data(data)


def _select_runtime_field_for_send(
    component: Any,
    component_state: RuntimeComponentState,
    field_name: str,
    step_info: RuntimeStepInfo | None,
) -> RuntimeArray:
    field = component_state.data.get(field_name)
    if step_info is None:
        return field

    settings = component.settings
    if settings.apply_time_interpolation:
        arr = jnp.asarray(field)
        left = jnp.take(arr, step_info.monthly_index_left, axis=-1)
        right = jnp.take(arr, step_info.monthly_index_right, axis=-1)
        return (
            step_info.monthly_weight_left * left
            + step_info.monthly_weight_right * right
        ).swapaxes(-2, -1)

    if settings.get_field_time_slice:
        return jnp.take(jnp.asarray(field), step_info.daily_index, axis=0)

    return field


def send_component_fields(
    component_state: RuntimeComponentState,
    component: Any | None = None,
    step_info: RuntimeStepInfo | None = None,
) -> RuntimeComponentState:
    """Move exported component data into outgoing fields."""

    outgoing = component_state.outgoing
    for field_name in component_state.fields_to_export:
        field_value = (
            component_state.data.get(field_name)
            if component is None
            else _select_runtime_field_for_send(
                component,
                component_state,
                field_name,
                step_info,
            )
        )
        outgoing = outgoing.set(field_name, field_value)
    return component_state.with_outgoing(outgoing)


def _step_data_component_state(
    component: Any,
    component_state: RuntimeComponentState,
) -> RuntimeComponentState:
    """Return a stepped pure data-forcing component state."""

    if component.__class__.__name__ == "ERA5Atmosphere":
        data = component_state.data
        land_surface_temperature = data.get("land_surface_temperature")
        sea_surface_temperature = data.get("sea_surface_temperature")
        total_surface_temperature = jnp.nan_to_num(
            jnp.asarray(land_surface_temperature),
            nan=0.0,
        ) + jnp.nan_to_num(
            jnp.asarray(sea_surface_temperature),
            nan=0.0,
        )
        return component_state.with_data(
            data.set("total_surface_temperature", total_surface_temperature)
        )

    return component_state


def step_component_state(
    component: Any,
    component_state: RuntimeComponentState,
    dt_seconds: float,
) -> RuntimeComponentState:
    """Return a stepped component state on the differentiable runtime path.

    VerCOR-owned slab components compute their pure kernels here. Supported
    data-forcing components either replay forcing through runtime sends or run
    small JAX-native diagnostic updates.
    """

    class_name = component.__class__.__name__
    if _is_supported_data_component(component):
        return _step_data_component_state(component, component_state)

    if not _is_slab_component(component):
        raise NotImplementedError(
            "Differentiable runtime currently supports VerCOR slab components "
            "and pure data-forcing components only"
        )

    data = component_state.data
    if class_name == "Atmosphere":
        temperature_2m = data.get("temperature_2m")
        try:
            sea_surface_temperature = data.get("sea_surface_temperature")
        except KeyError:
            sea_surface_temperature = _default_sea_surface_temperature(temperature_2m)

        sensible_heat_flux, latent_heat_flux, updated_temperature_2m = _bulk_flux_step(
            temperature_2m, sea_surface_temperature
        )
        u_velocity_10m, v_velocity_10m = _surface_wind_10m(
            component.grid.latitude, component.grid.longitude
        )
        data = data.set("u_velocity_10m", u_velocity_10m)
        data = data.set("v_velocity_10m", v_velocity_10m)
        data = data.set("sensible_heat_flux", sensible_heat_flux)
        data = data.set("latent_heat_flux", latent_heat_flux)
        data = data.set("temperature_2m", updated_temperature_2m)
        return component_state.with_data(data)

    if class_name == "Ocean":
        sea_surface_temperature = data.get("sea_surface_temperature")
        try:
            sensible_heat_flux = data.get("sensible_heat_flux")
        except KeyError:
            sensible_heat_flux = jnp.zeros_like(sea_surface_temperature)
        try:
            latent_heat_flux = data.get("latent_heat_flux")
        except KeyError:
            latent_heat_flux = jnp.zeros_like(sea_surface_temperature)

        updated_sst = _advance_sea_surface_temperature(
            sea_surface_temperature,
            sensible_heat_flux,
            latent_heat_flux,
            dt_seconds,
            component.rho,
            component.cp,
            component.H,
            component.lambda_relax,
            _REFERENCE_SEA_SURFACE_TEMPERATURE,
        )
        return component_state.with_data(
            data.set("sea_surface_temperature", updated_sst)
        )

    if class_name == "Land":
        soil_moisture = data.get("soil_moisture")
        try:
            latent_heat_flux = data.get("latent_heat_flux")
        except KeyError:
            latent_heat_flux = jnp.zeros_like(soil_moisture)
        updated_soil_moisture = _update_soil_moisture(
            soil_moisture,
            latent_heat_flux,
            dt_seconds,
        )
        return component_state.with_data(
            data.set("soil_moisture", updated_soil_moisture)
        )

    if class_name == "SeaIce":
        sea_surface_temperature = data.get("sea_surface_temperature")
        ice_fraction = _diagnose_ice_fraction(sea_surface_temperature)
        return component_state.with_data(data.set("ice_fraction", ice_fraction))

    raise NotImplementedError(f"No differentiable runtime step for {class_name}")


def step_slab_component_state(
    component: Any,
    component_state: RuntimeComponentState,
    dt_seconds: float,
) -> RuntimeComponentState:
    """Return a stepped slab component state.

    This compatibility wrapper delegates to ``step_component_state``.
    """

    if not _is_slab_component(component):
        raise NotImplementedError(
            "Differentiable runtime currently supports VerCOR slab components only"
        )
    return step_component_state(component, component_state, dt_seconds)
