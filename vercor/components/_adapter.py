"""Single private normalization bridge for protocol-first components."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import jax
import jax.numpy as jnp

from vercor.components._callable_wrappers import normalize_component_step_callable
from vercor.components._contracts import (
    normalize_field_values,
    validate_declared_updates,
)
from vercor.components.contracts import (
    Component,
    ComponentSpec,
    PrefillContext,
    PrefillResult,
    SetupContext,
    SetupResult,
    ValidationContext,
    _ComponentStepCallable,
    _ComponentStepReturn,
)
from vercor.components.contexts import StepContext
from vercor.dtypes import DTypePolicy, jax_zeros
from vercor.exceptions import ComponentError
from vercor.grids import RectilinearGrid
from vercor.state import ComponentState
from vercor.types import RuntimeArray


@dataclass(frozen=True)
class _ComponentDeclaration:
    """Validated public declaration retained until runtime preparation."""

    component: Component
    name: str
    grid: RectilinearGrid
    spec: ComponentSpec
    step: _ComponentStepCallable


@dataclass(frozen=True)
class _ComponentBinding:
    """Immutable runtime binding produced once after setup and dtype selection."""

    _component: Component
    name: str
    grid: RectilinearGrid
    spec: ComponentSpec
    _step: _ComponentStepCallable
    _data: Mapping[str, RuntimeArray]
    _payload: Any | None
    _dtype_policy: DTypePolicy

    @property
    def field_names(self) -> tuple[str, ...]:
        """Return prepared field names in stable insertion order."""

        return tuple(self._data)

    def step(
        self,
        fields: Mapping[str, RuntimeArray],
        context: StepContext,
        payload: Any | None = None,
    ) -> _ComponentStepReturn:
        """Delegate one step through the normalized public callback."""

        return self._step(fields, context, payload)

    def _create_runtime_payload(self) -> Any | None:
        """Return the copy-owned payload produced during setup."""

        return _copy_owned_pytree(self._payload)

    def _prefill_runtime_state_fields(
        self,
        data: dict[str, RuntimeArray],
        received: dict[str, RuntimeArray],
        sent: dict[str, RuntimeArray],
        contract: Any,
    ) -> None:
        """Apply an optional public prefill result to runtime stores."""

        hook = self.spec.lifecycle.prefill
        if hook is None:
            return
        result = hook(
            self._component,
            PrefillContext(
                fields=data,
                received=received,
                sent=sent,
                receives=contract.receives,
                sends=contract.sends,
            ),
        )
        if result is None:
            return
        if not isinstance(result, PrefillResult):
            raise ComponentError(
                f"Component '{self.name}' prefill must return PrefillResult or None; "
                f"got {type(result).__name__}."
            )
        validate_declared_updates(
            self.name,
            result.fields,
            (*self.spec.inputs, *self.spec.outputs),
            phase="prefill",
        )
        normalized_fields = normalize_field_values(
            component_name=self.name,
            grid=self.grid,
            fields=result.fields,
            policy=self._dtype_policy,
        )
        normalized_received = _normalize_prefill_contract_store(
            component=self,
            updates=result.received,
            declared=contract.receives,
            store_name="received",
        )
        normalized_sent = _normalize_prefill_contract_store(
            component=self,
            updates=result.sent,
            declared=contract.sends,
            store_name="sent",
        )
        data.update(normalized_fields)
        received.update(normalized_received)
        sent.update(normalized_sent)

    def _validate_runtime_state(self, component_state: Any, contract: Any) -> None:
        """Run optional author validation with a public immutable state view."""

        hook = self.spec.lifecycle.validate
        if hook is None:
            return
        hook(
            self._component,
            ValidationContext(
                state=ComponentState._from_runtime(
                    self.name,
                    self.grid,
                    component_state,
                ),
                payload=_copy_owned_pytree(component_state.payload),
                receives=contract.receives,
                sends=contract.sends,
            ),
        )


def validate_component_contract(component: object) -> None:
    """Validate the exact structural component contract immediately."""

    missing = [
        name
        for name in ("name", "grid", "spec", "step")
        if not hasattr(component, name)
    ]
    if missing:
        raise ComponentError(
            f"Component object {component.__class__.__name__!r} is missing required "
            f"attribute(s): {', '.join(missing)}."
        )
    name = getattr(component, "name")
    if not isinstance(name, str) or not name.strip():
        raise ComponentError("Component name must be a non-empty string.")
    if not isinstance(getattr(component, "grid"), RectilinearGrid):
        raise ComponentError(f"Component '{name}' grid must be RectilinearGrid.")
    if not isinstance(getattr(component, "spec"), ComponentSpec):
        raise ComponentError(f"Component '{name}' spec must be ComponentSpec.")
    if not callable(getattr(component, "step")):
        raise ComponentError(f"Component '{name}' step must be callable.")


def normalize_component(component: Component) -> _ComponentDeclaration:
    """Validate a public component once without running lifecycle setup."""

    validate_component_contract(component)
    normalized_step = getattr(component, "_normalized_step", None)
    return _ComponentDeclaration(
        component=component,
        name=component.name,
        grid=component.grid,
        spec=component.spec,
        step=(
            normalize_component_step_callable(component.step)
            if normalized_step is None
            else normalized_step
        ),
    )


def prepare_component(
    declaration: _ComponentDeclaration,
    context: SetupContext,
    dtype: DTypePolicy,
) -> _ComponentBinding:
    """Run setup once and freeze normalized fields/payload in a runtime binding."""

    grid = declaration.grid.with_precision(dtype)
    data = normalize_field_values(
        component_name=declaration.name,
        grid=grid,
        fields=declaration.spec.initial_fields,
        policy=dtype,
    )
    zeros = jax_zeros(grid.shape, dtype)
    for field_name in declaration.spec.outputs:
        data.setdefault(field_name, zeros)

    hook = declaration.spec.lifecycle.setup
    result = None if hook is None else hook(declaration.component, context)
    if result is not None and not isinstance(result, SetupResult):
        raise ComponentError(
            f"Component '{declaration.name}' setup must return SetupResult or None; "
            f"got {type(result).__name__}."
        )
    if result is not None:
        validate_declared_updates(
            declaration.name,
            result.fields,
            (*declaration.spec.inputs, *declaration.spec.outputs),
            phase="setup",
        )
        data.update(
            normalize_field_values(
                component_name=declaration.name,
                grid=grid,
                fields=result.fields,
                policy=dtype,
            )
        )
    payload = None if result is None else _copy_owned_pytree(result.payload)
    return _ComponentBinding(
        _component=declaration.component,
        name=declaration.name,
        grid=grid,
        spec=declaration.spec,
        _step=declaration.step,
        _data=MappingProxyType(dict(data)),
        _payload=payload,
        _dtype_policy=dtype,
    )


def _copy_owned_pytree(value: Any) -> Any:
    """Copy mutable array leaves while preserving arbitrary PyTree structure."""

    def copy_leaf(leaf: Any) -> Any:
        if isinstance(
            leaf,
            (str, bytes, int, float, complex, bool, type(None), frozenset, type),
        ):
            return leaf
        if isinstance(leaf, (jax.Array, jax.core.Tracer)):
            return leaf
        copy = getattr(leaf, "copy", None)
        module = type(leaf).__module__
        if callable(copy) and module.startswith("numpy"):
            return copy()
        try:
            return deepcopy(leaf)
        except Exception as exc:
            raise ComponentError(
                "Component payload leaves must be immutable, JAX arrays/tracers, "
                "copyable NumPy values, or deepcopy-compatible objects; got "
                f"{type(leaf).__name__}."
            ) from exc

    return jax.tree_util.tree_map(copy_leaf, value)


def _normalize_prefill_contract_store(
    *,
    component: _ComponentBinding,
    updates: Mapping[str, object],
    declared: tuple[str, ...],
    store_name: str,
) -> dict[str, RuntimeArray]:
    """Validate and normalize exchange-store updates from one prefill hook."""

    allowed = set(declared)
    undeclared = next((name for name in updates if name not in allowed), None)
    if undeclared is not None:
        raise ComponentError(
            f"Component '{component.name}' prefill {store_name} field "
            f"'{undeclared}' is not present in its exchange contract."
        )
    try:
        normalized = normalize_field_values(
            component_name=component.name,
            grid=component.grid,
            fields=updates,
            policy=component._dtype_policy,
        )
    except ComponentError as exc:
        invalid_name = None
        invalid_shape = None
        for name, value in updates.items():
            try:
                shape = jnp.asarray(value).shape
            except (TypeError, ValueError):
                raise ComponentError(
                    f"Component '{component.name}' prefill {store_name} field "
                    f"'{name}' must be a real numeric scalar or grid array."
                ) from exc
            if shape not in ((), component.grid.shape):
                invalid_name = name
                invalid_shape = shape
                break
        if invalid_name is None:
            raise
        raise ComponentError(
            f"Component '{component.name}' prefill {store_name} field "
            f"'{invalid_name}' has shape {invalid_shape}; expected scalar or "
            f"grid shape {component.grid.shape}."
        ) from exc
    invalid_name = next(
        (
            name
            for name, value in normalized.items()
            if value.shape != component.grid.shape
        ),
        None,
    )
    if invalid_name is not None:
        raise ComponentError(
            f"Component '{component.name}' prefill {store_name} field "
            f"'{invalid_name}' has shape {normalized[invalid_name].shape}; expected "
            f"grid shape {component.grid.shape}."
        )
    return normalized


__all__: list[str] = []
