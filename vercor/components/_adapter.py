from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vercor.components.base import Component
from vercor.components.contracts import (
    ComponentLike,
    ComponentSpec,
    ComponentStepReturn,
    FieldImportPolicy,
)
from vercor.components.contexts import SetupContext, StepContext
from vercor.exceptions import ComponentError
from vercor.grids import RectilinearGrid
from vercor.types import RuntimeArray


class _ComponentAdapter(Component):
    """Internal adapter for structural user components."""

    def __init__(
        self,
        component: ComponentLike,
        initial_fields: Mapping[str, RuntimeArray],
    ) -> None:
        self._component = component
        super().__init__(
            name=component.name,
            grid=component.grid,
            spec=component.spec,
        )
        self._refresh_from_component(initial_fields)

    def _refresh_from_component(
        self,
        initial_fields: Mapping[str, RuntimeArray],
    ) -> None:
        """Synchronize public component fields/spec/grid into internal storage."""

        self.name = self._component.name
        self.grid = self._component.grid
        self._spec = self._component.spec
        self._lifecycle_hooks = self._spec.lifecycle
        self._import_policy = getattr(
            self._component,
            "import_policy",
            FieldImportPolicy(),
        )
        self._data = {}
        self.seed_fields(initial_fields)

    def _lifecycle_hook_owner(self) -> ComponentLike:
        """Return the original structural object for public lifecycle hooks."""

        return self._component

    def initialize(self, context: SetupContext) -> None:
        """Initialize the wrapped component and resync its public setup data."""

        self._component.initialize(context)
        self._refresh_from_component(_validate_component_like(self._component))
        initialize_hook = self._lifecycle_hooks.initialize
        if initialize_hook is not None:
            initialize_hook(self._component, context)
        self._refresh_from_component(_validate_component_like(self._component))

    def step(
        self,
        fields: Mapping[str, RuntimeArray],
        context: StepContext,
        payload: Any | None = None,
    ) -> ComponentStepReturn:
        """Delegate one runtime step to the wrapped structural component."""

        return self._component.step(fields, context, payload)


def normalize_component(component: ComponentLike) -> Component:
    """Return an internal component object for public component-like input."""

    if isinstance(component, Component):
        return component

    initial_fields = _validate_component_like(component)
    return _ComponentAdapter(component, initial_fields)


def _validate_component_like(component: object) -> Mapping[str, RuntimeArray]:
    """Validate the public structural component contract before adapting."""

    missing = [
        attribute
        for attribute in (
            "name",
            "grid",
            "spec",
            "initial_fields",
            "initialize",
            "step",
        )
        if not hasattr(component, attribute)
    ]
    if missing:
        missing_names = ", ".join(missing)
        raise ComponentError(
            f"Component-like object {component.__class__.__name__!r} is missing "
            f"required public attribute(s): {missing_names}."
        )
    name = getattr(component, "name")
    if not isinstance(name, str) or not name.strip():
        raise ComponentError(
            f"Component-like object {component.__class__.__name__!r} has invalid "
            "name; expected a non-empty string."
        )
    grid = getattr(component, "grid")
    if not isinstance(grid, RectilinearGrid):
        raise ComponentError(
            f"Component-like object {component.__class__.__name__!r} has invalid "
            "grid; expected RectilinearGrid."
        )
    spec = getattr(component, "spec")
    if not isinstance(spec, ComponentSpec):
        raise ComponentError(
            f"Component-like object {component.__class__.__name__!r} has invalid "
            "spec; expected ComponentSpec."
        )
    for method_name in ("initial_fields", "initialize", "step"):
        if not callable(getattr(component, method_name)):
            raise ComponentError(
                f"Component-like object {component.__class__.__name__!r} has "
                f"invalid {method_name}; expected a callable."
            )
    initial_fields = getattr(component, "initial_fields")()
    if not isinstance(initial_fields, Mapping):
        raise ComponentError(
            f"Component-like object {component.__class__.__name__!r} returned "
            "invalid initial_fields; expected a mapping of field names to arrays."
        )
    return initial_fields


__all__ = ["normalize_component"]
