from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vercor.components.base import Component
from vercor.components.contracts import ComponentLike, _ComponentStepReturn
from vercor.components.contracts import FieldImportPolicy
from vercor.components.contexts import SetupContext, StepContext
from vercor.exceptions import ComponentError
from vercor.grids import RectilinearGrid
from vercor.types import RuntimeArray


class _ComponentAdapter(Component):
    """Internal adapter for structural user components."""

    def __init__(self, component: ComponentLike) -> None:
        self._component = component
        super().__init__(
            name=component.name,
            grid=component.grid,
            spec=component.spec,
        )
        self._refresh_from_component()

    def _refresh_from_component(self) -> None:
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
        self.seed_fields(self._component.initial_fields())

    def initialize(self, context: SetupContext) -> None:
        """Initialize the wrapped component and resync its public setup data."""

        self._component.initialize(context)
        self._refresh_from_component()

    def step(
        self,
        fields: Mapping[str, RuntimeArray],
        context: StepContext,
        payload: Any | None = None,
    ) -> _ComponentStepReturn:
        """Delegate one runtime step to the wrapped structural component."""

        return self._component.step(fields, context, payload)


def normalize_component(component: ComponentLike) -> Component:
    """Return an internal component object for public component-like input."""

    if isinstance(component, Component):
        return component

    _validate_component_like(component)
    return _ComponentAdapter(component)


def _validate_component_like(component: object) -> None:
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
    grid = getattr(component, "grid")
    if not isinstance(grid, RectilinearGrid):
        raise ComponentError(
            f"Component-like object {component.__class__.__name__!r} has invalid "
            "grid; expected RectilinearGrid."
        )


__all__ = ["normalize_component"]
