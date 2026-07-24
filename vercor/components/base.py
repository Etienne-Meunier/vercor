"""Callable convenience adapter for the structural component protocol."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from vercor.components._callable_wrappers import normalize_component_step_callable
from vercor.components.contracts import (
    ComponentSpec,
    _ComponentStepCallable,
    StepResult,
)
from vercor.components.contexts import StepContext
from vercor.grids import RectilinearGrid
from vercor.types import RuntimeArray


class CallableComponent:
    """Compose a structural component from an ordinary step callable.

    ``name`` must be non-empty, ``grid`` is the component layout, and ``spec``
    is the immutable author declaration (an empty declaration is used when it
    is omitted). ``step`` may accept fields only, fields plus ``StepContext``,
    or fields/context/payload. It returns declared output updates as a mapping
    or ``StepResult``. Setup and storage remain owned by the private binding.
    """

    name: str
    grid: RectilinearGrid
    spec: ComponentSpec

    def __init__(
        self,
        name: str,
        grid: RectilinearGrid,
        step: Callable[..., Mapping[str, RuntimeArray] | StepResult],
        *,
        spec: ComponentSpec | None = None,
    ) -> None:
        """Validate configuration and normalize the callable signature."""

        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")
        if not isinstance(grid, RectilinearGrid):
            raise TypeError("grid must be RectilinearGrid")
        if spec is not None and not isinstance(spec, ComponentSpec):
            raise TypeError("spec must be ComponentSpec or None")
        self.name = name
        self.grid = grid
        self.spec = ComponentSpec() if spec is None else spec
        self._normalized_step: _ComponentStepCallable = (
            normalize_component_step_callable(step)
        )

    def step(
        self,
        fields: Mapping[str, RuntimeArray],
        context: StepContext,
        payload: Any | None = None,
    ) -> Mapping[str, RuntimeArray] | StepResult:
        """Delegate one model step through the normalized callable signature."""

        return self._normalized_step(fields, context, payload)

    def __repr__(self) -> str:
        return f"CallableComponent(name={self.name!r}, grid={self.grid!r})"


__all__ = ["CallableComponent"]
