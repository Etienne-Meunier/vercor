from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from vercor.components.contracts import (
    AuthorStepCallable,
    ComponentHooks,
    ComponentStepReturn,
    FieldSpec,
)
from vercor.components._callable_wrappers import (
    _CallableRuntimeMixin,
    callable_component_options,
)
from vercor.components.base import Component
from vercor.components._lifecycle import ComponentLifecycleHooks
from vercor.exceptions import ComponentError
from vercor.grids import RectilinearGrid
from vercor.output.adapters import OutputSpec
from vercor.settings import Settings

if TYPE_CHECKING:
    from vercor.components.contexts import StepContext
    from vercor.types import RuntimeArray


class HostComponent(Component):
    """Base class for host-backed adapters that cannot run inside JAX scan."""

    def _requires_host_runtime(self) -> bool:
        """Return whether this component requires the host runtime path."""

        return True

    @classmethod
    def from_step(
        cls,
        name: str,
        grid: RectilinearGrid,
        step: AuthorStepCallable,
        *,
        spec: FieldSpec | None = None,
        payload: Any | None = None,
        settings: Settings | None = None,
        hooks: ComponentHooks | None = None,
        output: OutputSpec | None = None,
    ) -> "HostComponent":
        """Create a host-runtime component from a Python step callable."""

        options = callable_component_options(
            step,
            spec=spec,
            payload=payload,
            hooks=hooks,
        )
        return _CallableHostRuntimeComponent(
            name=name,
            grid=grid,
            step=options.step,
            payload=options.payload,
            settings=settings,
            output=output,
            field_spec=options.field_spec,
            lifecycle_hooks=options.lifecycle_hooks,
        )

    def step(
        self,
        fields: Mapping[str, "RuntimeArray"],
        context: "StepContext",
        payload: Any | None = None,
    ) -> ComponentStepReturn:
        """Return field updates for one host-runtime step."""

        _ = fields, context, payload
        raise ComponentError(
            f"Host component '{self.name}' must implement step(...) or be created "
            "with HostComponent.from_step(...)."
        )


class _CallableHostRuntimeComponent(_CallableRuntimeMixin, HostComponent):
    """Host-runtime component backed by an author-provided step callable."""

    def __init__(
        self,
        name: str,
        grid: RectilinearGrid,
        *,
        step: AuthorStepCallable,
        payload: Any | None,
        settings: Settings | None,
        output: OutputSpec | None,
        field_spec: FieldSpec,
        lifecycle_hooks: ComponentLifecycleHooks,
    ) -> None:
        if settings is None:
            Component.__init__(self, name=name, grid=grid, output=output)
        else:
            Component.__init__(
                self,
                name=name,
                grid=grid,
                settings=settings,
                output=output,
            )
        self._initialize_callable_runtime(
            step=step,
            payload=payload,
            field_spec=field_spec,
            lifecycle_hooks=lifecycle_hooks,
        )

    def step(
        self,
        fields: Mapping[str, "RuntimeArray"],
        context: "StepContext",
        payload: Any | None = None,
    ) -> ComponentStepReturn:
        """Return field updates from the callable-backed host component step."""

        return self._step(fields, context, payload)


__all__ = ["HostComponent"]
