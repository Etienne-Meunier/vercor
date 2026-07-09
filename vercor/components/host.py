from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from vercor.components.contracts import (
    LifecycleHooks,
    ComponentSpec,
    _AuthorStepCallable,
    _ComponentStepReturn,
)
from vercor.components._callable_wrappers import (
    _CallableRuntimeMixin,
    callable_component_options,
)
from vercor.components.base import Component
from vercor.exceptions import ComponentError
from vercor.grids import RectilinearGrid
from vercor.settings import Settings

if TYPE_CHECKING:
    from vercor.components.contexts import StepContext
    from vercor.types import RuntimeArray


class HostComponent(Component):
    """Base class for host-backed adapters that cannot run inside JAX scan."""

    def __init__(
        self,
        name: str,
        grid: RectilinearGrid,
        *,
        settings: Settings | None = None,
        spec: ComponentSpec | None = None,
    ) -> None:
        """Create a host-backed component configuration shell."""

        Component.__init__(
            self,
            name=name,
            grid=grid,
            settings=settings,
            spec=_host_execution_spec(ComponentSpec() if spec is None else spec),
        )

    @classmethod
    def from_step(
        cls,
        name: str,
        grid: RectilinearGrid,
        step: _AuthorStepCallable,
        *,
        spec: ComponentSpec | None = None,
        payload: Any | None = None,
        settings: Settings | None = None,
    ) -> "HostComponent":
        """Create a host-runtime component from a Python step callable."""

        options = callable_component_options(
            step,
            spec=_host_execution_spec(ComponentSpec() if spec is None else spec),
            payload=payload,
        )
        return _CallableHostRuntimeComponent(
            name=name,
            grid=grid,
            step=options.step,
            payload=options.payload,
            settings=settings,
            spec=options.spec,
            lifecycle_hooks=options.lifecycle_hooks,
        )

    def step(
        self,
        fields: Mapping[str, "RuntimeArray"],
        context: "StepContext",
        payload: Any | None = None,
    ) -> _ComponentStepReturn:
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
        step: _AuthorStepCallable,
        payload: Any | None,
        settings: Settings | None,
        spec: ComponentSpec,
        lifecycle_hooks: LifecycleHooks,
    ) -> None:
        HostComponent.__init__(
            self,
            name=name,
            grid=grid,
            settings=settings,
            spec=spec,
        )
        self._initialize_callable_runtime(
            step=step,
            payload=payload,
            spec=self.spec,
            lifecycle_hooks=lifecycle_hooks,
        )

    def step(
        self,
        fields: Mapping[str, "RuntimeArray"],
        context: "StepContext",
        payload: Any | None = None,
    ) -> _ComponentStepReturn:
        """Return field updates from the callable-backed host component step."""

        return self._step(fields, context, payload)


def _host_execution_spec(spec: ComponentSpec) -> ComponentSpec:
    """Return ``spec`` with host execution selected."""

    if spec.execution == "host":
        return spec
    return ComponentSpec(
        inputs=spec.inputs,
        outputs=spec.outputs,
        defaults=spec.defaults,
        execution="host",
        lifecycle=spec.lifecycle,
        output=spec.output,
    )


__all__ = ["HostComponent"]
