from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, final

from vercor.components.contracts import (
    AuthorFieldValues,
    AuthorStepCallable,
    ComponentHooks,
    FieldSpec,
    FieldNames,
)
from vercor.components._callable_wrappers import (
    _CallableRuntimeMixin,
    callable_component_options,
)
from vercor.components.base import Component
from vercor.components._lifecycle import ComponentLifecycleHooks
from vercor.exceptions import ComponentError
from vercor.grids import RectilinearGrid
from vercor.output.adapters import ComponentOutput
from vercor.settings import Settings

if TYPE_CHECKING:
    from vercor.components.contexts import StepContext
    from vercor.runtime.state import RuntimeComponentState


class HostComponent(Component):
    """Base class for host-backed adapters that cannot run inside JAX scan."""

    @classmethod
    def from_step(
        cls,
        name: str,
        grid: RectilinearGrid,
        step: AuthorStepCallable,
        *,
        inputs: FieldNames = (),
        outputs: FieldNames = (),
        defaults: AuthorFieldValues = None,
        payload: Any | None = None,
        settings: Settings | None = None,
        hooks: ComponentHooks | None = None,
        output: ComponentOutput | None = None,
    ) -> "HostComponent":
        """Create a host-runtime component from a Python step callable."""

        options = callable_component_options(
            step,
            inputs=inputs,
            outputs=outputs,
            defaults=defaults,
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

    @final
    def step_runtime_state(
        self,
        component_state: "RuntimeComponentState",
        context: StepContext,
    ) -> "RuntimeComponentState":
        """Reject accidental execution on the differentiable scanned runtime."""

        _ = component_state, context
        component_name = getattr(self, "name", self.__class__.__name__)
        raise ComponentError(
            f"Component '{component_name}' is host-backed and cannot run through "
            "the differentiable scanned runtime. Use Coupler.run() so VerCOR can "
            "select the host runtime path, or implement a differentiable Component."
        )

    @abstractmethod
    def step_host_runtime_state(
        self,
        component_state: "RuntimeComponentState",
        context: StepContext,
    ) -> "RuntimeComponentState":
        """Advance this non-differentiable host adapter by one runtime step."""


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
        output: ComponentOutput | None,
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

    def step_host_runtime_state(
        self,
        component_state: "RuntimeComponentState",
        context: StepContext,
    ) -> "RuntimeComponentState":
        """Advance this callable-backed host component one step."""

        return self._step_callable_runtime_state(component_state, context)


__all__ = ["HostComponent"]
