from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from vercor.components._protocols import HostRuntimeExecutionProtocol
from vercor.components._callable_wrappers import normalize_component_step_callable
from vercor.components._runtime_fields import apply_step_result, runtime_fields
from vercor.exceptions import ComponentError

if TYPE_CHECKING:
    from vercor.components.base import Component
    from vercor.components.contexts import StepContext
    from vercor.runtime.state import RuntimeComponentState


def host_component_names(
    components: Mapping[str, "Component"],
) -> list[str]:
    """Return names of components that require the Python host runtime."""

    return [
        name
        for name, component in components.items()
        if isinstance(component, HostRuntimeExecutionProtocol)
        and component._requires_host_runtime()
    ]


def step_component_runtime_state(
    component: "Component",
    component_state: "RuntimeComponentState",
    context: "StepContext",
    *,
    allow_host_runtime: bool,
) -> "RuntimeComponentState":
    """Advance ``component_state`` through the component's selected runtime path."""

    if isinstance(component, HostRuntimeExecutionProtocol):
        if not allow_host_runtime and component._requires_host_runtime():
            raise ComponentError(
                f"Component '{component.name}' is host-backed and cannot run "
                "through the differentiable scanned runtime. Use Coupler.run() "
                "so VerCOR can select the host runtime path, or implement a "
                "differentiable Component."
            )

    step = normalize_component_step_callable(component.step)
    return apply_step_result(
        component,
        component_state,
        step(
            runtime_fields(component_state),
            context,
            component_state.runtime_payload,
        ),
    )
