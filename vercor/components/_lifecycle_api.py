from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

import vercor.components._runtime_fields as _runtime_field_adapters
import vercor.components._runtime_validation as _runtime_field_validation
from vercor.components.contracts import (
    PrefillContext,
    PrefillResult,
    ValidationContext,
)
from vercor.state import ComponentState
from vercor.types import RuntimeArray

if TYPE_CHECKING:
    from vercor.components.base import Component
    from vercor.components.contexts import SetupContext
    from vercor._runtime.contracts import ExchangeContract
    from vercor._runtime.state import ComponentRuntimeState


class ComponentLifecycleMixin:
    """Default component lifecycle hook dispatch used by factory and subclasses."""

    def initialize(
        self,
        context: "SetupContext",
    ) -> None:
        """Optionally initialize component-owned runtime data before coupling."""

        component = cast("Component", self)
        hook = component._lifecycle_hooks.initialize
        if hook is not None:
            hook(component, context)
            return
        component.seed_declared_defaults(context.settings)

    def create_runtime_payload(self) -> Any | None:
        """Return optional immutable payload carried by runtime component state."""

        component = cast("Component", self)
        hook = component._lifecycle_hooks.create_payload
        if hook is not None:
            return hook(component)
        return self._default_runtime_payload()

    def _default_runtime_payload(self) -> Any | None:
        """Return the payload used when no lifecycle hook is installed."""

        return None

    def prefill_runtime_state_fields(
        self,
        data: dict[str, RuntimeArray],
        received: dict[str, RuntimeArray],
        sent: dict[str, RuntimeArray],
        contract: "ExchangeContract",
    ) -> None:
        """Optionally pre-seed fields required by runtime execution."""

        component = cast("Component", self)
        hook = component._lifecycle_hooks.prefill
        if hook is not None:
            result = hook(
                component,
                PrefillContext(
                    fields=MappingProxyType(data),
                    received=MappingProxyType(received),
                    sent=MappingProxyType(sent),
                    receives=contract.receives,
                    sends=contract.sends,
                ),
            )
            _apply_prefill_result(result, data, received, sent)
            return
        _runtime_field_adapters.prefill_declared_runtime_fields(component, data)
        _ = received, sent, contract

    def validate_runtime_state(
        self,
        component_state: "ComponentRuntimeState",
        contract: "ExchangeContract",
    ) -> None:
        """Optionally validate component-specific runtime fields before execution."""

        component = cast("Component", self)
        hook = component._lifecycle_hooks.validate
        if hook is not None:
            hook(
                component,
                ValidationContext(
                    state=ComponentState._from_runtime(
                        component.name,
                        component.grid,
                        component_state,
                    ),
                    payload=component_state.payload,
                    receives=contract.receives,
                    sends=contract.sends,
                ),
            )
            return
        _ = contract
        _runtime_field_validation.validate_declared_runtime_fields(
            component,
            component_state,
        )


def _apply_prefill_result(
    result: PrefillResult | None,
    data: dict[str, RuntimeArray],
    received: dict[str, RuntimeArray],
    sent: dict[str, RuntimeArray],
) -> None:
    """Apply field updates returned by a public prefill hook."""

    if result is None:
        return
    data.update(result.fields)
    received.update(result.received)
    sent.update(result.sent)


__all__ = ["ComponentLifecycleMixin"]
