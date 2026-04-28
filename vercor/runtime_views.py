from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vercor.grid import RectilinearGrid


@dataclass(frozen=True)
class RuntimeComponentView:
    """Explicit component metadata plus runtime fields for diagnostics/output."""

    name: str
    grid: RectilinearGrid
    data: Any = field(default_factory=dict)
    incoming: Any = field(default_factory=dict)
    outgoing: Any = field(default_factory=dict)

    @classmethod
    def from_runtime_state(
        cls,
        name: str,
        component: Any,
        component_state: Any,
    ) -> "RuntimeComponentView":
        """Create a field view from a component and its runtime state."""

        return cls(
            name=name,
            grid=component.grid,
            data=component_state.data,
            incoming=component_state.incoming,
            outgoing=component_state.outgoing,
        )

    @classmethod
    def from_coupler_state(
        cls,
        coupler: Any,
        runtime_state: Any,
        name: str,
    ) -> "RuntimeComponentView":
        """Create a field view from a coupler-owned component runtime state."""

        return cls.from_runtime_state(
            name,
            coupler.components[name],
            runtime_state.get_component_state(name),
        )
