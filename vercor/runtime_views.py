from __future__ import annotations

from dataclasses import dataclass, field

from vercor.grid import RectilinearGrid
from vercor.runtime import RuntimeComponentState, RuntimeFieldStore


@dataclass(frozen=True)
class RuntimeComponentView:
    """Explicit component metadata plus runtime fields for diagnostics/output."""

    name: str
    grid: RectilinearGrid
    data: RuntimeFieldStore = field(default_factory=RuntimeFieldStore.empty)
    incoming: RuntimeFieldStore = field(default_factory=RuntimeFieldStore.empty)
    outgoing: RuntimeFieldStore = field(default_factory=RuntimeFieldStore.empty)

    @classmethod
    def from_component_state(
        cls,
        name: str,
        grid: RectilinearGrid,
        component_state: RuntimeComponentState,
    ) -> "RuntimeComponentView":
        """Create a field view from component metadata and runtime state."""

        return cls(
            name=name,
            grid=grid,
            data=component_state.data,
            incoming=component_state.incoming,
            outgoing=component_state.outgoing,
        )
