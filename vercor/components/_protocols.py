from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class HostRuntimeExecutionProtocol(Protocol):
    """Private structural contract for components that require host stepping."""

    def _requires_host_runtime(self) -> bool:
        """Return whether this component requires Python host runtime stepping."""
        ...


__all__ = [
    "HostRuntimeExecutionProtocol",
]
