from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, TypeAlias, runtime_checkable

from vercor.dtypes import DTypePolicy

if TYPE_CHECKING:
    from vercor._runtime.run_context import RuntimeRunContext
    from vercor.state import RunState


@runtime_checkable
class ExecutionBackend(Protocol):
    """Public protocol for custom coupler runtime execution backends."""

    def run(
        self,
        state: "RunState",
        *,
        context: "RuntimeRunContext",
    ) -> "RunState":
        """Run a prepared runtime state and return the final state."""
        ...


ExecutionMode: TypeAlias = Literal["auto", "jax", "host"]


@dataclass(frozen=True)
class SurfaceMaskPolicy:
    """Policy for the default atmosphere/ocean/land surface-mask topology."""

    mode: Literal["auto", "required", "disabled"] = "auto"
    atmosphere: str = "ATM"
    ocean: str = "OCN"
    land: str = "LND"

    def __post_init__(self) -> None:
        """Validate surface-mask policy values."""

        if self.mode not in ("auto", "required", "disabled"):
            raise ValueError("mode must be one of 'auto', 'required', 'disabled'")
        for role, name in (
            ("atmosphere", self.atmosphere),
            ("ocean", self.ocean),
            ("land", self.land),
        ):
            if not isinstance(name, str) or not name:
                raise ValueError(f"{role} component name must be a non-empty string")


@dataclass(frozen=True)
class RuntimeOptions:
    """Public runtime policy for a coupled VerCOR run."""

    dtype: DTypePolicy = field(default_factory=DTypePolicy)
    surface_masks: SurfaceMaskPolicy | None = field(default_factory=SurfaceMaskPolicy)
    execution: ExecutionMode | ExecutionBackend = "auto"
    year_in_seconds: float = 365 * 86400.0

    def __post_init__(self) -> None:
        """Validate runtime policy values."""

        if isinstance(self.execution, str):
            if self.execution not in ("auto", "jax", "host"):
                raise ValueError(
                    "execution must be 'auto', 'jax', 'host', or a backend"
                )
            return

        run = getattr(self.execution, "run", None)
        if not callable(run):
            raise TypeError("execution backend must expose run(state, *, context)")


__all__ = [
    "DTypePolicy",
    "ExecutionBackend",
    "ExecutionMode",
    "RuntimeOptions",
    "SurfaceMaskPolicy",
]
