"""Public runtime extension contracts and runtime-state views."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, TypeAlias, runtime_checkable

from vercor.dtypes import DTypePolicy
from vercor.state import ComponentState, RunState

if TYPE_CHECKING:
    from vercor.clock import Clock
    from vercor.jax_logging import LoggerLike
    from vercor.types import RuntimeArray


ExecutionMode: TypeAlias = Literal["auto", "jax", "host"]


@dataclass(frozen=True)
class SurfaceMaskPolicy:
    """Policy for the bundled atmosphere/ocean/land surface-mask topology."""

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
    """Public static runtime policy for a coupled VerCOR run."""

    dtype: DTypePolicy = field(default_factory=DTypePolicy)
    execution: ExecutionMode | "ExecutionBackend" = "auto"
    surface_masks: SurfaceMaskPolicy | None = None
    model_year_seconds: float = 365 * 86400.0

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
            raise TypeError(
                "execution backend must expose run(state, *, context, driver)"
            )


@dataclass(frozen=True)
class ExecutionContext:
    """Stable public context supplied to custom execution backends."""

    clock: "Clock"
    run_order: tuple[str, ...]
    options: RuntimeOptions
    logger: "LoggerLike | None" = None


@runtime_checkable
class RuntimeDriver(Protocol):
    """Public driver used by custom backends to reuse VerCOR component stepping."""

    def step_component(
        self,
        state: RunState,
        component: str,
        *,
        step: int | "RuntimeArray",
    ) -> RunState:
        """Advance one component using VerCOR's exchange/step/send pipeline."""
        ...


@runtime_checkable
class ExecutionBackend(Protocol):
    """Public protocol for custom coupler runtime execution backends."""

    def run(
        self,
        state: RunState,
        *,
        context: ExecutionContext,
        driver: RuntimeDriver,
    ) -> RunState:
        """Run a prepared runtime state and return the final state."""
        ...


__all__ = [
    "ComponentState",
    "DTypePolicy",
    "ExecutionBackend",
    "ExecutionContext",
    "ExecutionMode",
    "RunState",
    "RuntimeDriver",
    "RuntimeOptions",
    "SurfaceMaskPolicy",
]
