"""Public runtime extension contracts and runtime-state views."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, TypeAlias, runtime_checkable

from vercor.dtypes import DTypePolicy
from vercor.state import ComponentState, RunState
from vercor.topology import TopologyPolicy

if TYPE_CHECKING:
    from vercor.clock import Clock
    from vercor.jax_logging import LoggerLike
    from vercor.types import RuntimeArray


ExecutionMode: TypeAlias = Literal["auto", "jax", "host"]


@dataclass(frozen=True)
class RuntimeOptions:
    """Public static runtime policy for a coupled VerCOR run."""

    dtype: DTypePolicy = field(default_factory=DTypePolicy)
    execution: ExecutionMode | "ExecutionBackend" = "auto"
    topology: TopologyPolicy | None = None
    model_year_seconds: float = 365 * 86400.0

    def __post_init__(self) -> None:
        """Validate runtime policy values."""

        if not isinstance(self.dtype, DTypePolicy):
            raise TypeError("dtype must be a DTypePolicy")

        if isinstance(self.execution, str):
            if self.execution not in ("auto", "jax", "host"):
                raise ValueError(
                    "execution must be 'auto', 'jax', 'host', or a backend"
                )
        else:
            run = getattr(self.execution, "run", None)
            if not callable(run):
                raise TypeError(
                    "execution backend must expose run(state, *, context, driver)"
                )

        if self.topology is not None:
            applies = getattr(self.topology, "applies", None)
            build = getattr(self.topology, "build", None)
            if not callable(applies) or not callable(build):
                raise TypeError(
                    "topology policy must expose applies(context) and build(context)"
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
]
