"""Public runtime extension contracts and runtime-state views."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, TypeAlias, runtime_checkable

import vercor.clock as _clock
import vercor.dtypes as _dtypes
import vercor.jax_logging as _jax_logging
import vercor.state as _state
import vercor.types as _types
from vercor.topology import TopologyPolicy as _TopologyPolicy

ExecutionMode: TypeAlias = Literal["auto", "jax", "host"]


@dataclass(frozen=True)
class RuntimeOptions:
    """Public static runtime policy for a coupled VerCOR run."""

    dtype: _dtypes.DTypePolicy = field(default_factory=_dtypes.DTypePolicy)
    execution: ExecutionMode | "ExecutionBackend" = "auto"
    topology: _TopologyPolicy | None = None
    model_year_seconds: float = 365 * 86400.0

    def __post_init__(self) -> None:
        """Validate runtime policy values."""

        if not isinstance(self.dtype, _dtypes.DTypePolicy):
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

    clock: _clock.Clock
    run_order: tuple[str, ...]
    options: RuntimeOptions
    logger: _jax_logging.LoggerLike | None = None


@runtime_checkable
class RuntimeDriver(Protocol):
    """Public driver used by custom backends to reuse VerCOR component stepping."""

    def step_component(
        self,
        state: _state.RunState,
        component: str,
        *,
        step: int | _types.RuntimeArray,
    ) -> _state.RunState:
        """Advance one component using VerCOR's exchange/step/send pipeline."""
        ...


@runtime_checkable
class ExecutionBackend(Protocol):
    """Public protocol for custom coupler runtime execution backends."""

    def run(
        self,
        state: _state.RunState,
        *,
        context: ExecutionContext,
        driver: RuntimeDriver,
    ) -> _state.RunState:
        """Run a prepared runtime state and return the final state."""
        ...


__all__ = [
    "ExecutionBackend",
    "ExecutionContext",
    "ExecutionMode",
    "RuntimeDriver",
    "RuntimeOptions",
]
