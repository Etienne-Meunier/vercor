"""Public output extension primitives for component adapter authors."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, TypeAlias

from vercor.calendar import ModelDateTime as _ModelDateTime
from vercor.components.contracts import Component as _Component
from vercor.jax_logging import LoggerLike as _LoggerLike
from vercor.state import ComponentState as _ComponentState

OutputFrequency: TypeAlias = Literal["step", "day", "month", "year"]


@dataclass(frozen=True)
class OutputVariable:
    """Array values with NetCDF dimension names and variable attributes."""

    dims: tuple[str, ...]
    values: Any
    attrs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PeriodOutput:
    """Period-output policy shared by bundled setup factories."""

    frequency: OutputFrequency = "step"
    variables: Sequence[str] = ()

    def __post_init__(self) -> None:
        """Validate and freeze the period-output configuration."""

        if self.frequency not in ("step", "day", "month", "year"):
            raise ValueError("frequency must be one of 'step', 'day', 'month', 'year'")
        if isinstance(self.variables, str):
            raise ValueError("variables must be a sequence of names, not a string")
        normalized = tuple(self.variables)
        if not all(isinstance(variable, str) for variable in normalized):
            raise ValueError("variables entries must be strings")
        object.__setattr__(self, "variables", normalized)


@dataclass(frozen=True)
class SnapshotContext:
    """Public payload passed to component snapshot writers."""

    component: _Component
    state: _ComponentState
    payload: Any | None
    output_path: Path
    time: datetime | _ModelDateTime
    logger: _LoggerLike | None


SnapshotWriter: TypeAlias = Callable[[SnapshotContext], None]


@dataclass(frozen=True)
class OutputConfig:
    """Public output extension specification for a component."""

    snapshot_writer: SnapshotWriter | None = None
    period: PeriodOutput | None = None


__all__ = [
    "OutputConfig",
    "OutputFrequency",
    "OutputVariable",
    "PeriodOutput",
    "SnapshotContext",
    "SnapshotWriter",
]
