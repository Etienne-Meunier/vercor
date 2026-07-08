"""Public output extension specifications."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

from vercor.calendar import ModelDateTime
from vercor.jax_logging import LoggerLike

if TYPE_CHECKING:
    from vercor.components.base import Component
    from vercor.setup_config import PeriodOutput
    from vercor.state import ComponentState


@dataclass(frozen=True)
class SnapshotContext:
    """Public payload passed to component snapshot writers."""

    component: "Component"
    state: "ComponentState"
    payload: Any | None
    output_path: Path
    time: datetime | ModelDateTime
    logger: LoggerLike | None


SnapshotWriter: TypeAlias = Callable[[SnapshotContext], None]


@dataclass(frozen=True)
class OutputConfig:
    """Public output extension specification for a component."""

    snapshot_writer: SnapshotWriter | None = None
    period: "PeriodOutput | None" = None


__all__ = [
    "OutputConfig",
    "SnapshotContext",
    "SnapshotWriter",
]
