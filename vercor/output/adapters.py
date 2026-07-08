"""Public output extension specifications."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

from vercor.calendar import ModelDateTime
from vercor.jax_logging import LoggerLike

if TYPE_CHECKING:
    from vercor.runtime.state import RuntimeComponentState


ComponentSnapshotWriter: TypeAlias = Callable[
    ["RuntimeComponentState", Path, datetime | ModelDateTime, LoggerLike | None],
    None,
]


@dataclass(frozen=True)
class ComponentOutput:
    """Public output extension specification for a component."""

    snapshot_writer: ComponentSnapshotWriter | None = None


__all__ = [
    "ComponentOutput",
    "ComponentSnapshotWriter",
]
