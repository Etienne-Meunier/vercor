"""Public output extension primitives for component adapter authors."""

from __future__ import annotations

from vercor.output.adapters import (
    OutputConfig,
    SnapshotContext,
    SnapshotWriter,
)
from vercor.output.variables import OutputVariable

__all__ = [
    "OutputConfig",
    "OutputVariable",
    "SnapshotContext",
    "SnapshotWriter",
]
