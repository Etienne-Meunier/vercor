"""Public output extension primitives for component adapter authors."""

from __future__ import annotations

from vercor.output.adapters import (
    OutputSpec,
    SnapshotContext,
    SnapshotWriter,
)
from vercor.output.variables import OutputVariable

__all__ = [
    "OutputSpec",
    "OutputVariable",
    "SnapshotContext",
    "SnapshotWriter",
]
