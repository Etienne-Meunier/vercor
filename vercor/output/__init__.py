"""Public output extension primitives for component adapter authors."""

from __future__ import annotations

from vercor.output.adapters import (
    ComponentOutputAdapter,
    ComponentSnapshotWriter,
    component_snapshot_writer,
    register_component_snapshot_writer,
)
from vercor.output.variables import OutputVariable

__all__ = [
    "ComponentOutputAdapter",
    "ComponentSnapshotWriter",
    "OutputVariable",
    "component_snapshot_writer",
    "register_component_snapshot_writer",
]
