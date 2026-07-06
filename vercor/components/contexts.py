from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from vercor.calendar import ModelDateTime
from vercor.jax_logging import LoggerLike
from vercor.settings import Settings


@dataclass(frozen=True)
class SetupContext:
    """Minimal setup context passed to component initialization hooks."""

    start: datetime | ModelDateTime
    dt_seconds: float
    run_order: Sequence[str]
    settings: Settings
    logger: LoggerLike


@dataclass(frozen=True)
class StepContext:
    """Minimal runtime step context passed to component step boundaries."""

    dt_seconds: float
    settings: Settings
    time: datetime | ModelDateTime | None = None
    logger: LoggerLike | None = None
    step: int = 0


__all__ = [
    "SetupContext",
    "StepContext",
]
