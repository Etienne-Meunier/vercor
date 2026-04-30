from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from logging import Logger

from vercor.clock import ModelDateTime
from vercor.run_sequence import RunSequence
from vercor.settings import VercorSettings


@dataclass(frozen=True)
class ComponentInitContext:
    """Minimal component initialization context owned by the coupler."""

    start: datetime | ModelDateTime
    dt_seconds: float
    run_sequence: RunSequence
    settings: VercorSettings
    logger: Logger


@dataclass(frozen=True)
class RuntimeStepContext:
    """Minimal runtime step context passed to component step boundaries."""

    dt_seconds: float
    settings: VercorSettings
    time: datetime | ModelDateTime | None = None
    logger: Logger | None = None
