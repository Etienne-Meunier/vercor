from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

from vercor.calendar import ModelDateTime
from vercor.dtypes import DTypePolicy
from vercor.jax_logging import LoggerLike
from vercor.physics import PhysicalConstants
from vercor.types import RuntimeArray


@dataclass(frozen=True)
class SetupContext:
    """Minimal setup context passed to component initialization hooks."""

    start: datetime | ModelDateTime
    dt_seconds: float
    run_order: Sequence[str]
    logger: LoggerLike
    constants: PhysicalConstants = field(default_factory=PhysicalConstants)
    dtype: DTypePolicy = field(default_factory=DTypePolicy.from_jax_config)


@dataclass(frozen=True)
class StepContext:
    """Minimal runtime step context passed to component step boundaries."""

    dt_seconds: float
    constants: PhysicalConstants = field(default_factory=PhysicalConstants)
    dtype: DTypePolicy = field(default_factory=DTypePolicy.from_jax_config)
    time: datetime | ModelDateTime | None = None
    logger: LoggerLike | None = None
    step: int | RuntimeArray = 0


__all__ = [
    "SetupContext",
    "StepContext",
]
