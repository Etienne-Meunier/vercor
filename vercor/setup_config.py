from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Literal

OutputFrequency = Literal["day", "month", "year"]


@dataclass(frozen=True)
class SpinupConfig:
    """Spinup policy shared by bundled setup factories."""

    enabled: bool = False
    duration: timedelta = timedelta(days=2)


@dataclass(frozen=True)
class PeriodOutputConfig:
    """Period-output policy shared by bundled setup factories."""

    frequency: OutputFrequency | None = None
    variables: Sequence[str] = ()

    def __post_init__(self) -> None:
        if self.frequency not in (None, "day", "month", "year"):
            raise ValueError("frequency must be one of None, 'day', 'month', 'year'")
        if isinstance(self.variables, str):
            raise ValueError("variables must be a sequence of names, not a string")
        normalized = tuple(self.variables)
        if not all(isinstance(variable, str) for variable in normalized):
            raise ValueError("variables entries must be strings")
        object.__setattr__(self, "variables", normalized)


__all__ = [
    "OutputFrequency",
    "PeriodOutputConfig",
    "SpinupConfig",
]
