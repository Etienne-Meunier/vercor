from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Literal

from vercor.output.adapters import OutputConfig

OutputFrequency = Literal["day", "month", "year"]


@dataclass(frozen=True)
class Spinup:
    """Spinup policy shared by bundled setup factories."""

    enabled: bool = False
    duration: timedelta = timedelta(days=2)


@dataclass(frozen=True)
class PeriodOutput:
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


@dataclass(frozen=True)
class JaxGCMConfig:
    """Configuration for the bundled JAXGCM/JCM atmosphere setup factory."""

    name: str = "ATM"
    custom_parameters: dict[str, float] | None = None
    model_timestep: timedelta = timedelta(minutes=30)
    save_interval: timedelta = timedelta(days=1)
    forcing_data: Any | None = None
    spinup: Spinup = field(default_factory=Spinup)
    output: OutputConfig = field(default_factory=OutputConfig)
    jitted: bool = True


@dataclass(frozen=True)
class VerosConfig:
    """Configuration for the bundled Veros ocean setup factory."""

    name: str = "OCN"
    custom_parameters: dict[str, Any] | None = None
    restore_to_climatology: bool = False
    spinup: Spinup = field(default_factory=Spinup)
    output: OutputConfig = field(default_factory=OutputConfig)
    jitted: bool = False


@dataclass(frozen=True)
class CAMulatorConfig:
    """Configuration for the bundled CAMulator atmosphere setup factory."""

    config_path: str
    name: str = "ATM"
    model_weights_path: str = "checkpoint.pt00091.pt"
    output_subfolder_name: str | None = None
    init_noise: float | None = None
    spinup: Spinup = field(default_factory=Spinup)
    output: OutputConfig = field(default_factory=OutputConfig)
    device: str = "cuda"
    output_cpus_number: int = 8
    logger: Any | None = None


__all__ = [
    "CAMulatorConfig",
    "JaxGCMConfig",
    "OutputFrequency",
    "PeriodOutput",
    "Spinup",
    "VerosConfig",
]
