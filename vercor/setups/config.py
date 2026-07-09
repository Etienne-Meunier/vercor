from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from vercor.output import OutputConfig, PeriodOutput


@dataclass(frozen=True)
class Spinup:
    """Spinup policy shared by bundled setup factories."""

    enabled: bool = False
    duration: timedelta = timedelta(days=2)


@dataclass(frozen=True)
class JAXGCMConfig:
    """Configuration for the bundled JAXGCM/JCM atmosphere setup factory."""

    name: str = "ATM"
    custom_parameters: Mapping[str, float] | None = None
    model_timestep: timedelta = timedelta(minutes=30)
    save_interval: timedelta = timedelta(days=1)
    forcing_data: Any | None = None
    spinup: Spinup = field(default_factory=Spinup)
    output: OutputConfig = field(default_factory=OutputConfig)
    jitted: bool = True

    def __post_init__(self) -> None:
        """Copy mutable caller-provided mappings into owned config state."""

        if self.custom_parameters is not None:
            object.__setattr__(
                self,
                "custom_parameters",
                dict(self.custom_parameters),
            )


@dataclass(frozen=True)
class VerosConfig:
    """Configuration for the bundled Veros ocean setup factory."""

    name: str = "OCN"
    custom_parameters: Mapping[str, Any] | None = None
    restore_to_climatology: bool = False
    spinup: Spinup = field(default_factory=Spinup)
    output: OutputConfig = field(default_factory=OutputConfig)
    jitted: bool = False

    def __post_init__(self) -> None:
        """Copy mutable caller-provided mappings into owned config state."""

        if self.custom_parameters is not None:
            object.__setattr__(
                self,
                "custom_parameters",
                dict(self.custom_parameters),
            )


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


def _default_jcm_atmosphere_config() -> JAXGCMConfig:
    """Return the historic defaults for the paired JCM setup factory."""

    return JAXGCMConfig(
        spinup=Spinup(enabled=True),
        output=OutputConfig(period=PeriodOutput(frequency="month")),
        jitted=True,
    )


@dataclass(frozen=True)
class JCMLandAtmosphereConfig:
    """Configuration for the bundled paired JCM land/atmosphere setup."""

    atmosphere: JAXGCMConfig = field(default_factory=_default_jcm_atmosphere_config)
    land_name: str = "LND"

    def __post_init__(self) -> None:
        """Validate paired setup component names."""

        if not isinstance(self.land_name, str) or not self.land_name:
            raise ValueError("land_name must be a non-empty string")


__all__ = [
    "CAMulatorConfig",
    "JAXGCMConfig",
    "JCMLandAtmosphereConfig",
    "Spinup",
    "VerosConfig",
]
