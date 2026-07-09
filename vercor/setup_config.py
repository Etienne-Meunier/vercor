from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Literal

from vercor.output import OutputConfig as _OutputConfig


@dataclass(frozen=True)
class SurfaceMaskPolicy:
    """Policy for the default atmosphere/ocean/land surface-mask topology."""

    mode: Literal["auto", "required", "disabled"] = "auto"
    atmosphere: str = "ATM"
    ocean: str = "OCN"
    land: str = "LND"

    def __post_init__(self) -> None:
        """Validate surface-mask policy values."""

        if self.mode not in ("auto", "required", "disabled"):
            raise ValueError("mode must be one of 'auto', 'required', 'disabled'")
        for role, name in (
            ("atmosphere", self.atmosphere),
            ("ocean", self.ocean),
            ("land", self.land),
        ):
            if not isinstance(name, str) or not name:
                raise ValueError(f"{role} component name must be a non-empty string")


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
    output: _OutputConfig = field(default_factory=_OutputConfig)
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
    output: _OutputConfig = field(default_factory=_OutputConfig)
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
    output: _OutputConfig = field(default_factory=_OutputConfig)
    device: str = "cuda"
    output_cpus_number: int = 8
    logger: Any | None = None


__all__ = [
    "CAMulatorConfig",
    "JAXGCMConfig",
    "Spinup",
    "SurfaceMaskPolicy",
    "VerosConfig",
]
