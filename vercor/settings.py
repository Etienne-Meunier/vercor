from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

from vercor.dtypes import DTypePolicy
from vercor.physical_constants import PHYSICAL_CONSTANT_SETTINGS


class SettingSpec(NamedTuple):
    """Metadata record for one VerCOR setting."""

    value: Any
    description: str
    units: str


CONTROL_SETTINGS: dict[str, SettingSpec] = {
    # Runtime settings
    "enable_x64": SettingSpec(
        False,
        "Enable 64-bit precision for JAX computations",
        "-",
    ),
    "identifier": SettingSpec("UNNAMED", "Identifier of the current simulation", "-"),
    "missval": SettingSpec(0.0, "Missing value for fields", "-"),
    "year_in_seconds": SettingSpec(
        365 * 86400.0,
        "Nominal model year length",
        "s",
    ),
}


DEFAULT_SETTINGS: dict[str, SettingSpec] = {
    **CONTROL_SETTINGS,
    **{
        name: SettingSpec(value, description, units)
        for name, (value, description, units) in PHYSICAL_CONSTANT_SETTINGS.items()
    },
}


class Settings:
    """Mutable metadata-backed settings container for couplers and components.

    Known default settings are class-level annotations for static type checkers;
    runtime values live in ``_settings`` and are resolved dynamically.
    """

    _settings: dict[str, SettingSpec]
    enable_x64: bool
    identifier: str
    missval: float
    year_in_seconds: float
    earth_radius: float
    gravity: float
    rhoAir: float
    rdair: float
    cpdair: float
    zvir: float
    p0: float
    mwdair: float
    cpwv: float
    cpvir: float
    cappa: float
    latice: float
    rgas: float
    umin_ocean: float
    umin_ice: float
    karman: float
    stefBoltz: float
    ocean_emissivity: float
    ice_emissivity: float
    snow_emissivity: float
    latvap: float
    latfresh: float
    gamma_blk: float
    zref: float
    ztref: float

    def __init__(
        self,
        *,
        custom: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Create settings from VerCOR defaults plus optional overrides."""

        object.__setattr__(self, "_settings", dict(DEFAULT_SETTINGS))
        for name, value in kwargs.items():
            if isinstance(value, SettingSpec):
                self._settings[name] = SettingSpec(
                    value.value,
                    value.description,
                    value.units,
                )
            elif name in self._settings:
                self.set(name, value)
            else:
                raise TypeError(
                    f"Unknown setting {name!r}; pass custom settings with "
                    "Settings(custom={...}) or add them with Settings.add()."
                )
        for name, value in (custom or {}).items():
            self.add(name, value)

    def __getattr__(self, name: str) -> Any:
        """Return the value of a setting by attribute name."""

        settings = object.__getattribute__(self, "_settings")
        if name in settings:
            return settings[name].value
        raise AttributeError(
            f"{self.__class__.__name__!s} has no setting named {name!r}"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        """Update an existing setting value through attribute assignment."""

        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return

        settings = object.__getattribute__(self, "_settings")
        if name not in settings:
            raise AttributeError(
                f"{self.__class__.__name__!s} has no setting named {name!r}; "
                "use add() to add custom settings"
            )
        self.set(name, value)

    def __contains__(self, name: object) -> bool:
        """Return whether ``name`` is a configured setting."""

        return name in self._settings

    def __dir__(self) -> list[str]:
        """Return normal instance attributes plus configured setting names."""

        names = set(super().__dir__())
        names.update(self._settings)
        return sorted(names)

    def __repr__(self) -> str:
        values = ", ".join(
            f"{name}={record.value!r}" for name, record in self._settings.items()
        )
        return f"{self.__class__.__name__}({values})"

    def add(
        self,
        name: str,
        value: Any,
        *,
        description: str = "-",
        units: str = "-",
    ) -> None:
        """Add a custom setting to this container."""

        if name in self._settings:
            raise KeyError(f"Setting {name!r} already exists")
        if isinstance(value, SettingSpec):
            self._settings[name] = SettingSpec(
                value.value,
                value.description,
                value.units,
            )
            return
        self._settings[name] = SettingSpec(value, description, units)

    def set(self, name: str, value: Any) -> None:
        """Update the value of an existing setting while preserving metadata."""

        if name not in self._settings:
            raise KeyError(f"Setting {name!r} does not exist")
        metadata = self._settings[name]
        if isinstance(value, SettingSpec):
            self._settings[name] = SettingSpec(
                value.value,
                value.description,
                value.units,
            )
            return
        self._settings[name] = SettingSpec(
            value,
            metadata.description,
            metadata.units,
        )

    def get(self, name: str) -> Any:
        """Return a setting value by name."""

        if name not in self._settings:
            raise KeyError(f"Setting {name!r} does not exist")
        return self._settings[name].value

    def get_metadata(self, name: str) -> SettingSpec:
        """Return the full metadata record for one setting."""

        if name not in self._settings:
            raise KeyError(f"Setting {name!r} does not exist")
        record = self._settings[name]
        return SettingSpec(record.value, record.description, record.units)

    def as_dict(self) -> dict[str, Any]:
        """Return a plain mapping of setting names to values."""

        return {name: record.value for name, record in self._settings.items()}

    @property
    def dtype_policy(self) -> DTypePolicy:
        """Return the canonical array dtype policy for these settings."""

        return DTypePolicy.from_settings(self)


__all__ = [
    "CONTROL_SETTINGS",
    "DEFAULT_SETTINGS",
    "SettingSpec",
    "Settings",
]
