from __future__ import annotations

from typing import Any, NamedTuple, cast

from vercor.dtypes import DTypePolicy


class Settings(NamedTuple):
    """Metadata record for one VerCOR setting."""

    value: Any
    description: str
    units: str


DEFAULT_SETTINGS: dict[str, Settings] = {
    # Runtime settings
    "enable_x64": Settings(False, "Enable 64-bit precision for JAX computations", "-"),
    "identifier": Settings("UNNAMED", "Identifier of the current simulation", "-"),
    "missval": Settings(0.0, "Missing value for fields", "-"),
    "apply_time_interpolation": Settings(
        False,
        "Apply monthly time interpolation to exported forcing data",
        "-",
    ),
    "get_field_time_slice": Settings(
        False,
        "Export only the relevant daily time slice from forcing data",
        "-",
    ),
    "year_in_seconds": Settings(365 * 86400.0, "Nominal model year length", "s"),
    # Physical constants
    "earth_radius": Settings(6.371e6, "Earth radius", "m"),
    # Bulk formula constants
    "gravity": Settings(9.81, "Acceleration due to gravity", "m/s^2"),
    "rhoAir": Settings(1.3, "Density of air", "kg/m^3"),
    "rdair": Settings(287.042, "Dry air gas constant", "J/(K*kg)"),
    "cpdair": Settings(
        1.00464e3,
        "Specific heat capacity of dry air",
        "J/(kg*K)",
    ),
    "zvir": Settings(
        0.608,
        "Dry-air water-vapor molecular mass ratio correction",
        "-",
    ),
    "p0": Settings(1e5, "Reference pressure for potential temperature", "Pa"),
    "mwdair": Settings(28.966, "Molecular weight of dry air", "kg/kmole"),
    "cpwv": Settings(1.810e3, "Specific heat of water vapor", "J/(kg*K)"),
    "cpvir": Settings(0.802, "Specific heat of vaporization ratio correction", "-"),
    "cappa": Settings(0.286, "Dry air gas constant over heat capacity", "-"),
    "latice": Settings(3.337e5, "Latent heat of fusion", "J/kg"),
    "rgas": Settings(8314.47, "Ideal gas constant", "J/(K*kmole)"),
    "umin_ocean": Settings(
        0.5,
        "Minimum atmospheric wind speed over ocean surface",
        "m/s",
    ),
    "umin_ice": Settings(
        1.0,
        "Minimum atmospheric wind speed over ice surface",
        "m/s",
    ),
    "karman": Settings(0.4, "von Karman constant", "-"),
    "stefBoltz": Settings(5.67e-8, "Stefan-Boltzmann constant", "W/(m^2*K^4)"),
    "ocean_emissivity": Settings(0.97, "Long-wave emissivity of ocean surface", "-"),
    "ice_emissivity": Settings(0.97, "Long-wave emissivity of sea ice", "-"),
    "snow_emissivity": Settings(0.99, "Long-wave emissivity of snow", "-"),
    "latvap": Settings(2.501e6, "Latent heat of vaporization", "J/kg"),
    "latfresh": Settings(3.34e5, "Latent heat of fusion", "J/kg"),
    "gamma_blk": Settings(0.1, "Bulk aerodynamic resistance", "-"),
    "zref": Settings(10.0, "Reference height", "m"),
    "ztref": Settings(2.0, "Reference height for air temperature", "m"),
}


def _copy_settings(settings: dict[str, Settings]) -> dict[str, Settings]:
    """Return independent settings records for a new settings container."""

    return {
        name: Settings(record.value, record.description, record.units)
        for name, record in settings.items()
    }


class VercorSettings:
    """Mutable metadata-backed settings container for couplers and components."""

    _settings: dict[str, Settings]

    def __init__(self, **kwargs: Any) -> None:
        """Create settings from VerCOR defaults plus optional overrides."""

        object.__setattr__(self, "_settings", _copy_settings(DEFAULT_SETTINGS))
        for name, value in kwargs.items():
            if isinstance(value, Settings):
                self._settings[name] = Settings(
                    value.value,
                    value.description,
                    value.units,
                )
            elif name in self._settings:
                self.set_value(name, value)
            else:
                self.add_setting(name, value)

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
                "use add_setting() to add custom settings"
            )
        self.set_value(name, value)

    def __contains__(self, name: object) -> bool:
        """Return whether ``name`` is a configured setting."""

        return name in self._settings

    def __repr__(self) -> str:
        values = ", ".join(
            f"{name}={record.value!r}" for name, record in self._settings.items()
        )
        return f"{self.__class__.__name__}({values})"

    def _bool_value(self, name: str) -> bool:
        return cast(bool, self.get_value(name))

    def _float_value(self, name: str) -> float:
        return cast(float, self.get_value(name))

    def _str_value(self, name: str) -> str:
        return cast(str, self.get_value(name))

    @property
    def enable_x64(self) -> bool:
        return self._bool_value("enable_x64")

    @enable_x64.setter
    def enable_x64(self, value: Any) -> None:
        self.set_value("enable_x64", value)

    @property
    def identifier(self) -> str:
        return self._str_value("identifier")

    @property
    def missval(self) -> float:
        return self._float_value("missval")

    @property
    def apply_time_interpolation(self) -> bool:
        return self._bool_value("apply_time_interpolation")

    @apply_time_interpolation.setter
    def apply_time_interpolation(self, value: Any) -> None:
        self.set_value("apply_time_interpolation", value)

    @property
    def get_field_time_slice(self) -> bool:
        return self._bool_value("get_field_time_slice")

    @get_field_time_slice.setter
    def get_field_time_slice(self, value: Any) -> None:
        self.set_value("get_field_time_slice", value)

    @property
    def year_in_seconds(self) -> float:
        return self._float_value("year_in_seconds")

    @year_in_seconds.setter
    def year_in_seconds(self, value: Any) -> None:
        self.set_value("year_in_seconds", value)

    @property
    def earth_radius(self) -> float:
        return self._float_value("earth_radius")

    @property
    def gravity(self) -> float:
        return self._float_value("gravity")

    @property
    def rhoAir(self) -> float:
        return self._float_value("rhoAir")

    @property
    def rdair(self) -> float:
        return self._float_value("rdair")

    @property
    def cpdair(self) -> float:
        return self._float_value("cpdair")

    @property
    def zvir(self) -> float:
        return self._float_value("zvir")

    @property
    def p0(self) -> float:
        return self._float_value("p0")

    @property
    def mwdair(self) -> float:
        return self._float_value("mwdair")

    @property
    def cpwv(self) -> float:
        return self._float_value("cpwv")

    @property
    def cpvir(self) -> float:
        return self._float_value("cpvir")

    @property
    def cappa(self) -> float:
        return self._float_value("cappa")

    @property
    def latice(self) -> float:
        return self._float_value("latice")

    @property
    def rgas(self) -> float:
        return self._float_value("rgas")

    @property
    def umin_ocean(self) -> float:
        return self._float_value("umin_ocean")

    @property
    def umin_ice(self) -> float:
        return self._float_value("umin_ice")

    @property
    def karman(self) -> float:
        return self._float_value("karman")

    @property
    def stefBoltz(self) -> float:
        return self._float_value("stefBoltz")

    @property
    def ocean_emissivity(self) -> float:
        return self._float_value("ocean_emissivity")

    @property
    def ice_emissivity(self) -> float:
        return self._float_value("ice_emissivity")

    @property
    def snow_emissivity(self) -> float:
        return self._float_value("snow_emissivity")

    @property
    def latvap(self) -> float:
        return self._float_value("latvap")

    @property
    def latfresh(self) -> float:
        return self._float_value("latfresh")

    @property
    def gamma_blk(self) -> float:
        return self._float_value("gamma_blk")

    @property
    def zref(self) -> float:
        return self._float_value("zref")

    @property
    def ztref(self) -> float:
        return self._float_value("ztref")

    def add_setting(
        self,
        name: str,
        value: Any,
        description: str = "-",
        units: str = "-",
    ) -> None:
        """Add a custom setting to this container."""

        if name in self._settings:
            raise KeyError(f"Setting {name!r} already exists")
        if isinstance(value, Settings):
            self._settings[name] = Settings(value.value, value.description, value.units)
            return
        self._settings[name] = Settings(value, description, units)

    def set_value(self, name: str, value: Any) -> None:
        """Update the value of an existing setting while preserving metadata."""

        if name not in self._settings:
            raise KeyError(f"Setting {name!r} does not exist")
        metadata = self._settings[name]
        if isinstance(value, Settings):
            self._settings[name] = Settings(value.value, value.description, value.units)
            return
        self._settings[name] = Settings(value, metadata.description, metadata.units)

    def get_value(self, name: str) -> Any:
        """Return a setting value by name."""

        if name not in self._settings:
            raise KeyError(f"Setting {name!r} does not exist")
        return self._settings[name].value

    def get_metadata(self, name: str) -> Settings:
        """Return the full metadata record for one setting."""

        if name not in self._settings:
            raise KeyError(f"Setting {name!r} does not exist")
        record = self._settings[name]
        return Settings(record.value, record.description, record.units)

    def as_values(self) -> dict[str, Any]:
        """Return a plain mapping of setting names to values."""

        return {name: record.value for name, record in self._settings.items()}

    @property
    def dtype_policy(self) -> DTypePolicy:
        """Return the canonical array dtype policy for these settings."""

        return DTypePolicy.from_settings(self)


ComponentSettings = VercorSettings
