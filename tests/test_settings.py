from __future__ import annotations

from datetime import datetime

import pytest

from tests._coverage_support import make_test_grid
from vercor.clock import Clock
from vercor.components import FieldImportPolicy
from vercor.components.data import DataComponent
from vercor.coupler import Coupler
from vercor.dtypes import DTypePolicy, SupportsEnableX64
from vercor.settings import (
    DEFAULT_SETTINGS,
    SettingSpec,
    Settings,
)


def test_default_settings_are_metadata_records() -> None:
    assert Settings is Settings
    assert isinstance(DEFAULT_SETTINGS["enable_x64"], SettingSpec)
    assert DEFAULT_SETTINGS["enable_x64"].value is False
    assert DEFAULT_SETTINGS["enable_x64"].description
    assert DEFAULT_SETTINGS["enable_x64"].units == "-"
    assert DEFAULT_SETTINGS["gravity"].units == "m/s^2"
    assert "apply_time_interpolation" not in DEFAULT_SETTINGS
    assert "apply_daily_time_selection" not in DEFAULT_SETTINGS


def test_constructor_overrides_known_setting_without_losing_metadata() -> None:
    settings = Settings(enable_x64=True)

    assert settings.enable_x64 is True
    metadata = settings.get_metadata("enable_x64")
    assert metadata.value is True
    assert metadata.description == DEFAULT_SETTINGS["enable_x64"].description
    assert metadata.units == DEFAULT_SETTINGS["enable_x64"].units


def test_constructor_rejects_unknown_kwargs_unless_marked_custom() -> None:
    with pytest.raises(TypeError, match="Unknown setting"):
        Settings(custom_parameter=3.0)

    settings = Settings(custom={"custom_parameter": 3.0})

    assert settings.custom_parameter == 3.0
    assert settings.get_metadata("custom_parameter") == SettingSpec(3.0, "-", "-")


def test_constructor_preserves_explicit_settings_metadata() -> None:
    metadata = SettingSpec(600.0, "Tracer timestep", "s")
    settings = Settings(custom={"dt_tracer": metadata})

    assert settings.dt_tracer == 600.0
    assert settings.get_metadata("dt_tracer") == metadata


def test_add_rejects_duplicates() -> None:
    settings = Settings()

    with pytest.raises(KeyError, match="enable_x64"):
        settings.add("enable_x64", True)


def test_set_rejects_unknown_settings() -> None:
    settings = Settings()

    with pytest.raises(KeyError, match="missing"):
        settings.set("missing", 1)


def test_default_settings_are_copied_per_instance() -> None:
    left = Settings()
    right = Settings()

    left.set("enable_x64", True)
    left.add("local_only", 1)

    assert right.enable_x64 is False
    assert "local_only" not in right.as_dict()


def test_settings_canonical_methods_update_values_and_metadata() -> None:
    settings = Settings()

    settings.add("local", 1.0, description="Local value", units="1")
    settings.set("local", 2.0)

    assert settings.get("local") == 2.0
    assert settings.as_dict()["local"] == 2.0
    assert settings.get_metadata("local") == SettingSpec(2.0, "Local value", "1")


def test_settings_container_uses_direct_default_mapping_copy() -> None:
    import vercor.settings as settings_module

    assert not hasattr(settings_module, "_copy_settings")


def test_attribute_access_and_assignment_update_setting_values() -> None:
    settings = Settings()

    settings.enable_x64 = True
    settings.cappa = 0.287

    assert settings.get("enable_x64") is True
    assert "year_in_seconds" not in settings.as_dict()
    assert settings.get_metadata("cappa") == SettingSpec(
        0.287,
        DEFAULT_SETTINGS["cappa"].description,
        DEFAULT_SETTINGS["cappa"].units,
    )
    with pytest.raises(AttributeError, match="new_parameter"):
        settings.new_parameter = 1


def test_known_setting_attributes_are_typed_annotations_not_descriptors() -> None:
    annotations = Settings.__annotations__

    for name in DEFAULT_SETTINGS:
        assert name in annotations
        assert name not in vars(Settings)
    assert isinstance(Settings.dtype_policy, property)


def test_dir_lists_default_and_custom_settings() -> None:
    settings = Settings(custom={"custom_parameter": 3.0})

    names = dir(settings)

    assert "enable_x64" in names
    assert "cappa" in names
    assert "custom_parameter" in names
    assert "dtype_policy" in names


def _precision_protocol_value(settings: SupportsEnableX64) -> bool:
    return DTypePolicy.from_settings(settings).enable_x64


def test_settings_satisfy_precision_protocol_after_dynamic_refactor() -> None:
    settings = Settings(enable_x64=True)

    assert _precision_protocol_value(settings) is True


def test_coupler_and_components_get_independent_settings_containers() -> None:
    class StaticDataComponent(DataComponent):
        pass

    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=3600.0, steps=1)
    )
    atmosphere = StaticDataComponent.from_fields(
        name="ATM",
        grid=make_test_grid(name="atm"),
        import_policy=FieldImportPolicy(time_interpolation=True),
    )
    ocean = StaticDataComponent.from_fields(
        name="OCN",
        grid=make_test_grid(name="ocn"),
        import_policy=FieldImportPolicy(daily_selection=True),
    )

    coupler.settings.enable_x64 = True

    assert coupler.settings is not atmosphere.settings
    assert atmosphere.settings is not ocean.settings
    assert coupler.settings.enable_x64 is True
    assert atmosphere.settings.enable_x64 is False
    assert atmosphere.import_policy.time_interpolation is True
    assert ocean.import_policy.time_interpolation is False
    assert ocean.import_policy.daily_selection is True
