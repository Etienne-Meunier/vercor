"""Bundled slab and data factories declare generic step-period output."""

from __future__ import annotations

from datetime import datetime
from inspect import Parameter, signature
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import h5netcdf
import jax.numpy as jnp
import numpy as np
import pytest

from tests._coverage_support import make_test_grid
from vercor.clock import Clock
from vercor.components import CallableComponent, ComponentSpec
from vercor.coupler import Coupler
from vercor.output import OutputSpec, OutputTarget, PeriodOutput
from vercor.setups import (
    make_era5_atmosphere,
    make_era5_land,
    make_era5_ocean,
    make_erainterim_ocean,
    make_slab_atmosphere,
    make_slab_land,
    make_slab_ocean,
    make_slab_seaice,
)
from vercor.setups._data import jcm_land as jcm_land_module
from vercor.setups._data._component_helpers import (
    time_interpolated_data_component,
)
from vercor.setups._output import bundled_output

_SLAB_FACTORIES = (
    make_slab_atmosphere,
    make_slab_land,
    make_slab_ocean,
    make_slab_seaice,
)
_DATA_FACTORIES = (
    make_era5_atmosphere,
    make_era5_land,
    make_era5_ocean,
    make_erainterim_ocean,
)


def _assert_step_period_output(component: Any) -> None:
    output = component.spec.output
    assert isinstance(output, OutputSpec)
    assert output.provider is None
    assert output.period == PeriodOutput(frequency="step")


@pytest.mark.parametrize(
    "factory",
    _SLAB_FACTORIES,
)
def test_all_bundled_slab_factories_declare_step_period_output(factory: Any) -> None:
    _assert_step_period_output(factory(make_test_grid(name="slab-output")))


@pytest.mark.parametrize("factory", _SLAB_FACTORIES)
def test_slab_factory_accepts_keyword_only_output_spec(factory: Any) -> None:
    output_parameter = signature(factory).parameters["output"]
    assert output_parameter.kind is Parameter.KEYWORD_ONLY
    assert output_parameter.default is None
    custom_output = OutputSpec(period=PeriodOutput(frequency="month"))

    component = factory(
        make_test_grid(name="configured-slab"),
        output=custom_output,
    )

    assert component.spec.output is custom_output


def test_bundled_output_rejects_invalid_override() -> None:
    with pytest.raises(TypeError, match="output must be OutputSpec or None"):
        bundled_output(cast(Any, "month"))


def test_shared_data_factory_declares_step_period_output() -> None:
    grid = make_test_grid(name="data-output")
    component = time_interpolated_data_component(
        name="DATA",
        grid=grid,
        fields={"temperature": jnp.full(grid.shape, 280.0)},
        inputs=("forcing",),
        outputs=("temperature",),
        initial_fields={"forcing": 1.0},
    )

    _assert_step_period_output(component)


def test_shared_data_factory_accepts_output_spec() -> None:
    grid = make_test_grid(name="configured-data-output")
    custom_output = OutputSpec(period=PeriodOutput(frequency="month"))

    component = time_interpolated_data_component(
        name="DATA",
        grid=grid,
        fields={"temperature": jnp.full(grid.shape, 280.0)},
        outputs=("temperature",),
        output=custom_output,
    )

    assert component.spec.output is custom_output


@pytest.mark.parametrize("factory", _DATA_FACTORIES)
def test_public_data_factory_accepts_keyword_only_output_spec(factory: Any) -> None:
    output_parameter = signature(factory).parameters["output"]
    assert output_parameter.kind is Parameter.KEYWORD_ONLY
    assert output_parameter.default is None
    custom_output = OutputSpec(period=PeriodOutput(frequency="month"))

    component = factory(output=custom_output)

    assert component.spec.output is custom_output


def _make_test_jcm_land(
    monkeypatch: pytest.MonkeyPatch,
    output: OutputSpec | None = None,
) -> Any:
    grid = make_test_grid(name="jcm-land-output")
    monkeypatch.setattr(
        jcm_land_module,
        "create_lnd_mask_from_ocn",
        lambda **kwargs: (
            jnp.ones(grid.shape),
            jnp.zeros(grid.shape),
        ),
    )
    coords = SimpleNamespace(
        horizontal=SimpleNamespace(
            longitudes=jnp.deg2rad(jnp.asarray([0.0, 180.0])),
            latitudes=jnp.deg2rad(jnp.asarray([-45.0, 45.0])),
        )
    )
    forcing = SimpleNamespace(
        stl_am=jnp.full(grid.shape, 280.0),
        soilw_am=jnp.full(grid.shape, 0.25),
    )

    if output is None:
        return jcm_land_module.make_jcm_land(
            cast(Any, coords),
            cast(Any, forcing),
            grid,
        )
    return jcm_land_module.make_jcm_land(
        cast(Any, coords),
        cast(Any, forcing),
        grid,
        output=output,
    )


def test_direct_jcm_land_data_factory_declares_step_period_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    component = _make_test_jcm_land(monkeypatch)
    _assert_step_period_output(component)


def test_direct_jcm_land_data_factory_accepts_output_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    custom_output = OutputSpec(period=PeriodOutput(frequency="month"))

    component = _make_test_jcm_land(monkeypatch, custom_output)

    assert component.spec.output is custom_output


def test_slab_period_file_contains_declared_outputs_only(tmp_path: Path) -> None:
    grid = make_test_grid(name="slab-period")
    component = make_slab_atmosphere(grid, name="ATM")
    coupler = Coupler(
        Clock(datetime(2000, 1, 1), dt_seconds=3600.0, steps=1),
        components=(component,),
        run_order=(component.name,),
        log_level="WARNING",
    )

    coupler.run(
        output=OutputTarget(
            tmp_path,
            write_final_fields=False,
            write_snapshots=False,
        )
    )

    with h5netcdf.File(tmp_path / "atm.averages.2000-01-01.nc", "r") as dataset:
        field_variables = set(dataset.variables) - {"time", "latitude", "longitude"}
        assert field_variables == set(component.spec.outputs)


def test_slab_output_spec_can_disable_period_files(tmp_path: Path) -> None:
    grid = make_test_grid(name="disabled-slab-period")
    component = make_slab_atmosphere(grid, output=OutputSpec())
    coupler = Coupler(
        Clock(datetime(2000, 1, 1), dt_seconds=3600.0, steps=1),
        components=(component,),
        run_order=(component.name,),
        log_level="WARNING",
    )

    coupler.run(
        output=OutputTarget(
            tmp_path,
            write_final_fields=False,
            write_snapshots=False,
        )
    )

    assert not tuple(tmp_path.glob("*.averages.*.nc"))


def test_slab_month_output_averages_coupler_step_samples(tmp_path: Path) -> None:
    grid = make_test_grid(name="monthly-slab-period")
    start = datetime(2000, 1, 30)
    dt_seconds = 86_400.0

    def run_without_output(steps: int) -> np.ndarray:
        component = make_slab_atmosphere(grid, output=OutputSpec())
        coupler = Coupler(
            Clock(start, dt_seconds=dt_seconds, steps=steps),
            components=(component,),
            run_order=(component.name,),
            log_level="WARNING",
        )
        return np.asarray(
            coupler.run().component(component.name).field("temperature_2m")
        )

    expected_mean = 0.5 * (run_without_output(1) + run_without_output(2))
    component = make_slab_atmosphere(
        grid,
        output=OutputSpec(
            period=PeriodOutput(
                frequency="month",
                variables=("temperature_2m",),
            )
        ),
    )
    coupler = Coupler(
        Clock(start, dt_seconds=dt_seconds, steps=2),
        components=(component,),
        run_order=(component.name,),
        log_level="WARNING",
    )

    coupler.run(
        output=OutputTarget(
            tmp_path,
            write_final_fields=False,
            write_snapshots=False,
        )
    )

    with h5netcdf.File(tmp_path / "atm.averages.2000-01.nc", "r") as dataset:
        assert set(dataset.variables) == {
            "time",
            "latitude",
            "longitude",
            "temperature_2m",
        }
        np.testing.assert_allclose(
            dataset.variables["temperature_2m"][0],
            expected_mean,
        )


def test_data_period_file_contains_declared_outputs_only(tmp_path: Path) -> None:
    grid = make_test_grid(name="data-period")
    component = time_interpolated_data_component(
        name="DATA",
        grid=grid,
        fields={"temperature": jnp.full(grid.shape, 280.0)},
        inputs=("forcing",),
        outputs=("temperature",),
        initial_fields={"forcing": 1.0},
    )
    coupler = Coupler(
        Clock(datetime(2000, 1, 1), dt_seconds=3600.0, steps=1),
        components=(component,),
        run_order=(component.name,),
        log_level="WARNING",
    )

    coupler.run(
        output=OutputTarget(
            tmp_path,
            write_final_fields=False,
            write_snapshots=False,
        )
    )

    with h5netcdf.File(tmp_path / "data.averages.2000-01-01.nc", "r") as dataset:
        np.testing.assert_allclose(dataset.variables["temperature"][0], 280.0)
        assert "forcing" not in dataset.variables


def test_custom_components_remain_period_output_opt_in(tmp_path: Path) -> None:
    grid = make_test_grid(name="custom-output")
    component = CallableComponent(
        "CUSTOM",
        grid,
        lambda fields: {"temperature": fields["temperature"] + 1.0},
        spec=ComponentSpec(
            outputs=("temperature",),
            initial_fields={"temperature": 280.0},
        ),
    )
    coupler = Coupler(
        Clock(datetime(2000, 1, 1), dt_seconds=3600.0, steps=1),
        components=(component,),
        run_order=(component.name,),
        log_level="WARNING",
    )

    assert component.spec.output.period is None
    coupler.run(
        output=OutputTarget(
            tmp_path,
            write_final_fields=False,
            write_snapshots=False,
        )
    )
    assert not tuple(tmp_path.glob("*.averages.*.nc"))
