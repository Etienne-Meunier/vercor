"""Bundled slab and data factories declare generic step-period output."""

from __future__ import annotations

from datetime import datetime
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
    make_slab_atmosphere,
    make_slab_land,
    make_slab_ocean,
    make_slab_seaice,
)
from vercor.setups._data import jcm_land as jcm_land_module
from vercor.setups._data._component_helpers import (
    time_interpolated_data_component,
)


def _assert_step_period_output(component: Any) -> None:
    output = component.spec.output
    assert isinstance(output, OutputSpec)
    assert output.provider is None
    assert output.period == PeriodOutput(frequency="step")


@pytest.mark.parametrize(
    "factory",
    (
        make_slab_atmosphere,
        make_slab_land,
        make_slab_ocean,
        make_slab_seaice,
    ),
)
def test_all_bundled_slab_factories_declare_step_period_output(factory: Any) -> None:
    _assert_step_period_output(factory(make_test_grid(name="slab-output")))


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


def test_direct_jcm_land_data_factory_declares_step_period_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    component = jcm_land_module.make_jcm_land(
        cast(Any, coords),
        cast(Any, forcing),
        grid,
    )

    _assert_step_period_output(component)


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
