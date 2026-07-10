from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime
import inspect
import importlib
from pathlib import Path
from typing import Any, cast

import h5netcdf
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.assertions import assert_allclose_compact
from tests._coverage_support import make_test_grid
from tests._runtime_helpers import (
    replace_runtime_topology_maps,
)
from vercor.clock import Clock
from vercor.components import Component, ComponentSpec
from vercor.exceptions import ComponentError, CouplerError
from vercor.setups._slab.atmosphere import make_slab_atmosphere
from vercor.setups._slab.land import make_slab_land
from vercor.setups._slab.ocean import make_slab_ocean
from vercor.setups._slab.seaice import make_slab_seaice
from vercor.coupler import Coupler
from vercor.exchanges import Exchange
from vercor.output import OutputConfig, PeriodOutput
from vercor.runtime import RuntimeOptions
from vercor._runtime.state import ComponentRuntimeState
from vercor.state import RunState
from vercor._runtime.stores import FieldStore


class _IdentityRegridder:
    def regrid(self, field: Any) -> Any:
        return jnp.asarray(field)

    def regrid_vector(self, u: Any, v: Any) -> tuple[Any, Any]:
        return jnp.asarray(u), jnp.asarray(v)


def _identity_factory(*args: Any, **kwargs: Any) -> _IdentityRegridder:
    _ = args, kwargs
    return _IdentityRegridder()


def _component_state(
    name: str,
    data: dict[str, jax.Array],
    receives: tuple[str, ...],
    sends: tuple[str, ...],
) -> ComponentRuntimeState:
    _ = name
    zeros = jnp.zeros((2, 2), dtype=jnp.float64)
    return ComponentRuntimeState(
        fields=FieldStore.from_mapping(
            {
                field: data.get(field, zeros)
                for field in sorted(set(data) | set(receives) | set(sends))
            }
        ),
        received=FieldStore.from_mapping(
            {field: data.get(field, zeros) for field in receives}
        ),
        sent=FieldStore.from_mapping(
            {field: data.get(field, zeros) for field in sends}
        ),
    )


def _make_coupler(steps: int) -> Coupler:
    grid = make_test_grid(name="runtime-run")
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=3600.0, steps=steps),
        components=(
            make_slab_atmosphere(grid),
            make_slab_ocean(grid),
            make_slab_land(grid),
            make_slab_seaice(grid),
        ),
        exchanges=(
            Exchange(
                source="OCN",
                target="ATM",
                fields=["sea_surface_temperature"],
                regrid=cast(Any, _identity_factory),
            ),
            Exchange(
                source="ATM",
                target="OCN",
                fields=["sensible_heat_flux", "latent_heat_flux"],
                regrid=cast(Any, _identity_factory),
            ),
            Exchange(
                source="ATM",
                target="LND",
                fields=["latent_heat_flux"],
                regrid=cast(Any, _identity_factory),
            ),
            Exchange(
                source="OCN",
                target="ICE",
                fields=["sea_surface_temperature"],
                regrid=cast(Any, _identity_factory),
            ),
        ),
        run_order=(
            "ATM",
            "OCN",
            "LND",
            "ICE",
        ),
    )
    regridders = cast(
        Any,
        {
            ("OCN", "ATM", "_identity_factory"): _IdentityRegridder(),
            ("ATM", "OCN", "_identity_factory"): _IdentityRegridder(),
            ("ATM", "LND", "_identity_factory"): _IdentityRegridder(),
            ("OCN", "ICE", "_identity_factory"): _IdentityRegridder(),
        },
    )
    replace_runtime_topology_maps(
        coupler,
        regridders=regridders,
        fractional_masks={
            key: jnp.ones((2, 2), dtype=jnp.float64) for key in regridders
        },
    )
    return coupler


def _make_initial_state(sea_surface_temperature: jax.Array) -> RunState:
    zeros = jnp.zeros_like(sea_surface_temperature)
    temperature_2m = jnp.full_like(sea_surface_temperature, 288.15)
    components = (
        _component_state(
            "ATM",
            {
                "temperature_2m": temperature_2m,
                "sensible_heat_flux": zeros,
                "latent_heat_flux": zeros,
                "u_velocity_10m": zeros,
                "v_velocity_10m": zeros,
                "sea_surface_temperature": sea_surface_temperature,
                "land_surface_temperature": temperature_2m,
                "soil_moisture": zeros,
                "ice_fraction": zeros,
            },
            receives=("sea_surface_temperature",),
            sends=(
                "temperature_2m",
                "sensible_heat_flux",
                "latent_heat_flux",
                "u_velocity_10m",
                "v_velocity_10m",
            ),
        ),
        _component_state(
            "OCN",
            {
                "sea_surface_temperature": sea_surface_temperature,
                "sensible_heat_flux": zeros,
                "latent_heat_flux": zeros,
                "u_velocity_10m": zeros,
                "v_velocity_10m": zeros,
                "u_velocity": zeros,
                "v_velocity": zeros,
                "specific_humidity": zeros,
                "temperature": zeros,
                "model_level_height": zeros,
                "net_shortwave_radiation_flux": zeros,
                "downward_longwave_radiation_flux": zeros,
                "ice_fraction": zeros,
            },
            receives=("sensible_heat_flux", "latent_heat_flux"),
            sends=("sea_surface_temperature",),
        ),
        _component_state(
            "LND",
            {
                "soil_moisture": jnp.full_like(sea_surface_temperature, 0.3),
                "land_surface_temperature": temperature_2m,
                "latent_heat_flux": zeros,
                "sensible_heat_flux": zeros,
            },
            receives=("latent_heat_flux",),
            sends=("soil_moisture", "land_surface_temperature"),
        ),
        _component_state(
            "ICE",
            {
                "ice_fraction": zeros,
                "sea_surface_temperature": sea_surface_temperature,
            },
            receives=("sea_surface_temperature",),
            sends=("ice_fraction",),
        ),
    )
    return RunState._from_runtime(
        component_names=("ATM", "OCN", "LND", "ICE"),
        components=components,
        fractional_masks=FieldStore.from_mapping(
            {
                "OCN|ATM|_identity_factory": jnp.ones_like(sea_surface_temperature),
                "ATM|OCN|_identity_factory": jnp.ones_like(sea_surface_temperature),
                "ATM|LND|_identity_factory": jnp.ones_like(sea_surface_temperature),
                "OCN|ICE|_identity_factory": jnp.ones_like(sea_surface_temperature),
            }
        ),
    )


def _runtime_state_with_sst(value: float) -> RunState:
    return _make_initial_state(jnp.full((2, 2), value, dtype=jnp.float64))


def _block_until_ready(value: RunState) -> RunState:
    for leaf in jax.tree_util.tree_leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return value


def _runtime_treedef_repr(value: RunState) -> str:
    return repr(jax.tree_util.tree_structure(value))


def _make_output_component(
    *,
    name: str = "model",
    frequency: str = "day",
    variables: tuple[str, ...] = (),
    snapshot_writer: Callable[..., None] | None = None,
) -> Component:
    grid = make_test_grid(name=f"period-output-{name}")

    def step(fields: Mapping[str, Any]) -> dict[str, Any]:
        return {"temperature": fields["temperature"] + 1.0}

    return Component.from_step(
        name=name,
        grid=grid,
        step=step,
        spec=ComponentSpec(
            outputs=("temperature",),
            defaults={"temperature": 0.0},
            output=OutputConfig(
                snapshot_writer=snapshot_writer,
                period=PeriodOutput(
                    frequency=cast(Any, frequency),
                    variables=variables,
                ),
            ),
        ),
    )


def _make_period_output_coupler(
    *,
    execution: str,
    frequency: str = "day",
    steps: int = 2,
    dt_seconds: float = 86_400.0,
    start: datetime = datetime(2000, 1, 1),
    component: Component | None = None,
) -> Coupler:
    selected_component = component or _make_output_component(frequency=frequency)
    return Coupler(
        clock=Clock(
            start=start,
            dt_seconds=dt_seconds,
            steps=steps,
        ),
        components=(selected_component,),
        run_order=(selected_component.name,),
        runtime=RuntimeOptions(execution=cast(Any, execution)),
        log_level="WARNING",
    )


def _read_period_temperatures(output_dir: Path) -> list[np.ndarray]:
    values = []
    for path in sorted(output_dir.glob("model.averages.*.nc")):
        with h5netcdf.File(path, "r") as dataset:
            values.append(np.asarray(dataset.variables["temperature"]))
    return values


@pytest.mark.parametrize("execution", ["host", "auto", "jax"])
def test_period_output_values_and_cadence_are_backend_consistent(
    execution: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / execution
    output_dir.mkdir()
    monkeypatch.chdir(output_dir)

    _make_period_output_coupler(execution=execution).run()

    actual = _read_period_temperatures(output_dir)
    assert len(actual) == 2
    assert_allclose_compact(actual[0], np.full((1, 2, 2), 1.0))
    assert_allclose_compact(actual[1], np.full((1, 2, 2), 2.0))


@pytest.mark.parametrize(
    ("frequency", "start", "dt_seconds", "steps", "expected_files", "means"),
    [
        ("step", datetime(2000, 1, 1), 86_400.0, 2, 2, (1.0, 2.0)),
        ("day", datetime(2000, 1, 1), 43_200.0, 4, 2, (1.5, 3.5)),
        ("month", datetime(2000, 1, 31), 86_400.0, 1, 1, (1.0,)),
        ("year", datetime(2000, 12, 31), 86_400.0, 1, 1, (1.0,)),
    ],
)
def test_period_output_precomputes_all_frequency_boundaries_and_resets(
    frequency: str,
    start: datetime,
    dt_seconds: float,
    steps: int,
    expected_files: int,
    means: tuple[float, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    _make_period_output_coupler(
        execution="jax",
        frequency=frequency,
        start=start,
        dt_seconds=dt_seconds,
        steps=steps,
    ).run()

    actual = _read_period_temperatures(tmp_path)
    assert len(actual) == expected_files
    for values, mean in zip(actual, means, strict=True):
        assert_allclose_compact(values, np.full((1, 2, 2), mean))


def test_mixed_component_period_frequencies_coexist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    daily = _make_output_component(name="daily", frequency="day")
    monthly = _make_output_component(name="monthly", frequency="month")
    coupler = Coupler(
        clock=Clock(datetime(2000, 1, 30), 86_400.0, 3),
        components=(daily, monthly),
        run_order=("daily", "monthly"),
        runtime=RuntimeOptions(execution="jax"),
        log_level="WARNING",
    )

    coupler.run()

    assert len(tuple(tmp_path.glob("daily.averages.*.nc"))) == 3
    monthly_paths = tuple(tmp_path.glob("monthly.averages.*.nc"))
    assert len(monthly_paths) == 1
    with h5netcdf.File(monthly_paths[0], "r") as dataset:
        assert_allclose_compact(
            np.asarray(dataset.variables["temperature"]),
            np.full((1, 2, 2), 1.5),
        )


@pytest.mark.parametrize(
    ("steps", "frequency"),
    [(0, "step"), (1, "day")],
)
def test_zero_step_and_incomplete_periods_do_not_write(
    steps: int,
    frequency: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    _make_period_output_coupler(
        execution="jax",
        frequency=frequency,
        steps=steps,
        dt_seconds=3_600.0,
    ).run()

    assert not tuple(tmp_path.glob("model.averages.*.nc"))


def test_period_output_rejects_unknown_variable_before_stepping() -> None:
    component = _make_output_component(variables=("missing",))
    coupler = _make_period_output_coupler(execution="jax", component=component)

    with pytest.raises(
        ComponentError,
        match="component 'model'.*unknown runtime field 'missing'",
    ):
        coupler.initial_state()


def test_period_output_empty_variable_selection_defaults_to_declared_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    _make_period_output_coupler(execution="jax", steps=1).run()

    with h5netcdf.File(tmp_path / "model.averages.2000-01-01.nc", "r") as dataset:
        assert "temperature" in dataset.variables


def test_period_output_rejects_outer_jit_and_grad() -> None:
    coupler = _make_period_output_coupler(execution="jax", steps=1)
    state = coupler.initial_state()

    with pytest.raises(CouplerError, match="Period output is an I/O workflow"):
        jax.jit(coupler.run)(state)

    def objective(value: jax.Array) -> jax.Array:
        traced_state = state.replace_fields(
            "model",
            {"temperature": jnp.full((2, 2), value)},
        )
        return jnp.sum(
            coupler.run(traced_state).component("model").field("temperature")
        )

    with pytest.raises(CouplerError, match="Differentiated.*disable"):
        jax.grad(objective)(jnp.asarray(1.0))


def test_snapshot_output_still_runs_with_period_output_configured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshots: list[Path] = []

    def write_snapshot(context: Any) -> None:
        snapshots.append(context.output_path)

    component = _make_output_component(snapshot_writer=write_snapshot)
    coupler = _make_period_output_coupler(
        execution="jax",
        steps=1,
        component=component,
    )
    monkeypatch.chdir(tmp_path)

    final_state = coupler.run()
    coupler.write_outputs(final_state, output_dir=tmp_path)

    assert snapshots == [tmp_path / "model.snapshot.nc"]


def test_period_file_io_stays_outside_scanned_chunk_body() -> None:
    backends = importlib.import_module("vercor._runtime.backends")
    source = inspect.getsource(backends._run_period_output_scanned_chunk)

    assert "write_period_output_boundary" not in source
    assert "write_netcdf" not in source
    assert "io_callback" not in source


def test_run_executes_pure_scanned_runtime_for_same_shapes_and_metadata() -> None:
    coupler = _make_coupler(steps=2)

    first = _block_until_ready(coupler.run(_runtime_state_with_sst(288.15)))
    second = _block_until_ready(coupler.run(_runtime_state_with_sst(291.15)))

    assert first._component_state("OCN").fields.get(
        "sea_surface_temperature"
    ).shape == (
        2,
        2,
    )
    assert second._component_state("OCN").fields.get(
        "sea_surface_temperature"
    ).shape == (
        2,
        2,
    )


def test_run_api_does_not_expose_state_donation() -> None:
    signature = inspect.signature(Coupler.run)

    assert "donate_state" not in signature.parameters


def test_run_preserves_runtime_treedef() -> None:
    coupler = _make_coupler(steps=1)

    first_state = _runtime_state_with_sst(287.15)
    second_state = _runtime_state_with_sst(292.15)
    first_final = _block_until_ready(coupler.run(first_state))
    second_final = _block_until_ready(coupler.run(second_state))

    expected_treedef = _runtime_treedef_repr(first_state)
    assert _runtime_treedef_repr(second_state) == expected_treedef
    assert _runtime_treedef_repr(first_final) == expected_treedef
    assert _runtime_treedef_repr(second_final) == expected_treedef
    assert first_final.component_names == first_state.component_names

    for before, after in zip(first_state._components, first_final._components):
        assert after.fields.field_names == before.fields.field_names
        assert after.received.field_names == before.received.field_names
        assert after.sent.field_names == before.sent.field_names


def test_runtime_profile_harness_exposes_cli_entrypoint() -> None:
    profile_runtime = importlib.import_module("examples.profile_runtime")

    assert callable(profile_runtime.main)
    parser = profile_runtime.build_parser()
    args = parser.parse_args(["--steps", "3", "--log-level", "WARNING"])
    assert args.steps == 3
    assert args.log_level == "WARNING"
    assert not hasattr(args, "donate_state")


def test_runtime_profile_harness_runs_small_slab_profile() -> None:
    profile_runtime = importlib.import_module("examples.profile_runtime")

    result = profile_runtime.profile_runtime(
        steps=1,
        grid_nx=2,
        grid_ny=2,
        log_level="WARNING",
    )

    assert result.run_seconds >= 0.0
    assert not hasattr(result, "compiled_cache_entries")
    assert result.final_state_leaves > 0
