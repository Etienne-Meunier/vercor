from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests._coverage_support import DummyComponent, RecordingRegridder, make_test_grid
from tests.assertions import assert_allclose_compact
from vercor.clock import Clock
from vercor.components.base import Shared
from vercor.coupler import Coupler
from vercor.exceptions import ComponentError, CouplerError, ExchangerError
from vercor.exchange import Exchange
from vercor.regridders.bilinear import bilinear
from vercor.regridders.conservative import conservative
from vercor.run_sequence import RunSequence


class _RecordingLogger:
    def __init__(self) -> None:
        self.info_messages: list[str] = []
        self.warning_messages: list[str] = []
        self.debug_messages: list[str] = []

    def info(self, message: str) -> None:
        self.info_messages.append(message)

    def warning(self, message: str) -> None:
        self.warning_messages.append(message)

    def debug(self, message: str) -> None:
        self.debug_messages.append(message)


class _FinalizeComponent(DummyComponent):
    def __init__(self, name: str) -> None:
        super().__init__(name=name, grid=make_test_grid(name=name.lower()))
        self.finalize_calls: list[Path | None] = []

    def finalize(self, coupler: Any, output_file_mask: Path | None = None) -> None:
        _ = coupler
        self.finalize_calls.append(output_file_mask)


class _RunComponent:
    def __init__(self, name: str, events: list[str], timestamp: datetime) -> None:
        self.name = name
        self.events = events
        self.outgoing_fields = Shared()
        self.outgoing_fields["temperature"] = (np.ones((2, 2)), timestamp, name)

    def receive_fields(self, time: datetime) -> None:
        self.events.append(f"receive:{self.name}:{time.isoformat()}")

    def step(self, dt: Any, time: datetime, coupler: Any) -> None:
        _ = coupler
        self.events.append(f"step:{self.name}:{time.isoformat()}:{dt.total_seconds()}")

    def send_fields(self, time: datetime, coupler: Any) -> None:
        _ = coupler
        self.events.append(f"send:{self.name}:{time.isoformat()}")


def make_coupler() -> Coupler:
    return Coupler(clock=Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1))


@pytest.mark.fast_always
def test_coupler_register_and_run_sequence_validation() -> None:
    coupler = make_coupler()
    atmosphere = DummyComponent(name="ATM", grid=make_test_grid(name="atm"))
    coupler.register(cast(Any, atmosphere))

    with pytest.raises(CouplerError, match="already registered"):
        coupler.register(cast(Any, atmosphere))

    with pytest.raises(CouplerError, match="not registered in coupler"):
        coupler.set_components_run_sequence(RunSequence(order=["ATM", "OCN"]))


@pytest.mark.parametrize(
    ("registered_names", "source", "destination"),
    [
        (["ATM"], "OCN", "ATM"),
        (["OCN"], "OCN", "ATM"),
    ],
)
def test_coupler_initialize_rejects_missing_exchange_endpoints(
    registered_names: list[str],
    source: str,
    destination: str,
) -> None:
    coupler = make_coupler()
    components = {
        "ATM": DummyComponent(name="ATM", grid=make_test_grid(name="atm")),
        "OCN": DummyComponent(name="OCN", grid=make_test_grid(name="ocn")),
    }
    for name in registered_names:
        coupler.register(cast(Any, components[name]))

    coupler.add_exchange(
        Exchange(
            source=source,
            destination=destination,
            field_names=["temperature"],
            regridder_factory=bilinear,
        )
    )

    with pytest.raises(CouplerError, match="not registered in coupler"):
        coupler.initialize()


def test_coupler_initialize_happy_path_builds_unique_regridders_and_supports_x64(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coupler = make_coupler()
    logger = _RecordingLogger()
    coupler.logger = cast(Any, logger)
    coupler.settings.enable_x64 = False

    lnd_mask = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    components = {
        "ATM": DummyComponent(name="ATM", grid=make_test_grid(name="atm")),
        "OCN": DummyComponent(
            name="OCN",
            grid=make_test_grid(
                name="ocn", binary_mask=np.asarray([[0.0, 1.0], [1.0, 0.0]])
            ),
        ),
        "LND": DummyComponent(
            name="LND",
            grid=make_test_grid(name="lnd", binary_mask=lnd_mask),
        ),
        "ICE": DummyComponent(name="ICE", grid=make_test_grid(name="ice")),
    }
    components["ATM"].data.update(
        {
            "downward_longwave_radiation_flux": np.full((2, 2), 1.0),
            "temperature_2m": np.full((2, 2), 2.0),
            "sensible_heat_flux": np.full((2, 2), 3.0),
        }
    )
    components["OCN"].data.update(
        {
            "temperature": np.full((2, 2), 4.0),
            "specific_humidity": np.full((2, 2), 5.0),
        }
    )
    components["LND"].data["soil_moisture"] = np.full((2, 2), 6.0)
    components["ICE"].data["ice_fraction"] = np.full((2, 2), 7.0)

    for component in components.values():
        coupler.register(cast(Any, component))

    exchanges = [
        Exchange(
            source="OCN",
            destination="ATM",
            field_names=["temperature"],
            regridder_factory=bilinear,
        ),
        Exchange(
            source="OCN",
            destination="ATM",
            field_names=["specific_humidity"],
            regridder_factory=bilinear,
        ),
        Exchange(
            source="ATM",
            destination="OCN",
            field_names=["downward_longwave_radiation_flux"],
            regridder_factory=conservative,
        ),
        Exchange(
            source="LND",
            destination="ATM",
            field_names=["soil_moisture"],
            regridder_factory=bilinear,
        ),
        Exchange(
            source="ATM",
            destination="LND",
            field_names=["temperature_2m"],
            regridder_factory=bilinear,
        ),
        Exchange(
            source="ICE",
            destination="ATM",
            field_names=["ice_fraction"],
            regridder_factory=bilinear,
        ),
        Exchange(
            source="ATM",
            destination="ICE",
            field_names=["sensible_heat_flux"],
            regridder_factory=bilinear,
        ),
    ]
    created_keys: list[tuple[str, str]] = []
    for exchange in exchanges:

        def fake_create(
            source_grid: Any,
            destination_grid: Any,
            exchange_name: str = exchange.name,
        ) -> RecordingRegridder:
            _ = exchange_name
            created_keys.append((source_grid.name, destination_grid.name))
            return RecordingRegridder()

        monkeypatch.setattr(exchange, "create", fake_create)
        coupler.add_exchange(exchange)

    def fake_create_exchange_masks() -> None:
        coupler.ocn_fmask_on_atm_grid = np.full((2, 2), 0.4)
        coupler.lnd_fmask_on_atm_grid = np.full((2, 2), 0.6)
        coupler.lnd_bmask_on_atm_grid = lnd_mask

    monkeypatch.setattr(coupler, "_create_exchange_masks", fake_create_exchange_masks)
    jax_calls: list[tuple[str, bool]] = []
    monkeypatch.setitem(
        sys.modules,
        "jax",
        SimpleNamespace(
            config=SimpleNamespace(
                update=lambda key, value: jax_calls.append((key, value))
            )
        ),
    )

    coupler.initialize(enable_x64_computations=True)

    assert coupler.settings.enable_x64 is True
    assert jax_calls == [("jax_enable_x64", True)]
    assert len(created_keys) == 6
    assert len(coupler._regridders) == 6
    assert any("already exists" in message for message in logger.warning_messages)
    assert isinstance(coupler._binary_masks[("ATM", "OCN", "conservative")], jax.Array)
    assert isinstance(
        coupler._fractional_masks[("ATM", "OCN", "conservative")], jax.Array
    )
    assert components["ATM"]._fields2import == [
        "temperature",
        "specific_humidity",
        "soil_moisture",
        "ice_fraction",
    ]
    assert components["ATM"]._fields2export == [
        "downward_longwave_radiation_flux",
        "temperature_2m",
        "sensible_heat_flux",
    ]
    assert_allclose_compact(
        coupler._fractional_masks[("OCN", "ATM", "bilinear")],
        np.full((2, 2), 0.4),
    )
    assert_allclose_compact(
        coupler._binary_masks[("LND", "ATM", "bilinear")],
        lnd_mask,
    )


def test_patch_exchange_masks_updates_only_expected_bilinear_pairs() -> None:
    coupler = make_coupler()
    ocn_key = ("OCN", "ATM", "bilinear")
    lnd_key = ("LND", "ATM", "bilinear")
    other_key = ("OCN", "ATM", "conservative")

    coupler._binary_masks = {
        ocn_key: np.zeros((2, 2)),
        lnd_key: np.zeros((2, 2)),
        other_key: np.full((2, 2), 9.0),
    }
    coupler._fractional_masks = {
        ocn_key: np.zeros((2, 2)),
        lnd_key: np.zeros((2, 2)),
        other_key: np.full((2, 2), 7.0),
    }
    coupler.ocn_fmask_on_atm_grid = np.full((2, 2), 0.25)
    coupler.lnd_bmask_on_atm_grid = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    coupler.lnd_fmask_on_atm_grid = np.full((2, 2), 0.75)

    coupler._patch_exchange_masks()

    assert_allclose_compact(coupler._fractional_masks[ocn_key], np.full((2, 2), 0.25))
    assert_allclose_compact(
        coupler._binary_masks[lnd_key], np.asarray([[1.0, 0.0], [0.0, 1.0]])
    )
    assert_allclose_compact(coupler._fractional_masks[lnd_key], np.full((2, 2), 0.75))
    assert_allclose_compact(coupler._binary_masks[other_key], np.full((2, 2), 9.0))
    assert_allclose_compact(coupler._fractional_masks[other_key], np.full((2, 2), 7.0))


def test_validate_land_mask_consistency_rejects_shape_and_value_mismatches() -> None:
    coupler = make_coupler()
    coupler.components = cast(
        Any,
        {
            "LND": DummyComponent(
                name="LND",
                grid=make_test_grid(name="lnd", binary_mask=np.ones((3, 2))),
            )
        },
    )
    coupler.lnd_bmask_on_atm_grid = np.ones((2, 2))

    with pytest.raises(CouplerError, match="does not match atmospheric grid shape"):
        coupler._validate_land_mask_consistency()

    coupler.components["LND"] = cast(
        Any,
        DummyComponent(
            name="LND",
            grid=make_test_grid(
                name="lnd",
                binary_mask=np.asarray([[1.0, 0.0], [1.0, 0.0]]),
            ),
        ),
    )
    coupler.lnd_bmask_on_atm_grid = np.asarray([[1.0, 0.0], [0.0, 1.0]])

    with pytest.raises(CouplerError, match="mismatched points: 2"):
        coupler._validate_land_mask_consistency()


def test_create_exchange_masks_rejects_non_identical_land_and_atmosphere_grids() -> (
    None
):
    coupler = make_coupler()
    coupler.components = cast(
        Any,
        {
            "ATM": DummyComponent(
                name="ATM",
                grid=make_test_grid(name="atm", latitude=np.asarray([0.0, 1.0])),
            ),
            "LND": DummyComponent(
                name="LND",
                grid=make_test_grid(name="lnd", latitude=np.asarray([0.0, 2.0])),
            ),
            "OCN": DummyComponent(
                name="OCN",
                grid=make_test_grid(
                    name="ocn", binary_mask=np.asarray([[1.0, 0.0], [0.0, 1.0]])
                ),
            ),
        },
    )

    with pytest.raises(CouplerError, match="must use identical horizontal grids"):
        coupler._create_exchange_masks()


def test_create_exchange_masks_rejects_missing_ocean_binary_mask() -> None:
    coupler = make_coupler()
    coupler.components = cast(
        Any,
        {
            "ATM": DummyComponent(name="ATM", grid=make_test_grid(name="atm")),
            "LND": DummyComponent(name="LND", grid=make_test_grid(name="lnd")),
            "OCN": DummyComponent(name="OCN", grid=make_test_grid(name="ocn")),
        },
    )

    with pytest.raises(ComponentError, match="has no binary mask defined"):
        coupler._create_exchange_masks()


def test_append_masks_to_output_appends_destination_exchange_masks() -> None:
    coupler = make_coupler()
    ocn_exchange = Exchange(
        source="OCN",
        destination="ATM",
        field_names=["temperature"],
        regridder_factory=bilinear,
    )
    lnd_exchange = Exchange(
        source="LND",
        destination="ATM",
        field_names=["temperature"],
        regridder_factory=bilinear,
    )
    coupler.exchanges = [ocn_exchange, lnd_exchange]
    coupler._binary_masks = {
        ("OCN", "ATM", "bilinear"): np.zeros((2, 2)),
        ("LND", "ATM", "bilinear"): np.ones((2, 2)),
    }
    coupler._fractional_masks = {
        ("OCN", "ATM", "bilinear"): np.full((2, 2), 0.25),
        ("LND", "ATM", "bilinear"): np.full((2, 2), 0.75),
    }

    shared = Shared()
    coupler.append_masks_to_output("ATM", shared)

    assert set(shared.field_names) == {
        "bmask_OCN_ATM_bilinear",
        "fmask_OCN_ATM_bilinear",
        "bmask_LND_ATM_bilinear",
        "fmask_LND_ATM_bilinear",
    }
    assert shared.component_names()["fmask_LND_ATM_bilinear"] == "ATM"


def test_interpolate_and_dispatch_fields_handles_scalar_and_vector_paths() -> None:
    coupler = make_coupler()
    source = DummyComponent(name="OCN", grid=make_test_grid(name="ocn"))
    destination = DummyComponent(name="ATM", grid=make_test_grid(name="atm"))
    timestamp = coupler.clock.start

    source.outgoing_fields["temperature"] = (
        jnp.full((2, 2), 5.0),
        timestamp,
        "OCN",
    )
    source.outgoing_fields["u_velocity"] = (
        np.full((2, 2), 1.0),
        timestamp,
        "OCN",
    )
    source.outgoing_fields["v_velocity"] = (
        np.full((2, 2), -1.0),
        timestamp,
        "OCN",
    )

    scalar_exchange = Exchange(
        source="OCN",
        destination="ATM",
        field_names=["temperature"],
        regridder_factory=bilinear,
    )
    vector_exchange = Exchange(
        source="OCN",
        destination="ATM",
        field_names=[("u_velocity", "v_velocity")],
        regridder_factory=conservative,
    )
    coupler.components = cast(Any, {"OCN": source, "ATM": destination})
    coupler.exchanges = [scalar_exchange, vector_exchange]
    coupler._regridders = cast(
        Any,
        {
            ("OCN", "ATM", "bilinear"): RecordingRegridder(
                scalar_result=jnp.asarray([[2.0, 4.0], [6.0, 8.0]])
            ),
            ("OCN", "ATM", "conservative"): RecordingRegridder(
                vector_result=(
                    np.full((2, 2), 9.0),
                    np.full((2, 2), -9.0),
                )
            ),
        },
    )
    coupler._fractional_masks = {
        ("OCN", "ATM", "bilinear"): np.asarray([[1.0, 0.5], [0.0, 1.0]]),
        ("OCN", "ATM", "conservative"): np.ones((2, 2)),
    }

    coupler.interpolate_and_dispatch_fields(cast(Any, destination), timestamp)

    assert_allclose_compact(
        destination.incoming_fields.temperature.data,
        np.asarray([[2.0, 2.0], [0.0, 8.0]]),
    )
    assert isinstance(destination.incoming_fields.temperature.data, jax.Array)
    assert_allclose_compact(
        destination.incoming_fields.u_velocity.data,
        np.full((2, 2), 9.0),
    )
    assert_allclose_compact(
        destination.incoming_fields.v_velocity.data,
        np.full((2, 2), -9.0),
    )


def test_interpolate_and_dispatch_fields_accepts_mixed_numpy_and_jax_arrays() -> None:
    coupler = make_coupler()
    source = DummyComponent(name="OCN", grid=make_test_grid(name="ocn"))
    destination = DummyComponent(name="ATM", grid=make_test_grid(name="atm"))
    timestamp = coupler.clock.start

    source.outgoing_fields["temperature"] = (
        np.full((2, 2), 5.0),
        timestamp,
        "OCN",
    )

    exchange = Exchange(
        source="OCN",
        destination="ATM",
        field_names=["temperature"],
        regridder_factory=bilinear,
    )
    coupler.components = cast(Any, {"OCN": source, "ATM": destination})
    coupler.exchanges = [exchange]
    coupler._regridders = cast(
        Any,
        {
            ("OCN", "ATM", "bilinear"): RecordingRegridder(
                scalar_result=jnp.asarray([[2.0, 4.0], [6.0, 8.0]])
            )
        },
    )
    coupler._fractional_masks = {
        ("OCN", "ATM", "bilinear"): np.asarray([[1.0, 0.5], [0.0, 1.0]]),
    }

    coupler.interpolate_and_dispatch_fields(cast(Any, destination), timestamp)

    assert isinstance(destination.incoming_fields.temperature.data, jax.Array)
    assert_allclose_compact(
        destination.incoming_fields.temperature.data,
        np.asarray([[2.0, 2.0], [0.0, 8.0]]),
    )


def test_interpolate_and_dispatch_fields_rejects_missing_scalar_and_vector_fields() -> (
    None
):
    coupler = make_coupler()
    timestamp = coupler.clock.start

    scalar_source = DummyComponent(name="OCN", grid=make_test_grid(name="ocn"))
    scalar_destination = DummyComponent(name="ATM", grid=make_test_grid(name="atm"))
    scalar_exchange = Exchange(
        source="OCN",
        destination="ATM",
        field_names=["temperature"],
        regridder_factory=bilinear,
    )
    coupler.components = cast(Any, {"OCN": scalar_source, "ATM": scalar_destination})
    coupler.exchanges = [scalar_exchange]
    coupler._regridders = cast(
        Any,
        {("OCN", "ATM", "bilinear"): RecordingRegridder(scalar_result=np.ones((2, 2)))},
    )
    coupler._fractional_masks = {("OCN", "ATM", "bilinear"): np.ones((2, 2))}

    with pytest.raises(ExchangerError, match="Field temperature not present"):
        coupler.interpolate_and_dispatch_fields(
            cast(Any, scalar_destination), timestamp
        )

    vector_source = DummyComponent(name="OCN", grid=make_test_grid(name="ocn"))
    vector_destination = DummyComponent(name="ATM", grid=make_test_grid(name="atm"))
    vector_source.outgoing_fields["u_velocity"] = (
        np.ones((2, 2)),
        timestamp,
        "OCN",
    )
    vector_exchange = Exchange(
        source="OCN",
        destination="ATM",
        field_names=[("u_velocity", "v_velocity")],
        regridder_factory=conservative,
    )
    coupler.components = cast(Any, {"OCN": vector_source, "ATM": vector_destination})
    coupler.exchanges = [vector_exchange]
    coupler._regridders = cast(
        Any,
        {
            ("OCN", "ATM", "conservative"): RecordingRegridder(
                vector_result=(np.ones((2, 2)), np.ones((2, 2)))
            )
        },
    )
    coupler._fractional_masks = {("OCN", "ATM", "conservative"): np.ones((2, 2))}

    with pytest.raises(ExchangerError, match="Not all fields in vector"):
        coupler.interpolate_and_dispatch_fields(
            cast(Any, vector_destination), timestamp
        )


def test_coupler_run_rejects_components_with_empty_outgoing_fields() -> None:
    coupler = make_coupler()
    atmosphere = DummyComponent(name="ATM", grid=make_test_grid(name="atm"))
    coupler.components = cast(Any, {"ATM": atmosphere})
    coupler.run_sequence = RunSequence(order=["ATM"])

    with pytest.raises(ComponentError, match="outgoing fields were not initialized"):
        coupler.run()


def test_coupler_finalize_calls_finalize_on_all_components() -> None:
    coupler = make_coupler()
    components = {
        "ATM": _FinalizeComponent("ATM"),
        "OCN": _FinalizeComponent("OCN"),
    }
    coupler.components = cast(Any, components)

    coupler.finalize(Path("snapshot"))

    assert components["ATM"].finalize_calls == [Path("snapshot")]
    assert components["OCN"].finalize_calls == [Path("snapshot")]


def test_coupler_string_representations_include_registered_state() -> None:
    coupler = make_coupler()
    atmosphere = DummyComponent(name="ATM", grid=make_test_grid(name="atm"))
    ocean = DummyComponent(name="OCN", grid=make_test_grid(name="ocn"))
    coupler.register(cast(Any, atmosphere))
    coupler.register(cast(Any, ocean))
    coupler.add_exchange(
        Exchange(
            source="ATM",
            destination="OCN",
            field_names=["temperature"],
            regridder_factory=bilinear,
        )
    )
    coupler.set_components_run_sequence(RunSequence(order=["ATM", "OCN"]))

    rendered = str(coupler)
    representation = repr(coupler)

    assert "Coupler:" in rendered
    assert "<DummyComponent>(ATM)" in rendered
    assert "ATM --(bilinear)--> OCN" in rendered
    assert "ATM, OCN" in rendered
    assert "run_sequence=ATM -> OCN" in representation


def test_coupler_run_happy_path_dispatches_and_steps_in_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coupler = make_coupler()
    coupler.logger = cast(Any, _RecordingLogger())
    timestamp = coupler.clock.start
    events: list[str] = []
    atmosphere = _RunComponent("ATM", events, timestamp)
    ocean = _RunComponent("OCN", events, timestamp)
    coupler.components = cast(Any, {"ATM": atmosphere, "OCN": ocean})
    coupler.run_sequence = RunSequence(order=["ATM", "OCN"])

    def fake_dispatch(component: Any, time: datetime) -> None:
        events.append(f"dispatch:{component.name}:{time.isoformat()}")

    monkeypatch.setattr(coupler, "interpolate_and_dispatch_fields", fake_dispatch)

    coupler.run()

    assert events == [
        "dispatch:ATM:2000-01-01T00:00:00",
        "receive:ATM:2000-01-01T00:00:00",
        "step:ATM:2000-01-01T00:00:00:60.0",
        "send:ATM:2000-01-01T00:00:00",
        "dispatch:OCN:2000-01-01T00:00:00",
        "receive:OCN:2000-01-01T00:00:00",
        "step:OCN:2000-01-01T00:00:00:60.0",
        "send:OCN:2000-01-01T00:00:00",
    ]
