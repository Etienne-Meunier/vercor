from __future__ import annotations

from datetime import datetime
from typing import Any, cast

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
        np.full((2, 2), 5.0),
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
                scalar_result=np.asarray([[2.0, 4.0], [6.0, 8.0]])
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
    assert_allclose_compact(
        destination.incoming_fields.u_velocity.data,
        np.full((2, 2), 9.0),
    )
    assert_allclose_compact(
        destination.incoming_fields.v_velocity.data,
        np.full((2, 2), -9.0),
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
