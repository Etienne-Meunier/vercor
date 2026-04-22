from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pytest

from tests.assertions import assert_allclose_compact
from vercor.exchange import Exchange
from vercor.grid import Grid, RectilinearGrid
from vercor.regridders.helpers import centers_to_edges, compute_land_mask
from vercor.run_sequence import RunSequence


@dataclass
class ExampleGrid(Grid):
    longitude_size: int = 3
    latitude_size: int = 2

    @property
    def shape(self) -> tuple[int, int]:
        return (self.latitude_size, self.longitude_size)


@pytest.mark.fast_always
def test_grid_and_rectilinear_grid_validations_and_reprs() -> None:
    grid = ExampleGrid(name="example", binary_mask=np.ones((2, 3)))
    assert "Grid name:  example" in str(grid)
    assert "longitude_size=3" in repr(grid)

    with pytest.raises(ValueError, match="Mask must be a 2D array"):
        ExampleGrid(name="bad-mask", binary_mask=np.ones((2, 3, 1)))

    rectilinear = RectilinearGrid(
        name="rect",
        longitude=np.asarray([0.0, 120.0, 240.0]),
        latitude=np.asarray([-45.0, 45.0]),
    )
    assert rectilinear.shape == (2, 3)
    assert "RectilinearGrid" in str(rectilinear)
    assert "shape=(2, 3)" in repr(rectilinear)

    with pytest.raises(ValueError, match="1D arrays"):
        RectilinearGrid(
            name="bad-dims",
            longitude=np.ones((2, 2)),
            latitude=np.asarray([-10.0, 10.0]),
        )

    with pytest.raises(ValueError, match="strictly monotonic"):
        RectilinearGrid(
            name="bad-order",
            longitude=np.asarray([0.0, 10.0, 5.0]),
            latitude=np.asarray([-10.0, 10.0]),
        )


def test_centers_to_edges_and_compute_land_mask_edge_cases() -> None:
    assert_allclose_compact(
        centers_to_edges(np.asarray([7.0]), "lon"), np.asarray([6.5, 7.5])
    )
    assert_allclose_compact(
        centers_to_edges(np.asarray([-89.0, 0.0, 89.0]), "lat"),
        np.asarray([-90.0, -44.5, 44.5, 90.0]),
    )

    periodic_edges = centers_to_edges(np.asarray([0.0, 90.0, 180.0, 270.0]), "lon")
    assert_allclose_compact(
        periodic_edges, np.asarray([-45.0, 45.0, 135.0, 225.0, 315.0])
    )

    clamped_edges = centers_to_edges(
        np.asarray([-170.0, -80.0, 10.0, 100.0, 190.0]),
        "lon",
    )
    assert clamped_edges[0] == -180.0
    assert clamped_edges[-1] == 180.0

    land_mask = compute_land_mask(np.asarray([[-0.5, 0.9995], [0.7, 1.0]]))
    assert_allclose_compact(land_mask, np.asarray([[1, 0], [1, 0]]))


def test_exchange_create_and_run_sequence_iteration() -> None:
    source_grid = RectilinearGrid(
        name="src",
        longitude=np.asarray([0.0, 180.0]),
        latitude=np.asarray([-45.0, 45.0]),
    )
    destination_grid = RectilinearGrid(
        name="dst",
        longitude=np.asarray([90.0, 270.0]),
        latitude=np.asarray([-30.0, 30.0]),
    )
    calls: list[tuple[RectilinearGrid, RectilinearGrid]] = []

    def dummy_factory(
        src: RectilinearGrid,
        dst: RectilinearGrid,
    ) -> object:
        calls.append((src, dst))
        return {"source": src.name, "destination": dst.name}

    exchange = Exchange(
        source="OCN",
        destination="ATM",
        field_names=["temperature", ("u_velocity", "v_velocity")],
        regridder_factory=cast(Any, dummy_factory),
    )

    created = exchange.create(source_grid, destination_grid)

    assert exchange.name == "OCN --(dummy_factory)--> ATM"
    assert exchange.interpolation_type == "dummy_factory"
    assert "Source component: OCN" in str(exchange)
    assert "fields=['temperature', ('u_velocity', 'v_velocity')]" in repr(exchange)
    assert created == {"source": "src", "destination": "dst"}
    assert calls == [(source_grid, destination_grid)]

    sequence = RunSequence(order=["OCN", "ATM", "LND"])
    assert list(sequence) == ["OCN", "ATM", "LND"]
