from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from vercor.clock import Clock, ModelDateTime
from vercor.components.base import Component, Shared
from vercor.grid import RectilinearGrid
from vercor.settings import VercorSettings
from vercor.types import RuntimeArray


def make_test_grid(
    name: str = "grid",
    *,
    longitude: NDArray | None = None,
    latitude: NDArray | None = None,
    binary_mask: NDArray | None = None,
) -> RectilinearGrid:
    lon = (
        np.asarray(longitude, dtype=float)
        if longitude is not None
        else np.asarray([0.0, 1.0], dtype=float)
    )
    lat = (
        np.asarray(latitude, dtype=float)
        if latitude is not None
        else np.asarray([-1.0, 1.0], dtype=float)
    )
    mask = None if binary_mask is None else np.asarray(binary_mask)
    return RectilinearGrid(name=name, longitude=lon, latitude=lat, binary_mask=mask)


@dataclass
class CoverageCouplerStub:
    clock: Clock = field(
        default_factory=lambda: Clock(
            start=datetime(2000, 1, 1),
            dt_seconds=60.0,
            steps=1,
        )
    )
    settings: VercorSettings = field(default_factory=VercorSettings)
    logger: logging.Logger = field(
        default_factory=lambda: logging.getLogger("coverage-tests")
    )
    appended_components: list[str] = field(default_factory=list)

    def append_masks_to_output(self, name: str, shared_fields: Shared) -> None:
        self.appended_components.append(name)
        shared_fields[f"mask_for_{name.lower()}"] = (
            np.ones((1, 1), dtype=float),
            self.clock.start,
            name,
        )


class DummyComponent(Component):
    def initialize(self, coupler: Any) -> None:
        self.data.setdefault("temperature", np.zeros(self.grid.shape, dtype=float))

    def step(
        self,
        dt: timedelta,
        time: datetime | ModelDateTime,
        coupler: Any,
    ) -> None:
        self.data["step_seconds"] = np.asarray(dt.total_seconds(), dtype=float)
        self.data["step_marker"] = np.full(self.grid.shape, 1.0, dtype=float)


class RecordingRegridder:
    def __init__(
        self,
        *,
        scalar_result: RuntimeArray | None = None,
        vector_result: tuple[RuntimeArray, RuntimeArray] | None = None,
    ) -> None:
        self.scalar_result = scalar_result
        self.vector_result = vector_result
        self.calls: list[tuple[NDArray, ...]] = []

    def __call__(
        self, *args: RuntimeArray
    ) -> RuntimeArray | tuple[RuntimeArray, RuntimeArray]:
        self.calls.append(tuple(np.asarray(arg) for arg in args))
        if len(args) == 1:
            if self.scalar_result is not None:
                return self.scalar_result
            return np.asarray(args[0])

        if self.vector_result is not None:
            return self.vector_result
        return np.asarray(args[0]), np.asarray(args[1])
