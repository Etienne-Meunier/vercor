from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from vercor.clock import Clock
from vercor.grid import RectilinearGrid
from vercor.settings import VercorSettings
from vercor.types import RuntimeArray


@dataclass
class DummyCoupler:
    clock: Clock
    settings: VercorSettings


def make_coupler(year_in_seconds: float) -> DummyCoupler:
    clock = Clock(start=datetime(2000, 1, 1), dt_seconds=1.0, steps=1)
    settings = VercorSettings(year_in_seconds=year_in_seconds)
    return DummyCoupler(clock=clock, settings=settings)


@dataclass
class DummyComponentA:
    name: str = "a"


@dataclass
class DummyComponentB:
    name: str = "b"


@dataclass
class DummyGridComponent:
    grid: RectilinearGrid
    fields: dict[str, RuntimeArray]

    @property
    def data(self) -> dict[str, RuntimeArray]:
        return self.fields

    def get(self, field_name: str) -> RuntimeArray:
        if field_name not in self.fields:
            raise KeyError(field_name)
        return self.fields[field_name]
