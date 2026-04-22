from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from vercor.clock import DateTime360
from vercor.components import JAXGCM


def _make_component(output_frequency: str | None) -> JAXGCM:
    component = JAXGCM.__new__(JAXGCM)
    component.output_frequency = output_frequency
    return component


@pytest.mark.fast_always
def test_should_write_output_frequency_cases(select_fast_cases) -> None:
    cases = [
        ("none-always", None, datetime(2026, 2, 20, 0, 0, 0), timedelta(hours=1), True),
        (
            "day-boundary",
            "day",
            datetime(2026, 2, 20, 23, 0, 0),
            timedelta(hours=1),
            True,
        ),
        (
            "day-not-boundary",
            "day",
            datetime(2026, 2, 20, 22, 0, 0),
            timedelta(hours=1),
            False,
        ),
        (
            "month-boundary",
            "month",
            datetime(2026, 2, 28, 23, 0, 0),
            timedelta(hours=1),
            True,
        ),
        (
            "month-not-boundary",
            "month",
            datetime(2026, 2, 27, 23, 0, 0),
            timedelta(hours=1),
            False,
        ),
        (
            "year-boundary",
            "year",
            datetime(2026, 12, 31, 23, 0, 0),
            timedelta(hours=1),
            True,
        ),
        (
            "year-not-boundary",
            "year",
            datetime(2026, 12, 30, 23, 0, 0),
            timedelta(hours=1),
            False,
        ),
        (
            "invalid-frequency",
            "hour",
            datetime(2026, 2, 20, 23, 0, 0),
            timedelta(hours=1),
            False,
        ),
        (
            "360-day-boundary",
            "day",
            DateTime360(2026, 2, 20, 23, 0, 0, 0, 50),
            timedelta(hours=1),
            True,
        ),
        (
            "360-month-boundary",
            "month",
            DateTime360(2026, 2, 30, 23, 0, 0, 0, 60),
            timedelta(hours=1),
            True,
        ),
        (
            "360-year-boundary",
            "year",
            DateTime360(2026, 12, 30, 23, 0, 0, 0, 360),
            timedelta(hours=1),
            True,
        ),
    ]

    for _case_id, frequency, time, dt, expected in select_fast_cases(
        cases,
        case_id=lambda case: case[0],
        min_cases=3,
    ):
        component = _make_component(frequency)
        assert component._should_write_output(time, dt) is expected
