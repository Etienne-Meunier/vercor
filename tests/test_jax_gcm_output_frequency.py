from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import pytest

from tests.conftest import SelectFastCases
from vercor.calendar import DateTime360
from vercor.output import PeriodOutput
from vercor.output._period import should_write_period_output


@dataclass(frozen=True)
class OutputFrequencyCase:
    case_id: str
    output_period: PeriodOutput | None
    time: datetime | DateTime360
    dt: timedelta
    expected: bool


@pytest.mark.fast_always
def test_should_write_period_output_frequency_cases(
    select_fast_cases: SelectFastCases,
) -> None:
    cases = [
        OutputFrequencyCase(
            case_id="none-disabled",
            output_period=None,
            time=datetime(2026, 2, 20, 0, 0, 0),
            dt=timedelta(hours=1),
            expected=False,
        ),
        OutputFrequencyCase(
            case_id="step-always",
            output_period=PeriodOutput(frequency="step"),
            time=datetime(2026, 2, 20, 0, 0, 0),
            dt=timedelta(hours=1),
            expected=True,
        ),
        OutputFrequencyCase(
            case_id="day-boundary",
            output_period=PeriodOutput(frequency="day"),
            time=datetime(2026, 2, 20, 23, 0, 0),
            dt=timedelta(hours=1),
            expected=True,
        ),
        OutputFrequencyCase(
            case_id="day-not-boundary",
            output_period=PeriodOutput(frequency="day"),
            time=datetime(2026, 2, 20, 22, 0, 0),
            dt=timedelta(hours=1),
            expected=False,
        ),
        OutputFrequencyCase(
            case_id="month-boundary",
            output_period=PeriodOutput(frequency="month"),
            time=datetime(2026, 2, 28, 23, 0, 0),
            dt=timedelta(hours=1),
            expected=True,
        ),
        OutputFrequencyCase(
            case_id="month-not-boundary",
            output_period=PeriodOutput(frequency="month"),
            time=datetime(2026, 2, 27, 23, 0, 0),
            dt=timedelta(hours=1),
            expected=False,
        ),
        OutputFrequencyCase(
            case_id="year-boundary",
            output_period=PeriodOutput(frequency="year"),
            time=datetime(2026, 12, 31, 23, 0, 0),
            dt=timedelta(hours=1),
            expected=True,
        ),
        OutputFrequencyCase(
            case_id="year-not-boundary",
            output_period=PeriodOutput(frequency="year"),
            time=datetime(2026, 12, 30, 23, 0, 0),
            dt=timedelta(hours=1),
            expected=False,
        ),
        OutputFrequencyCase(
            case_id="360-day-boundary",
            output_period=PeriodOutput(frequency="day"),
            time=DateTime360(2026, 2, 20, 23, 0, 0, 0, 50),
            dt=timedelta(hours=1),
            expected=True,
        ),
        OutputFrequencyCase(
            case_id="360-month-boundary",
            output_period=PeriodOutput(frequency="month"),
            time=DateTime360(2026, 2, 30, 23, 0, 0, 0, 60),
            dt=timedelta(hours=1),
            expected=True,
        ),
        OutputFrequencyCase(
            case_id="360-year-boundary",
            output_period=PeriodOutput(frequency="year"),
            time=DateTime360(2026, 12, 30, 23, 0, 0, 0, 360),
            dt=timedelta(hours=1),
            expected=True,
        ),
    ]

    for case in select_fast_cases(
        cases,
        case_id=lambda case: case.case_id,
        min_cases=3,
    ):
        assert (
            should_write_period_output(
                case.output_period,
                time=case.time,
                dt=case.dt,
            )
            is case.expected
        )


def test_is_period_end_stays_false_within_same_day() -> None:
    time = datetime(2026, 2, 20, 12, 0, 0)

    assert (
        should_write_period_output(
            PeriodOutput(frequency="day"),
            time=time,
            dt=timedelta(minutes=30),
        )
        is False
    )


def test_period_output_rejects_invalid_frequency() -> None:
    with pytest.raises(ValueError, match="frequency must be one of"):
        PeriodOutput(frequency="hour")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="frequency must be one of"):
        PeriodOutput(frequency=12)  # type: ignore[arg-type]
