from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import pytest

from tests.conftest import SelectFastCases
from vercor.clock import Clock, DateTime360, DateTime365


@dataclass(frozen=True)
class StringCase:
    case_id: str
    model_time: DateTime360 | DateTime365
    formatter: str
    expected: str


@dataclass(frozen=True)
class ArithmeticCase:
    case_id: str
    base_time: DateTime360
    delta: timedelta
    expected: tuple[int, ...]


@dataclass(frozen=True)
class ClockIterationCase:
    case_id: str
    clock: Clock
    expected_times: list[datetime | DateTime360 | DateTime365]


@pytest.mark.fast_always
def test_model_datetime_string_and_repr_cases(
    select_fast_cases: SelectFastCases,
) -> None:
    cases = [
        StringCase(
            case_id="365-str-no-microseconds",
            model_time=DateTime365(2025, 1, 2, 3, 4, 5, 0, 2),
            formatter="str",
            expected=str(datetime(2025, 1, 2, 3, 4, 5)),
        ),
        StringCase(
            case_id="360-str-microseconds",
            model_time=DateTime360(2025, 1, 2, 3, 4, 5, 123456, 2),
            formatter="str",
            expected=str(datetime(2025, 1, 2, 3, 4, 5, 123456)),
        ),
        StringCase(
            case_id="360-repr",
            model_time=DateTime360(2026, 12, 30, 6, 7, 8, 900000, 360),
            formatter="repr",
            expected="DateTime360(2026, 12, 30, 6, 7, 8, 900000, 360)",
        ),
        StringCase(
            case_id="365-strftime-common",
            model_time=DateTime365(2025, 1, 2, 3, 4, 5, 123456, 2),
            formatter="%Y-%m-%d %H:%M:%S.%f",
            expected=datetime(2025, 1, 2, 3, 4, 5, 123456).strftime(
                "%Y-%m-%d %H:%M:%S.%f"
            ),
        ),
        StringCase(
            case_id="360-strftime-dayofyear",
            model_time=DateTime360(2026, 12, 30, 6, 7, 8, 900000, 360),
            formatter="%j",
            expected="360",
        ),
        StringCase(
            case_id="365-literal-percent-f",
            model_time=DateTime365(2025, 1, 2, 3, 4, 5, 123456, 2),
            formatter="%%f %f",
            expected="%f 123456",
        ),
    ]

    for case in select_fast_cases(
        cases, case_id=lambda case: case.case_id, min_cases=2
    ):
        if case.formatter == "str":
            assert str(case.model_time) == case.expected
        elif case.formatter == "repr":
            assert repr(case.model_time) == case.expected
        else:
            assert case.model_time.strftime(case.formatter) == case.expected


@pytest.mark.fast_always
def test_model_datetime_arithmetic_cases(
    select_fast_cases: SelectFastCases,
) -> None:
    cases = select_fast_cases(
        [
            ArithmeticCase(
                case_id="add-wrap-360",
                base_time=DateTime360(2025, 12, 30, 6, 0, 0, 0, 360),
                delta=timedelta(days=1),
                expected=(2026, 1, 1, 1, 6),
            ),
            ArithmeticCase(
                case_id="radd-hours",
                base_time=DateTime360(2025, 1, 1, 0, 0, 0, 0, 1),
                delta=timedelta(hours=12),
                expected=(2025, 1, 1, 12),
            ),
            ArithmeticCase(
                case_id="sub-second",
                base_time=DateTime360(2025, 1, 1, 0, 0, 0, 0, 1),
                delta=-timedelta(seconds=1),
                expected=(2024, 12, 30, 360, 23, 59, 59),
            ),
        ],
        case_id=lambda case: case.case_id,
        min_cases=2,
    )

    for case in cases:
        out = case.base_time + case.delta
        if case.case_id == "add-wrap-360":
            assert (
                out.year,
                out.month,
                out.day,
                out.day_of_year,
                out.hour,
            ) == case.expected
        elif case.case_id == "radd-hours":
            assert (out.year, out.month, out.day, out.hour) == case.expected
        else:
            assert isinstance(out, DateTime360)
            assert (
                out.year,
                out.month,
                out.day,
                out.day_of_year,
                out.hour,
                out.minute,
                out.second,
            ) == case.expected


def test_model_datetime_subtract_model_datetime_returns_timedelta() -> None:
    earlier = DateTime360(2025, 1, 1, 0, 0, 0, 0, 1)
    later = DateTime360(2025, 1, 2, 1, 2, 3, 400000, 2)

    diff = later - earlier
    assert diff == timedelta(days=1, hours=1, minutes=2, seconds=3, microseconds=400000)


def test_model_datetime_comparisons() -> None:
    t0 = DateTime365(2025, 1, 1, 0, 0, 0, 0, 1)
    t1 = DateTime365(2025, 1, 1, 0, 0, 1, 0, 1)

    assert t0 < t1
    assert t0 <= t1
    assert t1 > t0
    assert t1 >= t0
    assert t0 != t1
    assert t0 == t0


def test_model_datetime_mixed_calendar_arithmetic_and_order_raise() -> None:
    t360 = DateTime360(2025, 1, 1, 0, 0, 0, 0, 1)
    t365 = DateTime365(2025, 1, 1, 0, 0, 0, 0, 1)

    with pytest.raises(TypeError, match="different calendars"):
        _ = t360 - t365

    with pytest.raises(TypeError, match="different calendars"):
        _ = t360 < t365

    assert (t360 == t365) is False


def test_model_datetime_day_of_year_must_match_calendar_date() -> None:
    with pytest.raises(ValueError, match="day_of_year is inconsistent"):
        DateTime365(2025, 1, 2, 0, 0, 0, 0, 1)

    with pytest.raises(ValueError, match="day_of_year is inconsistent"):
        DateTime360(2025, 2, 1, 0, 0, 0, 0, 60)


@pytest.mark.fast_always
def test_clock_iteration_cases(select_fast_cases: SelectFastCases) -> None:
    cases = [
        ClockIterationCase(
            case_id="360-wrap",
            clock=Clock(
                start=datetime(2025, 12, 30, 6, 0, 0),
                dt_seconds=86400.0,
                steps=3,
                year_type="360",
            ),
            expected_times=[
                DateTime360(2025, 12, 30, 6, 0, 0, 0, 360),
                DateTime360(2026, 1, 1, 6, 0, 0, 0, 1),
                DateTime360(2026, 1, 2, 6, 0, 0, 0, 2),
            ],
        ),
        ClockIterationCase(
            case_id="noleap-skip-feb29",
            clock=Clock(
                start=datetime(2024, 2, 28, 0, 0, 0),
                dt_seconds=86400.0,
                steps=3,
                year_type="noleap",
            ),
            expected_times=[
                DateTime365(2024, 2, 28, 0, 0, 0, 0, 59),
                DateTime365(2024, 3, 1, 0, 0, 0, 0, 60),
                DateTime365(2024, 3, 2, 0, 0, 0, 0, 61),
            ],
        ),
        ClockIterationCase(
            case_id="noleap-gregorian-january",
            clock=Clock(
                start=datetime(2025, 1, 30, 12, 0, 0),
                dt_seconds=86400.0,
                steps=2,
                year_type="noleap",
            ),
            expected_times=[
                DateTime365(2025, 1, 30, 12, 0, 0, 0, 30),
                DateTime365(2025, 1, 31, 12, 0, 0, 0, 31),
            ],
        ),
        ClockIterationCase(
            case_id="leap-keeps-feb29",
            clock=Clock(
                start=datetime(2024, 2, 29),
                dt_seconds=86400.0,
                steps=2,
                year_type="leap",
            ),
            expected_times=[
                datetime(2024, 2, 29, 0, 0, 0),
                datetime(2024, 3, 1, 0, 0, 0),
            ],
        ),
    ]

    for case in select_fast_cases(
        cases,
        case_id=lambda case: case.case_id,
        min_cases=2,
    ):
        values = list(case.clock.iter())
        actual_times = [time for _, time, _ in values]
        assert actual_times == case.expected_times


def test_clock_rejects_invalid_year_type() -> None:
    with pytest.raises(ValueError, match="year_type must be one of"):
        Clock(
            start=datetime(2025, 1, 1),
            dt_seconds=3600.0,
            steps=1,
            year_type="gregorian",  # type: ignore[arg-type]
        )


def test_clock_rejects_non_positive_dt_seconds() -> None:
    with pytest.raises(ValueError, match="dt_seconds must be positive"):
        Clock(start=datetime(2025, 1, 1), dt_seconds=0.0, steps=1)

    with pytest.raises(ValueError, match="dt_seconds must be positive"):
        Clock(start=datetime(2025, 1, 1), dt_seconds=-1.0, steps=1)


def test_clock_rejects_negative_steps() -> None:
    with pytest.raises(ValueError, match="steps must be non-negative"):
        Clock(start=datetime(2025, 1, 1), dt_seconds=3600.0, steps=-1)


def test_clock_360_rejects_day_31_start() -> None:
    with pytest.raises(ValueError, match="start day must be between 1 and 30"):
        Clock(
            start=datetime(2025, 1, 31),
            dt_seconds=3600.0,
            steps=1,
            year_type="360",
        )


def test_clock_noleap_rejects_feb_29_start() -> None:
    with pytest.raises(ValueError, match="cannot be February 29"):
        Clock(
            start=datetime(2024, 2, 29),
            dt_seconds=3600.0,
            steps=1,
            year_type="noleap",
        )


def test_clock_360_rolls_microseconds_across_year_boundary() -> None:
    clock = Clock(
        start=datetime(2025, 12, 30, 23, 59, 59, 999999),
        dt_seconds=0.000002,
        steps=2,
        year_type="360",
    )

    values = list(clock.iter())
    actual_times = [time for _, time, _ in values]

    assert actual_times == [
        DateTime360(2025, 12, 30, 23, 59, 59, 999999, 360),
        DateTime360(2026, 1, 1, 0, 0, 0, 1, 1),
    ]


def test_clock_noleap_100_year_daily_run_reaches_year_100() -> None:
    clock = Clock(
        start=datetime(2000, 1, 3, 0, 0, 0),
        dt_seconds=86400.0,
        steps=365 * 100 - 2,
        year_type="noleap",
    )

    values = list(clock.iter())
    _, last_time, _ = values[-1]

    assert isinstance(last_time, DateTime365)
    assert last_time.year - 2000 + 1 == 100
