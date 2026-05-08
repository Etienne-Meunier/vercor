from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import cast

import jax

from vercor.clock import Clock, ModelDateTime
from vercor.dtypes import as_jax_index_array, as_jax_real_array
from vercor.settings import VercorSettings
from vercor.time_selection import (
    datetime_to_seconds_in_year,
    get_periodic_interval,
    is_leap_year,
)
from vercor.types import RuntimeArray


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class RuntimeStepInfo:
    """Precomputed time-selection metadata for one runtime step."""

    monthly_index_left: RuntimeArray
    monthly_index_right: RuntimeArray
    monthly_weight_left: RuntimeArray
    monthly_weight_right: RuntimeArray
    daily_index: RuntimeArray

    @classmethod
    def from_sequences(
        cls,
        monthly_index_left: Sequence[int],
        monthly_index_right: Sequence[int],
        monthly_weight_left: Sequence[float],
        monthly_weight_right: Sequence[float],
        daily_index: Sequence[int],
    ) -> "RuntimeStepInfo":
        """Create scan metadata from host-precomputed index and weight arrays."""

        return cls(
            monthly_index_left=as_jax_index_array(monthly_index_left),
            monthly_index_right=as_jax_index_array(monthly_index_right),
            monthly_weight_left=as_jax_real_array(monthly_weight_left),
            monthly_weight_right=as_jax_real_array(monthly_weight_right),
            daily_index=as_jax_index_array(daily_index),
        )

    def tree_flatten(self) -> tuple[tuple[RuntimeArray, ...], None]:
        return (
            (
                self.monthly_index_left,
                self.monthly_index_right,
                self.monthly_weight_left,
                self.monthly_weight_right,
                self.daily_index,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(
        cls, aux_data: None, children: tuple[RuntimeArray, ...]
    ) -> "RuntimeStepInfo":
        _ = aux_data
        (
            monthly_index_left,
            monthly_index_right,
            monthly_weight_left,
            monthly_weight_right,
            daily_index,
        ) = children
        return cls(
            monthly_index_left=monthly_index_left,
            monthly_index_right=monthly_index_right,
            monthly_weight_left=monthly_weight_left,
            monthly_weight_right=monthly_weight_right,
            daily_index=daily_index,
        )


def runtime_daily_index(time: datetime | ModelDateTime, year_type: str) -> int:
    """Return the no-leap daily forcing index for a runtime timestamp."""

    if year_type == "360":
        month_lengths = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
        month_length = month_lengths[time.month - 1]
        mapped_day = ((time.day - 1) * (month_length - 1)) // 29 + 1
        day_of_year = sum(month_lengths[: time.month - 1]) + mapped_day
    elif year_type == "noleap":
        model_day_of_year = getattr(time, "day_of_year", None)
        if model_day_of_year is None:
            raise ValueError("ModelDateTime.day_of_year is not initialized")
        day_of_year = model_day_of_year
    else:
        day_of_year = time.timetuple().tm_yday
        if is_leap_year(time.year) and day_of_year > 59:
            day_of_year -= 1

    return day_of_year - 1


def runtime_step_info_from_times(
    times: Sequence[datetime | ModelDateTime],
    *,
    year_type: str,
    year_in_seconds: float,
) -> RuntimeStepInfo:
    """Build runtime time-selection metadata for one or more timestamps."""

    monthly_index_left: list[int] = []
    monthly_index_right: list[int] = []
    monthly_weight_left: list[float] = []
    monthly_weight_right: list[float] = []
    daily_index: list[int] = []

    for time in times:
        total_seconds = datetime_to_seconds_in_year(time)
        (n1, f1), (n2, f2) = get_periodic_interval(
            current_time=total_seconds,
            cycle_length=year_in_seconds,
            rec_spacing=year_in_seconds / 12.0,
            n_rec=12,
        )
        monthly_index_left.append(n1)
        monthly_index_right.append(n2)
        monthly_weight_left.append(f1)
        monthly_weight_right.append(f2)
        daily_index.append(runtime_daily_index(time, year_type))

    return RuntimeStepInfo.from_sequences(
        monthly_index_left,
        monthly_index_right,
        monthly_weight_left,
        monthly_weight_right,
        daily_index,
    )


def build_runtime_step_info(clock: Clock, settings: VercorSettings) -> RuntimeStepInfo:
    """Build scanned-runtime time metadata for every clock step."""

    times = [time for _, time, _ in clock.iter()]
    return runtime_step_info_from_times(
        times,
        year_type=clock.year_type,
        year_in_seconds=settings.year_in_seconds,
    )


def initial_runtime_step_info(
    clock: Clock, settings: VercorSettings
) -> RuntimeStepInfo:
    """Return scalar runtime time metadata for the first clock step."""

    clock_iter = clock.iter()
    try:
        _, first_time, _ = next(clock_iter)
    except StopIteration:
        first_time = clock.start
    return scalar_runtime_step_info(first_time, clock, settings)


def scalar_runtime_step_info(
    time: datetime | ModelDateTime,
    clock: Clock,
    settings: VercorSettings,
) -> RuntimeStepInfo:
    """Return scalar runtime time metadata for one clock timestamp."""

    batched_step_info = runtime_step_info_from_times(
        [time],
        year_type=clock.year_type,
        year_in_seconds=settings.year_in_seconds,
    )
    return cast(
        RuntimeStepInfo,
        jax.tree_util.tree_map(lambda value: value[0], batched_step_info),
    )
