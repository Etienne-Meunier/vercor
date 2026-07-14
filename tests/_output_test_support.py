"""Test harness adapting legacy extraction tests to the v4 accumulator."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta
from types import MappingProxyType
from typing import Any

from vercor.calendar import ModelDateTime
from vercor.output import OutputFrame, OutputVariable, PeriodOutput
from vercor.output._period import (
    period_mean_sample_to_output_variable,
    should_write_period_output,
)
from vercor.output._netcdf import write_netcdf_dataset
from vercor.output._session import _OutputAccumulator


class ComponentOutputAdapter:
    """Exercise native extraction helpers through the v4 immutable accumulator."""

    def __init__(
        self,
        *,
        empty_error_message: str,
        time_dim: str = "time",
        value_dims_for_sample: Callable[[OutputVariable], Sequence[str]] | None = None,
        dimension_order: Sequence[str] | None = None,
    ) -> None:
        self._value: _OutputAccumulator | None = None
        self._snapshot: _OutputAccumulator | None = None
        self._snapshot_time: datetime | ModelDateTime | None = None
        self._empty_error_message = empty_error_message
        self._time_dim = time_dim
        self._value_dims_for_sample = value_dims_for_sample
        self._dimension_order = (
            None if dimension_order is None else tuple(dimension_order)
        )

    @property
    def accumulator(self) -> "ComponentOutputAdapter":
        return self

    @property
    def empty(self) -> bool:
        return self._value is None

    @property
    def variables(self) -> Mapping[str, Any]:
        if self._value is None:
            return MappingProxyType({})
        return MappingProxyType(
            {
                name: _AccumulatedView(self._value, index)
                for index, name in enumerate(self._value.names)
            }
        )

    @property
    def snapshot_empty(self) -> bool:
        return self._snapshot is None

    @property
    def snapshot_time(self) -> datetime | ModelDateTime | None:
        return self._snapshot_time

    @property
    def snapshot_variables(self) -> Mapping[str, OutputVariable]:
        return MappingProxyType(
            {}
            if self._snapshot is None
            else dict(self._snapshot.mean_frame().variables)
        )

    def reset(self) -> None:
        self._value = None
        self._snapshot = None
        self._snapshot_time = None

    def clear(self) -> None:
        self._value = None

    def accumulate(
        self,
        variables: Mapping[str, OutputVariable],
        *,
        summation_dim: str | None = None,
    ) -> None:
        frame = OutputFrame(variables, sample_dimension=summation_dim)
        current = (
            _OutputAccumulator.zeros_from_frame(frame)
            if self._value is None
            else self._value
        )
        self._value = current.add_frame(frame)

    def add_samples(
        self,
        samples: Mapping[str, OutputVariable],
        *,
        summation_dim: str | None = None,
    ) -> None:
        self.accumulate(samples, summation_dim=summation_dim)

    def mean_samples(self) -> dict[str, OutputVariable]:
        if self._value is None:
            raise ValueError(self._empty_error_message)
        return dict(self._value.mean_frame().variables)

    def record_snapshot(
        self,
        variables: Mapping[str, OutputVariable],
        *,
        summation_dim: str | None = None,
        time: datetime | ModelDateTime | None = None,
    ) -> None:
        frame = OutputFrame(variables, sample_dimension=summation_dim)
        self._snapshot = _OutputAccumulator.zeros_from_frame(frame).add_frame(frame)
        self._snapshot_time = time

    def write_snapshot(
        self,
        output: str,
        *,
        build_coordinate_variables: Callable[
            [Mapping[str, OutputVariable]], Mapping[str, OutputVariable]
        ],
        build_data_variables: (
            Callable[[Mapping[str, OutputVariable]], Mapping[str, OutputVariable]]
            | None
        ) = None,
        logger: Any | None = None,
    ) -> None:
        if self._snapshot is None:
            raise ValueError(self._empty_error_message)
        self._write(
            self._snapshot,
            output,
            build_coordinate_variables=build_coordinate_variables,
            build_data_variables=build_data_variables,
            logger=logger,
        )

    def write_period_average(
        self,
        output: str,
        *,
        build_coordinate_variables: Callable[
            [Mapping[str, OutputVariable]], Mapping[str, OutputVariable]
        ],
        build_data_variables: (
            Callable[[Mapping[str, OutputVariable]], Mapping[str, OutputVariable]]
            | None
        ) = None,
        logger: Any | None = None,
    ) -> None:
        if self._value is None:
            raise ValueError(self._empty_error_message)
        self._write(
            self._value,
            output,
            build_coordinate_variables=build_coordinate_variables,
            build_data_variables=build_data_variables,
            logger=logger,
        )
        self._value = None

    def write_period_average_if_due(
        self,
        *,
        time: datetime | ModelDateTime,
        dt: timedelta,
        output_frequency: str | None,
        output: str | Callable[[Any], str],
        build_coordinate_variables: Callable[
            [Mapping[str, OutputVariable]], Mapping[str, OutputVariable]
        ],
        build_data_variables: (
            Callable[[Mapping[str, OutputVariable]], Mapping[str, OutputVariable]]
            | None
        ) = None,
        logger: Any | None = None,
    ) -> bool:
        if output_frequency is None:
            return False
        period = PeriodOutput(frequency=output_frequency)  # type: ignore[arg-type]
        if not should_write_period_output(period, time=time, dt=dt):
            return False
        path = output(time) if callable(output) else output
        self.write_period_average(
            path,
            build_coordinate_variables=build_coordinate_variables,
            build_data_variables=build_data_variables,
            logger=logger,
        )
        return True

    def record_period_average_if_due(
        self,
        variables: Mapping[str, OutputVariable],
        *,
        summation_dim: str | None = None,
        **kwargs: Any,
    ) -> bool:
        self.accumulate(variables, summation_dim=summation_dim)
        return self.write_period_average_if_due(**kwargs)

    def _write(
        self,
        accumulator: _OutputAccumulator,
        output: str,
        *,
        build_coordinate_variables: Callable[
            [Mapping[str, OutputVariable]], Mapping[str, OutputVariable]
        ],
        build_data_variables: (
            Callable[[Mapping[str, OutputVariable]], Mapping[str, OutputVariable]]
            | None
        ),
        logger: Any | None,
    ) -> None:
        means = accumulator.mean_frame().variables
        variables = {
            name: period_mean_sample_to_output_variable(
                sample,
                time_dim=self._time_dim,
                value_dims=(
                    tuple(self._value_dims_for_sample(sample))
                    if self._value_dims_for_sample is not None
                    else None
                ),
                dimension_order=self._dimension_order,
            )
            for name, sample in means.items()
        }
        data_variables = (
            variables
            if build_data_variables is None
            else dict(build_data_variables(variables))
        )
        write_netcdf_dataset(
            output=output,
            coordinate_variables=build_coordinate_variables(data_variables),
            data_variables=data_variables,
            logger=logger,
        )


class _AccumulatedView:
    def __init__(self, accumulator: _OutputAccumulator, index: int) -> None:
        self.dims = accumulator.dims[index]
        self.attrs = dict(accumulator.attrs[index])
        self.sum_values = accumulator.sum_values[index]
        self.counts = accumulator.counts[index]
