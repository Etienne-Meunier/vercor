"""Private backend-neutral period-output schemas and runtime sessions."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

from vercor.calendar import ModelDateTime
from vercor.clock import Clock
from vercor.exceptions import ComponentError, CouplerError
from vercor.jax_logging import LoggerLike
from vercor.output import OutputVariable, PeriodOutput
from vercor.output._dataset import time_coordinate_variable
from vercor.output._period import (
    AccumulatedPeriodVariable,
    PeriodAverageAccumulator,
    _sample_sum_and_counts,
    period_mean_output_variables,
    should_write_period_output,
)
from vercor.output._period_files import write_period_output_netcdf
from vercor.pytree import PyTreeNodeMixin

if TYPE_CHECKING:
    from vercor.components.base import Component
    from vercor.state import RunState
    from vercor._runtime.state import ComponentRuntimeState

_Time = datetime | ModelDateTime
_SampleExtractor = Callable[["ComponentRuntimeState"], Mapping[str, OutputVariable]]
_CoordinateBuilder = Callable[
    [_Time, Mapping[str, OutputVariable]], Mapping[str, OutputVariable]
]
_DataDecorator = Callable[[Mapping[str, OutputVariable]], Mapping[str, OutputVariable]]
_FilenamePolicy = Callable[[_Time], str]


@dataclass(frozen=True)
class _PeriodOutputSchema:
    """Static extraction and file-decoration policy for one component."""

    component_name: str
    period: PeriodOutput
    variable_names: tuple[str, ...]
    variable_dims: tuple[tuple[str, ...], ...]
    sample: _SampleExtractor
    build_coordinate_variables: _CoordinateBuilder
    decorate_data_variables: _DataDecorator
    filename: _FilenamePolicy
    summation_dim: str | None = None
    time_dim: str = "time"
    dimension_order: tuple[str, ...] | None = None
    empty_error_message: str = "Period output requires at least one sample."
    writer: Callable[..., None] = write_period_output_netcdf
    take_initial_accumulated_variables: (
        Callable[[], Mapping[str, AccumulatedPeriodVariable]] | None
    ) = None
    sample_accumulator: (
        Callable[["ComponentRuntimeState"], "_PeriodOutputAccumulator | None"] | None
    ) = None


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class _PeriodOutputAccumulator(PyTreeNodeMixin):
    """Immutable JAX PyTree of running sums and finite-value counts."""

    pytree_children = ("sum_values", "counts")
    pytree_aux_data = ("names", "dims", "attrs")

    names: tuple[str, ...]
    dims: tuple[tuple[str, ...], ...]
    attrs: tuple[tuple[tuple[str, Any], ...], ...]
    sum_values: tuple[jax.Array, ...]
    counts: tuple[jax.Array, ...]

    @classmethod
    def zeros_from_samples(
        cls,
        samples: Mapping[str, OutputVariable],
        *,
        summation_dim: str | None,
    ) -> "_PeriodOutputAccumulator":
        """Create a shape-stable empty accumulator from representative samples."""

        if not samples:
            raise ValueError("Period output requires at least one sampled variable.")
        names: list[str] = []
        dims: list[tuple[str, ...]] = []
        attrs: list[tuple[tuple[str, Any], ...]] = []
        sums: list[jax.Array] = []
        counts: list[jax.Array] = []
        for name, sample in samples.items():
            sample_dims, sum_values, sample_counts = _sample_sum_and_counts(
                name,
                sample,
                summation_dim=summation_dim,
            )
            names.append(name)
            dims.append(sample_dims)
            attrs.append(tuple(sample.attrs.items()))
            sums.append(jnp.zeros_like(sum_values))
            counts.append(jnp.zeros_like(sample_counts))
        return cls(
            names=tuple(names),
            dims=tuple(dims),
            attrs=tuple(attrs),
            sum_values=tuple(sums),
            counts=tuple(counts),
        )

    @classmethod
    def from_accumulated_variables(
        cls,
        variables: Mapping[str, AccumulatedPeriodVariable],
    ) -> "_PeriodOutputAccumulator":
        """Copy an existing host-owned window into immutable JAX storage."""

        return cls(
            names=tuple(variables),
            dims=tuple(variable.dims for variable in variables.values()),
            attrs=tuple(
                tuple(variable.attrs.items()) for variable in variables.values()
            ),
            sum_values=tuple(variable.sum_values for variable in variables.values()),
            counts=tuple(variable.counts for variable in variables.values()),
        )

    def add_samples(
        self,
        samples: Mapping[str, OutputVariable],
        *,
        summation_dim: str | None,
    ) -> "_PeriodOutputAccumulator":
        """Return a new accumulator containing one additional sample mapping."""

        if tuple(samples) != self.names:
            raise ValueError("Period output variables changed across samples.")
        sums: list[jax.Array] = []
        counts: list[jax.Array] = []
        for index, (name, sample) in enumerate(samples.items()):
            dims, sample_sum, sample_counts = _sample_sum_and_counts(
                name,
                sample,
                summation_dim=summation_dim,
            )
            if dims != self.dims[index]:
                raise ValueError(f"Period output variable {name!r} dimensions changed.")
            if sample_sum.shape != self.sum_values[index].shape:
                raise ValueError(f"Period output variable {name!r} shape changed.")
            sums.append(self.sum_values[index] + sample_sum)
            counts.append(self.counts[index] + sample_counts)
        return _PeriodOutputAccumulator(
            names=self.names,
            dims=self.dims,
            attrs=self.attrs,
            sum_values=tuple(sums),
            counts=tuple(counts),
        )

    def merge(
        self,
        other: "_PeriodOutputAccumulator",
    ) -> "_PeriodOutputAccumulator":
        """Return the exact sum/count merge of two compatible accumulators."""

        if self.names != other.names:
            raise ValueError("Period output variables changed across accumulators.")
        if self.dims != other.dims:
            raise ValueError("Period output dimensions changed across accumulators.")
        if tuple(value.shape for value in self.sum_values) != tuple(
            value.shape for value in other.sum_values
        ):
            raise ValueError("Period output shapes changed across accumulators.")
        return _PeriodOutputAccumulator(
            names=self.names,
            dims=self.dims,
            attrs=self.attrs,
            sum_values=tuple(
                left + right
                for left, right in zip(self.sum_values, other.sum_values, strict=True)
            ),
            counts=tuple(
                left + right
                for left, right in zip(self.counts, other.counts, strict=True)
            ),
        )

    def reset(self) -> "_PeriodOutputAccumulator":
        """Return an empty accumulator with the same static variable schema."""

        return _PeriodOutputAccumulator(
            names=self.names,
            dims=self.dims,
            attrs=self.attrs,
            sum_values=tuple(jnp.zeros_like(value) for value in self.sum_values),
            counts=tuple(jnp.zeros_like(value) for value in self.counts),
        )

    def mean_samples(self) -> dict[str, OutputVariable]:
        """Return reduced variables using the shared finite-count semantics."""

        return {
            name: AccumulatedPeriodVariable(
                dims=self.dims[index],
                sum_values=self.sum_values[index],
                counts=self.counts[index],
                attrs=dict(self.attrs[index]),
            ).mean_sample()
            for index, name in enumerate(self.names)
        }


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class _PeriodOutputSession(PyTreeNodeMixin):
    """Immutable per-run accumulator bundle carried across scan chunks."""

    pytree_children = ("accumulators",)

    accumulators: tuple[_PeriodOutputAccumulator, ...]

    def accumulate(
        self,
        schemas: Sequence[_PeriodOutputSchema],
        state: "RunState",
    ) -> "_PeriodOutputSession":
        """Sample every configured component from one post-step runtime state."""

        accumulated = []
        for schema, accumulator in zip(schemas, self.accumulators, strict=True):
            component_state = state._component_state(schema.component_name)
            sampled_accumulator = (
                schema.sample_accumulator(component_state)
                if schema.sample_accumulator is not None
                else None
            )
            if sampled_accumulator is not None:
                accumulated.append(accumulator.merge(sampled_accumulator))
            else:
                accumulated.append(
                    accumulator.add_samples(
                        schema.sample(component_state),
                        summation_dim=schema.summation_dim,
                    )
                )
        return _PeriodOutputSession(accumulators=tuple(accumulated))


@dataclass(frozen=True)
class _PeriodOutputBoundary:
    """One ordered scan stop and the component outputs due at that stop."""

    stop_step: int
    time: _Time
    due_schema_indices: tuple[int, ...]
    output_filenames: tuple[str, ...]


@dataclass(frozen=True)
class _PeriodOutputPlan:
    """Static schemas, initial accumulators, and coalesced clock boundaries."""

    schemas: tuple[_PeriodOutputSchema, ...]
    initial_session: _PeriodOutputSession
    boundaries: tuple[_PeriodOutputBoundary, ...]


def has_period_output(components: Mapping[str, "Component"]) -> bool:
    """Return whether any configured component requests period output."""

    return any(
        component.spec.output.period is not None for component in components.values()
    )


def validate_period_output_run_state_not_traced(state: "RunState") -> None:
    """Reject differentiated period-output runs before schema extraction."""

    if any(
        isinstance(leaf, jax.core.Tracer) for leaf in jax.tree_util.tree_leaves(state)
    ):
        raise CouplerError(
            "Period output is an I/O workflow and cannot run with traced "
            "RunState leaves. Differentiated or outer-jitted runs must "
            "disable component period output."
        )


def validate_period_output_execution(execution: object) -> None:
    """Reject custom backends until they can carry an output session."""

    if not isinstance(execution, str):
        raise CouplerError(
            "Custom execution backends do not support period output because "
            "the public backend contract has no period-session hook. Configure "
            "a built-in host/auto/JAX backend or disable component period output."
        )


def validate_period_output_component_state(
    component: "Component",
    state: "ComponentRuntimeState",
) -> None:
    """Validate generic selected fields against initialized runtime state."""

    period = component.spec.output.period
    if (
        period is None
        or hasattr(component, "_period_output_schema_factory")
        or _period_output_handled_by_step(component)
    ):
        return
    selected = tuple(period.variables) or tuple(component.spec.outputs)
    if not selected:
        raise ComponentError(
            f"Period output for component {component.name!r} has no variables. "
            "Select PeriodOutput.variables or declare component outputs."
        )
    missing = tuple(name for name in selected if name not in state.fields)
    if missing:
        available = ", ".join(state.fields.field_names) or "<none>"
        raise ComponentError(
            f"Period output for component {component.name!r} selected unknown "
            f"runtime field {missing[0]!r}. Available state fields: {available}."
        )


def build_period_output_plan(
    components: Mapping[str, "Component"],
    state: "RunState",
    clock: Clock,
) -> _PeriodOutputPlan:
    """Build static schemas and ordered chunk boundaries before execution."""

    schemas: list[_PeriodOutputSchema] = []
    accumulators: list[_PeriodOutputAccumulator] = []
    for name, component in components.items():
        period = component.spec.output.period
        if period is None or _period_output_handled_by_step(component):
            continue
        component_state = state._component_state(name)
        validate_period_output_component_state(component, component_state)
        factory = getattr(component, "_period_output_schema_factory", None)
        schema = (
            factory(component, component_state)
            if callable(factory)
            else _generic_period_output_schema(component, component_state, period)
        )
        schemas.append(schema)
        initial_variables = (
            schema.take_initial_accumulated_variables()
            if schema.take_initial_accumulated_variables is not None
            else {}
        )
        if initial_variables:
            accumulators.append(
                _PeriodOutputAccumulator.from_accumulated_variables(initial_variables)
            )
        else:
            sampled_accumulator = (
                schema.sample_accumulator(component_state)
                if schema.sample_accumulator is not None
                else None
            )
            accumulators.append(
                sampled_accumulator.reset()
                if sampled_accumulator is not None
                else _PeriodOutputAccumulator.zeros_from_samples(
                    schema.sample(component_state),
                    summation_dim=schema.summation_dim,
                )
            )

    boundaries = _period_output_boundaries(tuple(schemas), clock)
    return _PeriodOutputPlan(
        schemas=tuple(schemas),
        initial_session=_PeriodOutputSession(tuple(accumulators)),
        boundaries=boundaries,
    )


def write_period_output_boundary(
    plan: _PeriodOutputPlan,
    session: _PeriodOutputSession,
    boundary: _PeriodOutputBoundary,
    *,
    logger: LoggerLike | None,
) -> _PeriodOutputSession:
    """Write completed reductions and reset only schemas due at this boundary."""

    accumulators = list(session.accumulators)
    for index, output_filename in zip(
        boundary.due_schema_indices,
        boundary.output_filenames,
        strict=True,
    ):
        schema = plan.schemas[index]
        accumulator = accumulators[index]
        mean_accumulator = PeriodAverageAccumulator()
        mean_accumulator.add_samples(accumulator.mean_samples())
        mean_variables = period_mean_output_variables(
            mean_accumulator,
            empty_error_message=schema.empty_error_message,
            time_dim=schema.time_dim,
            dimension_order=schema.dimension_order,
        )
        schema.writer(
            output_filename,
            mean_variables=mean_variables,
            build_coordinate_variables=lambda variables, schema=schema: (
                schema.build_coordinate_variables(boundary.time, variables)
            ),
            build_data_variables=schema.decorate_data_variables,
            logger=logger,
        )
        accumulators[index] = accumulator.reset()
    return _PeriodOutputSession(tuple(accumulators))


def _generic_period_output_schema(
    component: "Component",
    state: "ComponentRuntimeState",
    period: PeriodOutput,
) -> _PeriodOutputSchema:
    selected = tuple(period.variables) or tuple(component.spec.outputs)
    dims = {
        name: _generic_field_dims(
            name,
            tuple(state.fields.get(name).shape),
            component.grid.shape,
        )
        for name in selected
    }

    def sample(component_state: "ComponentRuntimeState") -> dict[str, OutputVariable]:
        return {
            name: OutputVariable(
                dims[name],
                component_state.fields.get(name),
                {"component": component.name, "field_name": name},
            )
            for name in selected
        }

    def coordinates(
        time: _Time,
        variables: Mapping[str, OutputVariable],
    ) -> dict[str, OutputVariable]:
        _ = variables
        return {
            "time": time_coordinate_variable(time),
            "latitude": OutputVariable(("nlat",), component.grid.latitude),
            "longitude": OutputVariable(("nlon",), component.grid.longitude),
        }

    return _PeriodOutputSchema(
        component_name=component.name,
        period=period,
        variable_names=selected,
        variable_dims=tuple(dims[name] for name in selected),
        sample=sample,
        build_coordinate_variables=coordinates,
        decorate_data_variables=dict,
        filename=lambda time: (
            f"{component.name}.averages.{time.strftime('%Y-%m-%d')}.nc"
        ),
    )


def _period_output_handled_by_step(component: "Component") -> bool:
    """Return whether a bundled host adapter owns its native period writes."""

    return bool(
        component.spec.output.period is not None
        and getattr(component, "_period_output_handled_by_step", False)
    )


def _generic_field_dims(
    variable_name: str,
    shape: tuple[int, ...],
    grid_shape: tuple[int, int],
) -> tuple[str, ...]:
    if len(shape) >= 2 and shape[-2:] == grid_shape:
        prefix = tuple(
            f"{variable_name}_dim_{index}" for index in range(len(shape) - 2)
        )
        return (*prefix, "nlat", "nlon")
    return tuple(f"{variable_name}_dim_{index}" for index in range(len(shape)))


def _period_output_boundaries(
    schemas: tuple[_PeriodOutputSchema, ...],
    clock: Clock,
) -> tuple[_PeriodOutputBoundary, ...]:
    raw_boundaries: list[tuple[int, _Time, tuple[int, ...]]] = []
    last_stop = 0
    for step, time, dt in clock.iter():
        due = tuple(
            index
            for index, schema in enumerate(schemas)
            if should_write_period_output(schema.period, time=time, dt=dt)
        )
        if due:
            last_stop = step + 1
            raw_boundaries.append((last_stop, time, due))
    if last_stop < clock.steps:
        final_time: _Time = clock.start
        for _, final_time, _ in clock.iter():
            pass
        raw_boundaries.append((clock.steps, final_time, ()))

    base_filenames = tuple(
        tuple(schemas[index].filename(time) for index in due)
        for _, time, due in raw_boundaries
    )
    path_counts = Counter(
        filename for filenames in base_filenames for filename in filenames
    )
    schema_path_counts = Counter(
        (index, filename)
        for (_, _, due), filenames in zip(
            raw_boundaries,
            base_filenames,
            strict=True,
        )
        for index, filename in zip(due, filenames, strict=True)
    )
    schema_indices_by_path: dict[str, set[int]] = {}
    for (_, _, due), filenames in zip(
        raw_boundaries,
        base_filenames,
        strict=True,
    ):
        for index, filename in zip(due, filenames, strict=True):
            schema_indices_by_path.setdefault(filename, set()).add(index)

    reserved_unique_paths = {
        filename for filename, count in path_counts.items() if count == 1
    }
    allocated_paths = set(reserved_unique_paths)
    allocated_filenames: list[tuple[str, ...]] = []
    request_index = 0
    for (stop_step, time, due), filenames in zip(
        raw_boundaries,
        base_filenames,
        strict=True,
    ):
        boundary_filenames = []
        for index, filename in zip(due, filenames, strict=True):
            if path_counts[filename] == 1:
                allocated = filename
            else:
                allocated = filename
                if schema_path_counts[(index, filename)] > 1:
                    allocated = _disambiguated_period_filename(
                        allocated,
                        time=time,
                        step=stop_step - 1,
                    )
                if len(schema_indices_by_path[filename]) > 1:
                    allocated = _schema_disambiguated_period_filename(
                        allocated,
                        component_name=schemas[index].component_name,
                        schema_index=index,
                    )
                collision_index = 0
                candidate = allocated
                while candidate in allocated_paths:
                    candidate = _record_disambiguated_period_filename(
                        allocated,
                        request_index=request_index,
                        collision_index=collision_index,
                    )
                    collision_index += 1
                allocated = candidate
                allocated_paths.add(allocated)
            boundary_filenames.append(allocated)
            request_index += 1
        allocated_filenames.append(tuple(boundary_filenames))

    return tuple(
        _PeriodOutputBoundary(
            stop_step=stop_step,
            time=time,
            due_schema_indices=due,
            output_filenames=filenames,
        )
        for (stop_step, time, due), filenames in zip(
            raw_boundaries,
            allocated_filenames,
            strict=True,
        )
    )


def _disambiguated_period_filename(
    filename: str,
    *,
    time: _Time,
    step: int,
) -> str:
    """Return a deterministic sub-daily filename for a colliding date path."""

    stem = filename[:-3] if filename.endswith(".nc") else filename
    timestamp = (
        f"{time.hour:02d}{time.minute:02d}{time.second:02d}." f"{time.microsecond:06d}"
    )
    return f"{stem}T{timestamp}.step{step:08d}.nc"


def _schema_disambiguated_period_filename(
    filename: str,
    *,
    component_name: str,
    schema_index: int,
) -> str:
    """Add a path-safe component/schema discriminator to a filename."""

    stem = filename[:-3] if filename.endswith(".nc") else filename
    component_token = _sanitize_period_filename_token(component_name)
    return f"{stem}.component-{component_token}.schema{schema_index:04d}.nc"


def _record_disambiguated_period_filename(
    filename: str,
    *,
    request_index: int,
    collision_index: int,
) -> str:
    """Resolve a generated-name collision with a stable record discriminator."""

    stem = filename[:-3] if filename.endswith(".nc") else filename
    return f"{stem}.record{request_index:08d}.collision{collision_index:04d}.nc"


def _sanitize_period_filename_token(value: str) -> str:
    """Return an ASCII filename token without path separators."""

    characters: list[str] = []
    replacing = False
    for character in value:
        if character.isascii() and (character.isalnum() or character in "-_"):
            characters.append(character)
            replacing = False
        elif not replacing:
            characters.append("-")
            replacing = True
    return "".join(characters).strip("-_") or "component"


__all__ = []
