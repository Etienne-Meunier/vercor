from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime
from math import isfinite
from numbers import Integral, Real
from typing import Any, cast

import jax
from jax.errors import JaxRuntimeError

from vercor.calendar import ModelDateTime
from vercor.clock import Clock
from vercor.dtypes import as_jax_index_array
from vercor.exceptions import CouplerError
from vercor.jax_logging import LoggerLike
from vercor.runtime import ExecutionBackend, ExecutionContext
from vercor.settings import Settings
from vercor.state import RunState
from vercor.types import RuntimeArray
from vercor._runtime.dispatch_context import RuntimeDispatchContext
from vercor._runtime.driver import step_runtime_component
from vercor._runtime.interrupts import RuntimeInterruptController
from vercor._runtime.progress import (
    log_scanned_component_progress,
    log_scanned_step_progress,
    runtime_component_progress_message,
    runtime_step_progress_message,
    runtime_step_progress_messages,
)
from vercor._runtime.run_context import RuntimeRunContext
from vercor._runtime.time import build_runtime_step_info, scalar_runtime_step_info
from vercor.output._session import (
    _PeriodOutputPlan,
    _PeriodOutputSession,
    write_period_output_boundary,
)


def run_compiled_scanned_runtime(
    runtime_state: RunState,
    *,
    context: RuntimeRunContext,
) -> RunState:
    """Run a pure runtime state through a one-shot compiled scanned path."""

    try:

        def scanned_runtime(
            state: RunState,
        ) -> RunState:
            return run_scanned_runtime(
                state,
                run_order=context.run_order,
                clock=context.clock,
                settings=context.dispatch_context.settings,
                model_year_seconds=context.options.model_year_seconds,
                logger=context.logger,
                dispatch_context=context.dispatch_context,
                interrupts=context.interrupts,
            )

        compiled_runtime = cast(
            Callable[[RunState], RunState],
            jax.jit(scanned_runtime),
        )
        return compiled_runtime(runtime_state)
    except JaxRuntimeError as error:
        context.interrupts.raise_if_jax_callback_interrupted(
            error,
            "compiled scanned runtime",
        )


def run_host_runtime(
    runtime_state: RunState,
    *,
    run_order: Sequence[str],
    clock: Clock,
    settings: Settings,
    model_year_seconds: float,
    logger: LoggerLike,
    dispatch_context: RuntimeDispatchContext,
    interrupts: RuntimeInterruptController,
) -> RunState:
    """Run the host-enabled runtime path for non-differentiable adapters."""

    for n, time, dt in clock.iter():
        interrupts.checkpoint("host runtime step")
        logger.info(runtime_step_progress_message(n, time, dt))
        step_info = scalar_runtime_step_info(
            time,
            clock,
            settings,
            model_year_seconds=model_year_seconds,
        )

        for component_name in run_order:
            interrupts.checkpoint(f"host runtime component {component_name}")
            logger.info(runtime_component_progress_message(component_name))
            runtime_state = step_runtime_component(
                runtime_state,
                component_name,
                step_info,
                dispatch_context=dispatch_context,
                allow_host_runtime=True,
                time=time,
                logger=logger,
                step=n,
            )
            interrupts.checkpoint(f"host runtime component {component_name}")
        interrupts.checkpoint("host runtime step")

    return runtime_state


def run_host_period_output_runtime(
    runtime_state: RunState,
    *,
    context: RuntimeRunContext,
    output_plan: _PeriodOutputPlan,
) -> RunState:
    """Run host steps through the shared immutable period-output session."""

    session = output_plan.initial_session
    boundaries = {boundary.stop_step: boundary for boundary in output_plan.boundaries}
    for n, time, dt in context.clock.iter():
        context.interrupts.checkpoint("host runtime step")
        context.logger.info(runtime_step_progress_message(n, time, dt))
        step_info = scalar_runtime_step_info(
            time,
            context.clock,
            context.dispatch_context.settings,
            model_year_seconds=context.options.model_year_seconds,
        )
        for component_name in context.run_order:
            context.interrupts.checkpoint(f"host runtime component {component_name}")
            context.logger.info(runtime_component_progress_message(component_name))
            runtime_state = step_runtime_component(
                runtime_state,
                component_name,
                step_info,
                dispatch_context=context.dispatch_context,
                allow_host_runtime=True,
                time=time,
                logger=context.logger,
                step=n,
            )
            context.interrupts.checkpoint(f"host runtime component {component_name}")
        session = session.accumulate(output_plan.schemas, runtime_state)
        boundary = boundaries.get(n + 1)
        if boundary is not None:
            session = write_period_output_boundary(
                output_plan,
                session,
                boundary,
                logger=context.logger,
            )
        context.interrupts.checkpoint("host runtime step")
    return runtime_state


def run_compiled_period_output_runtime(
    runtime_state: RunState,
    *,
    context: RuntimeRunContext,
    output_plan: _PeriodOutputPlan,
) -> RunState:
    """Run compiled scan chunks and write only completed reductions on host."""

    step_infos = build_runtime_step_info(
        context.clock,
        context.dispatch_context.settings,
        model_year_seconds=context.options.model_year_seconds,
    )
    step_indices = as_jax_index_array(range(context.clock.steps))
    step_progress_messages = runtime_step_progress_messages(context.clock)

    @jax.jit
    def compiled_chunk(
        state: RunState,
        session: _PeriodOutputSession,
        chunk_indices: RuntimeArray,
        chunk_infos: Any,
    ) -> tuple[RunState, _PeriodOutputSession]:
        return _run_period_output_scanned_chunk(
            state,
            session,
            chunk_indices,
            chunk_infos,
            context=context,
            output_plan=output_plan,
            step_progress_messages=step_progress_messages,
        )

    session = output_plan.initial_session
    start = 0
    try:
        for boundary in output_plan.boundaries:
            stop = boundary.stop_step
            chunk_infos = jax.tree_util.tree_map(
                lambda value: value[start:stop],
                step_infos,
            )
            runtime_state, session = compiled_chunk(
                runtime_state,
                session,
                step_indices[start:stop],
                chunk_infos,
            )
            session = write_period_output_boundary(
                output_plan,
                session,
                boundary,
                logger=context.logger,
            )
            start = stop
    except JaxRuntimeError as error:
        context.interrupts.raise_if_jax_callback_interrupted(
            error,
            "compiled period-output runtime",
        )
    return runtime_state


def _run_period_output_scanned_chunk(
    runtime_state: RunState,
    session: _PeriodOutputSession,
    step_indices: RuntimeArray,
    step_infos: Any,
    *,
    context: RuntimeRunContext,
    output_plan: _PeriodOutputPlan,
    step_progress_messages: Sequence[str],
) -> tuple[RunState, _PeriodOutputSession]:
    """Scan one chunk without period-output callbacks or file I/O."""

    def step_all_components(
        carry: tuple[RunState, _PeriodOutputSession],
        scan_input: tuple[RuntimeArray, Any],
    ) -> tuple[tuple[RunState, _PeriodOutputSession], None]:
        state, output_session = carry
        step_index, step_info = scan_input
        context.interrupts.scanned_checkpoint("scanned runtime step", step_index)
        log_scanned_step_progress(
            context.logger,
            step_index,
            step_progress_messages,
        )
        for component_name in context.run_order:
            context.interrupts.scanned_checkpoint(
                f"scanned runtime component {component_name}",
                step_index,
            )
            log_scanned_component_progress(context.logger, component_name)
            state = step_runtime_component(
                state,
                component_name,
                step_info,
                dispatch_context=context.dispatch_context,
                allow_host_runtime=False,
                logger=context.logger,
                step=step_index,
            )
            context.interrupts.scanned_checkpoint(
                f"scanned runtime component {component_name}",
                step_index,
            )
        output_session = output_session.accumulate(output_plan.schemas, state)
        context.interrupts.scanned_checkpoint("scanned runtime step", step_index)
        return (state, output_session), None

    (final_state, final_session), _ = jax.lax.scan(
        step_all_components,
        (runtime_state, session),
        (step_indices, step_infos),
        length=step_indices.shape[0],
    )
    return final_state, final_session


def run_scanned_runtime(
    runtime_state: RunState,
    *,
    run_order: Sequence[str],
    clock: Clock,
    settings: Settings,
    model_year_seconds: float,
    logger: LoggerLike,
    dispatch_context: RuntimeDispatchContext,
    interrupts: RuntimeInterruptController,
) -> RunState:
    """Run the unified runtime path under ``jax.lax.scan`` and return state."""

    step_infos = build_runtime_step_info(
        clock,
        settings,
        model_year_seconds=model_year_seconds,
    )
    step_indices = as_jax_index_array(range(clock.steps))
    step_progress_messages = runtime_step_progress_messages(clock)

    def step_all_components(
        state: RunState,
        scan_input: tuple[RuntimeArray, Any],
    ) -> tuple[RunState, None]:
        step_index, step_info = scan_input
        interrupts.scanned_checkpoint(
            "scanned runtime step",
            step_index,
        )
        log_scanned_step_progress(logger, step_index, step_progress_messages)
        for component_name in run_order:
            interrupts.scanned_checkpoint(
                f"scanned runtime component {component_name}",
                step_index,
            )
            log_scanned_component_progress(logger, component_name)
            state = step_runtime_component(
                state,
                component_name,
                step_info,
                dispatch_context=dispatch_context,
                allow_host_runtime=False,
                logger=logger,
                step=step_index,
            )
            interrupts.scanned_checkpoint(
                f"scanned runtime component {component_name}",
                step_index,
            )
        interrupts.scanned_checkpoint(
            "scanned runtime step",
            step_index,
        )
        return state, None

    try:
        final_state, _ = jax.lax.scan(
            step_all_components,
            runtime_state,
            (step_indices, step_infos),
            length=clock.steps,
        )
    except JaxRuntimeError as error:
        interrupts.raise_if_jax_callback_interrupted(
            error,
            "scanned runtime",
        )
    return final_state


def run_custom_backend(
    backend: ExecutionBackend,
    state: RunState,
    *,
    context: RuntimeRunContext,
) -> RunState:
    """Delegate execution to a custom backend and validate its public result."""

    public_context = ExecutionContext(
        clock=context.clock,
        run_order=tuple(context.run_order),
        options=context.options,
        logger=context.logger,
    )
    context.interrupts.checkpoint("custom runtime")
    result = backend.run(
        state,
        context=public_context,
        driver=_RuntimeDriverAdapter(context),
    )
    context.interrupts.checkpoint("custom runtime")
    if not isinstance(result, RunState):
        backend_name = backend.__class__.__qualname__
        actual_type = type(result).__name__
        raise CouplerError(
            f"Execution backend {backend_name}.run(...) must return RunState; "
            f"got {actual_type}."
        )
    return result


class _RuntimeDriverAdapter:
    """Public driver implementation backed by VerCOR's private dispatch."""

    def __init__(self, context: RuntimeRunContext) -> None:
        self._context = context

    def step_component(
        self,
        state: RunState,
        component: str,
        *,
        step: int | RuntimeArray,
    ) -> RunState:
        """Advance one validated component at the requested clock step."""

        self._validate_state(state)
        self._validate_component(component)
        step_index = _validated_step_index(step)
        _validate_step_range(step_index, self._context.clock)
        step_time = _clock_time_at_step(self._context, step_index)
        step_info = scalar_runtime_step_info(
            step_time,
            self._context.clock,
            self._context.dispatch_context.settings,
            model_year_seconds=self._context.options.model_year_seconds,
        )
        label = f"custom runtime component {component}"
        self._context.interrupts.checkpoint(label)
        result = step_runtime_component(
            state,
            component,
            step_info,
            dispatch_context=self._context.dispatch_context,
            allow_host_runtime=True,
            time=step_time,
            logger=self._context.logger,
            step=step,
        )
        self._context.interrupts.checkpoint(label)
        return result

    @staticmethod
    def _validate_state(state: RunState) -> None:
        """Reject values outside the public runtime-state contract."""

        if not isinstance(state, RunState):
            raise CouplerError(
                "RuntimeDriver.step_component state must be a RunState; "
                f"got {type(state).__name__}."
            )

    def _validate_component(self, component: str) -> None:
        """Reject names outside the prepared component dispatch mapping."""

        components = self._context.dispatch_context.components
        if isinstance(component, str) and component in components:
            return
        prepared_names = ", ".join(components) or "<none>"
        raise CouplerError(
            f"RuntimeDriver.step_component component {component!r} is not in "
            f"prepared components: {prepared_names}."
        )


def _validated_step_index(step: int | RuntimeArray) -> int:
    """Return a concrete scalar integer value for a custom driver step."""

    shape = getattr(step, "shape", None)
    if shape is not None and tuple(shape) != ():
        raise CouplerError(
            "RuntimeDriver.step_component step must be scalar; "
            f"got shape {tuple(shape)}."
        )

    value: object = step
    if shape is not None:
        try:
            value = step.item()  # type: ignore[union-attr]
        except Exception as error:
            raise CouplerError(
                "RuntimeDriver.step_component step must be a concrete scalar "
                f"integer value; got {type(step).__name__}."
            ) from error

    if isinstance(value, bool):
        raise CouplerError(
            "RuntimeDriver.step_component step must be a scalar integer index; "
            "boolean values are not accepted."
        )
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        numeric_value = float(value)
        if isfinite(numeric_value) and numeric_value.is_integer():
            return int(numeric_value)
        raise CouplerError(
            "RuntimeDriver.step_component step must be an integer value; "
            f"got {value!r}."
        )
    raise CouplerError(
        "RuntimeDriver.step_component step must be a concrete scalar integer "
        f"value; got {type(value).__name__}."
    )


def _validate_step_range(step_index: int, clock: Clock) -> None:
    """Reject a driver step outside the configured clock range."""

    if 0 <= step_index < clock.steps:
        return
    raise CouplerError(
        f"RuntimeDriver.step_component step {step_index} is outside "
        f"[0, {clock.steps})."
    )


def _clock_time_at_step(
    context: RuntimeRunContext,
    step_index: int,
) -> datetime | ModelDateTime:
    """Return the exact model time for a validated clock step."""

    for index, time, _ in context.clock.iter():
        if index == step_index:
            return time
    raise CouplerError(
        f"RuntimeDriver.step_component could not resolve clock time for "
        f"validated step {step_index}."
    )


__all__ = [
    "run_compiled_scanned_runtime",
    "run_compiled_period_output_runtime",
    "run_custom_backend",
    "run_host_period_output_runtime",
    "run_host_runtime",
    "run_scanned_runtime",
]
