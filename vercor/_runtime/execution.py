"""Private workflow validation, chunking, and core execution coordination."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from vercor.components.runtime_execution import host_component_names
from vercor.exceptions import ComponentError, CouplerError
from vercor.runtime import (
    ExecutionBackend,
    ExecutionChunk,
    ExecutionContext,
    ExecutionPlan,
    StepPlan,
    WorkflowContext,
)
from vercor.state import RunState
from vercor._runtime.backends import (
    _JaxChunkExecutor,
    build_jax_chunk_executor,
    build_runtime_execution_data,
    execute_custom_chunk,
    execute_host_chunk,
    execute_jax_chunk,
)
from vercor._runtime.run_context import RuntimeRunContext
from vercor._runtime.state_validation import (
    validate_runtime_state as validate_runtime_state_schema,
)
from vercor.output._session import (
    _PeriodOutputBoundary,
    build_period_output_plan,
    has_period_output,
    validate_period_output_run_state_not_traced,
    write_period_output_boundary,
)


def build_validated_execution_plan(context: RuntimeRunContext) -> ExecutionPlan:
    """Build and completely validate one workflow plan before execution."""

    component_names = tuple(context.dispatch_context.components)
    workflow_context = WorkflowContext(
        clock=context.clock,
        component_names=component_names,
        default_order=tuple(context.run_order),
    )
    plan = context.options.workflow.build(workflow_context)
    if not isinstance(plan, ExecutionPlan):
        raise CouplerError(
            "Workflow.build(context) must return ExecutionPlan; "
            f"got {type(plan).__name__}."
        )
    if len(plan.steps) != context.clock.steps:
        raise CouplerError(
            "Workflow execution plan must contain exactly "
            f"{context.clock.steps} step plans; got {len(plan.steps)}."
        )
    registered = frozenset(component_names)
    for position, step_plan in enumerate(plan.steps):
        if not isinstance(step_plan, StepPlan):
            raise CouplerError(
                f"Workflow execution plan entry {position} must be StepPlan; "
                f"got {type(step_plan).__name__}."
            )
        if isinstance(step_plan.step, bool) or not isinstance(step_plan.step, int):
            raise CouplerError(
                f"Workflow step at position {position} must use an integer "
                f"clock index; got {step_plan.step!r}."
            )
        if step_plan.step != position:
            raise CouplerError(
                f"Workflow step at position {position} expected {position}; "
                f"got {step_plan.step} (clock indices must be exact)."
            )
        if not isinstance(step_plan.components, tuple):
            raise CouplerError(
                f"Workflow step {step_plan.step} components must be a tuple."
            )
        invalid_type = next(
            (name for name in step_plan.components if not isinstance(name, str)),
            None,
        )
        if invalid_type is not None:
            raise CouplerError(
                f"Workflow step {step_plan.step} component names must be str; "
                f"got {type(invalid_type).__name__}."
            )
        unknown = next(
            (name for name in step_plan.components if name not in registered),
            None,
        )
        if unknown is not None:
            raise CouplerError(
                f"Workflow step {step_plan.step} references unknown component "
                f"{unknown!r}."
            )
        duplicate = next(
            (
                name
                for name in step_plan.components
                if step_plan.components.count(name) > 1
            ),
            None,
        )
        if duplicate is not None:
            raise CouplerError(
                f"Workflow step {step_plan.step} contains duplicate component "
                f"{duplicate!r}."
            )
    return plan


def execute_plan(
    state: RunState,
    *,
    plan: ExecutionPlan,
    context: RuntimeRunContext,
) -> RunState:
    """Execute validated chunks while the core owns I/O and cancellation."""

    output_enabled = has_period_output(context.dispatch_context.components)
    output_plan = None
    output_session = None
    output_boundaries: dict[int, _PeriodOutputBoundary] = {}
    if output_enabled:
        validate_period_output_run_state_not_traced(state)
    execution_data = build_runtime_execution_data(context)
    if output_enabled:
        output_plan = build_period_output_plan(
            context.dispatch_context.components,
            state,
            context.clock,
            clock_steps=execution_data.clock_steps,
        )
        output_session = output_plan.initial_session
        output_boundaries = {
            boundary.stop_step: boundary for boundary in output_plan.boundaries
        }

    chunks = _execution_chunks(plan, output_enabled=output_enabled)
    scheduled_names = _scheduled_component_names(plan)
    scheduled_components = {
        name: context.dispatch_context.components[name] for name in scheduled_names
    }
    host_names = host_component_names(scheduled_components)
    backend = context.options.backend
    if backend == "jax" and host_names:
        raise ComponentError(
            "RuntimeOptions(backend='jax') cannot run host-backed "
            f"component(s): {', '.join(host_names)}"
        )
    selected = ("host" if host_names else "jax") if backend == "auto" else backend
    if selected == "host" and host_names:
        _warn_non_differentiable_host_runtime(context, host_names)

    public_context = ExecutionContext(
        clock=context.clock,
        component_names=tuple(context.dispatch_context.components),
        options=context.options,
        logger=context.logger,
    )
    jax_executors: dict[tuple[str, ...], _JaxChunkExecutor] = {}
    for chunk in chunks:
        context.interrupts.checkpoint("runtime chunk")
        if selected == "jax":
            components = chunk.steps[0].components
            executor = jax_executors.get(components)
            if executor is None:
                executor = build_jax_chunk_executor(
                    components,
                    context=context,
                    execution_data=execution_data,
                )
                jax_executors[components] = executor
            state = execute_jax_chunk(
                state,
                chunk=chunk,
                context=context,
                execution_data=execution_data,
                executor=executor,
            )
        elif selected == "host":
            state = execute_host_chunk(
                state,
                chunk=chunk,
                context=context,
                execution_data=execution_data,
            )
        else:
            state = execute_custom_chunk(
                cast(ExecutionBackend, selected),
                state,
                public_context=public_context,
                chunk=chunk,
                context=context,
                execution_data=execution_data,
            )
        _validate_chunk_result(state, context=context)
        if output_plan is not None and output_session is not None:
            # Output-enabled chunks are deliberately one step so sampling and
            # accumulation remain core-owned for every backend.
            output_session = output_session.accumulate(output_plan.schemas, state)
            stop = chunk.steps[-1].step + 1
            boundary = output_boundaries.get(stop)
            if boundary is not None:
                output_session = write_period_output_boundary(
                    output_plan,
                    output_session,
                    boundary,
                    logger=context.logger,
                )
        context.interrupts.checkpoint("runtime chunk")
    return state


def _execution_chunks(
    plan: ExecutionPlan,
    *,
    output_enabled: bool,
) -> tuple[ExecutionChunk, ...]:
    """Split a plan into contiguous uniform schedules and output sample points."""

    if output_enabled:
        return tuple(ExecutionChunk((step,)) for step in plan.steps)
    chunks: list[ExecutionChunk] = []
    current: list[StepPlan] = []
    current_components: tuple[str, ...] | None = None
    for step in plan.steps:
        if current and step.components != current_components:
            chunks.append(ExecutionChunk(tuple(current)))
            current = []
        current.append(step)
        current_components = step.components
    if current:
        chunks.append(ExecutionChunk(tuple(current)))
    return tuple(chunks)


def _scheduled_component_names(plan: ExecutionPlan) -> tuple[str, ...]:
    """Return first-use ordered names scheduled anywhere in ``plan``."""

    return tuple(dict.fromkeys(name for step in plan.steps for name in step.components))


def _validate_chunk_result(state: RunState, *, context: RuntimeRunContext) -> None:
    """Validate every built-in or custom backend result at the core boundary."""

    dispatch = context.dispatch_context
    validate_runtime_state_schema(
        state,
        components=dispatch.components,
        exchanges=dispatch.exchanges,
        regridders=dispatch.regridders,
        contracts=dispatch.contracts,
        run_order=tuple(dispatch.components),
    )


def _warn_non_differentiable_host_runtime(
    context: RuntimeRunContext,
    host_names: Sequence[str],
) -> None:
    """Warn that scheduled host components make the loop non-differentiable."""

    context.logger.warning(
        "Coupled loop is not differentiable because host-backed component(s) "
        f"require the Python runtime: {', '.join(host_names)}"
    )


__all__ = ["build_validated_execution_plan", "execute_plan"]
