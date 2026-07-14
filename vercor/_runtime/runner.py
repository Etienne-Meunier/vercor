"""Runtime signal scope and delegation to workflow execution coordination."""

from __future__ import annotations

from vercor.state import RunState
from vercor._runtime.execution import (
    build_validated_execution_plan,
    execute_plan,
)
from vercor._runtime.run_context import RuntimeRunContext


def run_coupler_runtime(
    runtime_state: RunState,
    *,
    context: RuntimeRunContext,
) -> RunState:
    """Build, validate, chunk, and execute one configured workflow."""

    plan = build_validated_execution_plan(context)
    return execute_plan(runtime_state, plan=plan, context=context)


__all__ = ["run_coupler_runtime"]
