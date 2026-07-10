from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from vercor.components.runtime_execution import host_component_names
from vercor.exceptions import ComponentError
from vercor.jax_logging import LoggerLike
from vercor.runtime import ExecutionBackend
from vercor.state import RunState
from vercor._runtime.backends import (
    run_compiled_scanned_runtime,
    run_custom_backend,
    run_host_runtime,
)
from vercor._runtime.run_context import RuntimeRunContext


def run_coupler_runtime(
    runtime_state: RunState,
    *,
    context: RuntimeRunContext,
) -> RunState:
    """Select a validated runtime execution mode and delegate to its backend."""

    with context.interrupts.signal_scope():
        if not isinstance(context.execution, str):
            return run_custom_backend(
                cast(ExecutionBackend, context.execution),
                runtime_state,
                context=context,
            )

        host_names = host_component_names(context.dispatch_context.components)
        if context.execution == "jax":
            if host_names:
                raise ComponentError(
                    "RuntimeOptions(execution='jax') cannot run host-backed "
                    f"component(s): {', '.join(host_names)}"
                )
            return run_compiled_scanned_runtime(runtime_state, context=context)

        if context.execution == "host":
            if host_names:
                _warn_non_differentiable_host_runtime(
                    context.logger,
                    host_names,
                )
            return _run_host_backend(runtime_state, context=context)

        if not host_names:
            return run_compiled_scanned_runtime(runtime_state, context=context)
        _warn_non_differentiable_host_runtime(
            context.logger,
            host_names,
        )
        return _run_host_backend(runtime_state, context=context)


def _run_host_backend(
    runtime_state: RunState,
    *,
    context: RuntimeRunContext,
) -> RunState:
    """Delegate to the host backend using the selected runtime context."""

    return run_host_runtime(
        runtime_state,
        run_order=context.run_order,
        clock=context.clock,
        settings=context.dispatch_context.settings,
        model_year_seconds=context.options.model_year_seconds,
        logger=context.logger,
        dispatch_context=context.dispatch_context,
        interrupts=context.interrupts,
    )


def _warn_non_differentiable_host_runtime(
    logger: LoggerLike,
    host_names: Sequence[str],
) -> None:
    """Warn that host-backed components make the coupled loop non-differentiable."""

    names = ", ".join(host_names)
    logger.warning(
        "Coupled loop is not differentiable because host-backed component(s) "
        f"require the Python runtime: {names}"
    )


__all__ = ["run_coupler_runtime"]
