from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from vercor.runtime import ExecutionBackend, ExecutionContext
from vercor.calendar import ModelDateTime
from vercor._runtime.driver import step_runtime_component
from vercor._runtime.time import scalar_runtime_step_info

if TYPE_CHECKING:
    from vercor._runtime.run_context import RuntimeRunContext
    from vercor.state import RunState
    from vercor.types import RuntimeArray


class _JAXScannedBackend:
    """Private backend for the differentiable compiled scanned runtime."""

    def run(
        self,
        state: "RunState",
        *,
        context: "RuntimeRunContext",
    ) -> "RunState":
        """Run the built-in scanned backend."""

        from vercor._runtime.runner import _run_compiled_scanned_runtime

        return _run_compiled_scanned_runtime(state, context=context)


class _HostLoopBackend:
    """Private backend for the Python host loop runtime."""

    def run(
        self,
        state: "RunState",
        *,
        context: "RuntimeRunContext",
    ) -> "RunState":
        """Run the built-in host-loop backend."""

        from vercor._runtime.runner import run_host_runtime

        return run_host_runtime(
            state,
            run_order=context.run_order,
            clock=context.clock,
            settings=context.dispatch_context.settings,
            model_year_seconds=context.options.model_year_seconds,
            logger=context.logger,
            dispatch_context=context.dispatch_context,
            interrupts=context.interrupts,
        )


def run_custom_backend(
    backend: ExecutionBackend,
    state: "RunState",
    *,
    context: "RuntimeRunContext",
) -> "RunState":
    """Delegate runtime execution to a public custom backend."""

    public_context = ExecutionContext(
        clock=context.clock,
        run_order=tuple(context.run_order),
        options=context.options,
        logger=context.logger,
    )
    return backend.run(
        state,
        context=public_context,
        driver=_RuntimeDriverAdapter(context),
    )


class _RuntimeDriverAdapter:
    """Public driver implementation backed by VerCOR's private runtime dispatch."""

    def __init__(self, context: "RuntimeRunContext") -> None:
        self._context = context

    def step_component(
        self,
        state: "RunState",
        component: str,
        *,
        step: int | "RuntimeArray",
    ) -> "RunState":
        """Advance one component with private dispatch/regridding mechanics."""

        step_index = _host_step_index(step)
        step_time = _clock_time_at_step(self._context, step_index)
        step_info = scalar_runtime_step_info(
            step_time,
            self._context.clock,
            self._context.dispatch_context.settings,
            model_year_seconds=self._context.options.model_year_seconds,
        )
        return step_runtime_component(
            state,
            component,
            step_info,
            dispatch_context=self._context.dispatch_context,
            allow_host_runtime=True,
            time=step_time,
            logger=self._context.logger,
            step=step,
        )


def _host_step_index(step: int | "RuntimeArray") -> int:
    """Return a best-effort host integer step index for custom backends."""

    try:
        return int(step)
    except (TypeError, ValueError):
        return 0


def _clock_time_at_step(
    context: "RuntimeRunContext",
    step_index: int,
) -> datetime | ModelDateTime:
    """Return the model time for ``step_index`` from the runtime clock."""

    for index, time, _ in context.clock.iter():
        if index == step_index:
            return time
    return context.clock.start


__all__ = [
    "_HostLoopBackend",
    "_JAXScannedBackend",
    "run_custom_backend",
]
