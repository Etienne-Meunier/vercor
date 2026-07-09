from __future__ import annotations

from typing import TYPE_CHECKING

from vercor.config import ExecutionBackend

if TYPE_CHECKING:
    from vercor._runtime.run_context import RuntimeRunContext
    from vercor.state import RunState


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

    return backend.run(state, context=context)


__all__ = [
    "_HostLoopBackend",
    "_JAXScannedBackend",
    "run_custom_backend",
]
