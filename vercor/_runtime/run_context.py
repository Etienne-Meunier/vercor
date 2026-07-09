from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from vercor.clock import Clock
from vercor.runtime import RuntimeOptions
from vercor.jax_logging import LoggerLike
from vercor._runtime.dispatch_context import RuntimeDispatchContext
from vercor._runtime.interrupts import RuntimeInterruptController


@dataclass(frozen=True)
class RuntimeRunContext:
    """Static inputs required to execute one configured coupler runtime."""

    run_order: Sequence[str]
    clock: Clock
    logger: LoggerLike
    dispatch_context: RuntimeDispatchContext
    interrupts: RuntimeInterruptController
    options: RuntimeOptions

    @property
    def execution(self) -> object:
        """Return selected execution policy for private runtime helpers."""

        return self.options.execution


__all__ = ["RuntimeRunContext"]
