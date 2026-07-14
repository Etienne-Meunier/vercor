"""Public coupling orchestration facade."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vercor.clock import Clock
from vercor.components import Component
from vercor.coupler import Coupler
from vercor.exchanges import Exchange
from vercor.runtime import RuntimeOptions

if TYPE_CHECKING:
    from vercor.jax_logging import LoggerLike


@dataclass(frozen=True)
class CouplerSpec:
    """Plain reusable recipe for constructing a configured coupler."""

    components: tuple[Component, ...]
    exchanges: tuple[Exchange, ...] = ()
    run_order: tuple[str, ...] = ()
    runtime: RuntimeOptions = field(default_factory=RuntimeOptions)

    def __init__(
        self,
        *,
        components: Sequence[Component],
        exchanges: Sequence[Exchange] = (),
        run_order: Sequence[str] = (),
        runtime: RuntimeOptions | None = None,
    ) -> None:
        """Create an immutable coupler recipe from public configuration objects."""

        object.__setattr__(self, "components", tuple(components))
        object.__setattr__(self, "exchanges", tuple(exchanges))
        object.__setattr__(self, "run_order", tuple(run_order))
        object.__setattr__(
            self,
            "runtime",
            RuntimeOptions() if runtime is None else runtime,
        )

    def build(
        self,
        clock: Clock,
        *,
        logger: "LoggerLike | None" = None,
        log_level: int | str = "INFO",
    ) -> Coupler:
        """Build a coupler from this recipe and a concrete clock."""

        return Coupler(
            clock,
            components=self.components,
            exchanges=self.exchanges,
            run_order=self.run_order,
            runtime=self.runtime,
            logger=logger,
            log_level=log_level,
        )


__all__ = ["Coupler", "CouplerSpec", "Exchange"]
