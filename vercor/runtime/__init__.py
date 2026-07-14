"""Public workflow planning and runtime execution extension contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from numbers import Real
from typing import Literal, Protocol, runtime_checkable

import vercor.clock as _clock
import vercor.dtypes as _dtypes
import vercor.jax_logging as _jax_logging
import vercor.state as _state
from vercor._field_names import freeze_name_sequence as _freeze_name_sequence
from vercor.topology import TopologyPolicy as _TopologyPolicy


@dataclass(frozen=True)
class WorkflowContext:
    """Describe the clock and registered names available to a workflow.

    Attributes:
        clock: Complete public clock whose integer indices the plan must cover.
        component_names: Every registered component name in stable registration
            order. A workflow may schedule only these names.
        default_order: Constructor-supplied Coupler order repeated by
            :class:`SequentialWorkflow`. Custom workflows may reorder or omit
            these names and may schedule a registered name absent from this
            default.
    """

    clock: _clock.Clock
    component_names: tuple[str, ...]
    default_order: tuple[str, ...]

    def __post_init__(self) -> None:
        """Copy caller-owned component sequences into immutable tuples."""

        object.__setattr__(
            self,
            "component_names",
            _freeze_name_sequence(
                self.component_names,
                label="WorkflowContext.component_names",
            ),
        )
        object.__setattr__(
            self,
            "default_order",
            _freeze_name_sequence(
                self.default_order,
                label="WorkflowContext.default_order",
            ),
        )


@dataclass(frozen=True)
class StepPlan:
    """Specify work for one absolute clock step.

    ``step`` must equal the plan entry's zero-based clock position. Components
    are executed in tuple order; names must be registered strings and may not
    repeat within a step. An empty tuple advances time without dispatching a
    component.
    """

    step: int
    components: tuple[str, ...]

    def __post_init__(self) -> None:
        """Copy the ordered component sequence into an immutable tuple."""

        object.__setattr__(
            self,
            "components",
            _freeze_name_sequence(
                self.components,
                label="StepPlan.components",
            ),
        )


@dataclass(frozen=True)
class ExecutionPlan:
    """Store the complete static workflow result.

    Runtime validation requires exactly ``clock.steps`` entries with absolute
    indices ``0`` through ``clock.steps - 1`` in order. Every entry must be a
    :class:`StepPlan`, and all component names are checked against the complete
    registered-name set before any component is stepped.
    """

    steps: tuple[StepPlan, ...] = ()

    def __post_init__(self) -> None:
        """Copy the complete step sequence into an immutable tuple."""

        object.__setattr__(self, "steps", tuple(self.steps))


@runtime_checkable
class Workflow(Protocol):
    """Construct a completely validated static execution plan."""

    def build(self, context: WorkflowContext) -> ExecutionPlan:
        """Return one ascending plan entry for every step in ``context.clock``.

        The runtime validates the returned type, exact clock coverage and
        indices, registered string names, and per-step name uniqueness before
        execution.
        """
        ...


@dataclass(frozen=True)
class SequentialWorkflow:
    """Build the Coupler's default component order at every clock step.

    The result has one :class:`StepPlan` for each absolute clock index and uses
    :attr:`WorkflowContext.default_order` unchanged at every step.
    """

    def build(self, context: WorkflowContext) -> ExecutionPlan:
        """Return the complete ascending sequential execution plan."""

        return ExecutionPlan(
            steps=tuple(
                StepPlan(step=step, components=context.default_order)
                for step in range(context.clock.steps)
            )
        )


@dataclass(frozen=True)
class ExecutionChunk:
    """Store one core-defined contiguous execution-plan slice.

    Built-in JAX chunks contain a uniform component schedule. Output cadence
    may split that schedule into smaller chunks. Custom backends receive these
    exact :class:`StepPlan` objects and must consume them in order through the
    supplied :class:`RuntimeDriver` rather than constructing replacement plans.
    """

    steps: tuple[StepPlan, ...]

    def __post_init__(self) -> None:
        """Copy the chunk step sequence into an immutable tuple."""

        object.__setattr__(self, "steps", tuple(self.steps))


@dataclass(frozen=True)
class RuntimeOptions:
    """Own immutable runtime policy.

    Attributes:
        dtype: Precision policy applied at the prepared runtime boundary.
        backend: ``"auto"`` selects host execution only when a scheduled
            component is host-backed; ``"jax"`` forces scanned JAX execution
            and rejects scheduled host components; ``"host"`` forces the
            Python driver; an :class:`ExecutionBackend` handles core-defined
            chunks through the public driver.
        workflow: Static plan builder. Defaults to :class:`SequentialWorkflow`.
        topology: Optional topology policy, or ``None`` for no topology patch.
        model_year_seconds: Model-year duration in seconds used for periodic
            monthly forcing indices and interpolation weights.
    """

    dtype: _dtypes.DTypePolicy = field(default_factory=_dtypes.DTypePolicy)
    backend: Literal["auto", "jax", "host"] | "ExecutionBackend" = "auto"
    workflow: Workflow = field(default_factory=SequentialWorkflow)
    topology: _TopologyPolicy | None = None
    model_year_seconds: float = 365 * 86400.0

    def __post_init__(self) -> None:
        """Validate static extension contracts before runtime preparation."""

        if not isinstance(self.dtype, _dtypes.DTypePolicy):
            raise TypeError("dtype must be a DTypePolicy")
        if isinstance(self.backend, str):
            if self.backend not in ("auto", "jax", "host"):
                raise ValueError(
                    "backend must be 'auto', 'jax', 'host', or an execution backend"
                )
        elif not callable(getattr(self.backend, "execute", None)):
            raise TypeError(
                "execution backend must expose "
                "execute(state, *, context, chunk, driver)"
            )
        if not callable(getattr(self.workflow, "build", None)):
            raise TypeError("workflow must expose build(context)")
        if self.topology is not None and not callable(
            getattr(self.topology, "build", None)
        ):
            raise TypeError("topology policy must expose build(context)")
        if isinstance(self.model_year_seconds, bool) or not isinstance(
            self.model_year_seconds, Real
        ):
            raise TypeError("model_year_seconds must be a finite positive real number")
        try:
            normalized_model_year_seconds = float(self.model_year_seconds)
        except OverflowError as exc:
            raise ValueError(
                "model_year_seconds must be a finite positive real number"
            ) from exc
        if not isfinite(normalized_model_year_seconds) or (
            normalized_model_year_seconds <= 0.0
        ):
            raise ValueError("model_year_seconds must be a finite positive real number")
        object.__setattr__(
            self,
            "model_year_seconds",
            normalized_model_year_seconds,
        )


@dataclass(frozen=True)
class ExecutionContext:
    """Expose public run metadata to one custom-backend invocation.

    ``component_names`` contains every registered name, including components
    omitted from the current chunk. ``clock`` and ``options`` describe the full
    run; ``logger`` is the configured public logger. Prepared bindings, output
    sessions, cancellation controllers, and other private runtime state remain
    core-owned.
    """

    clock: _clock.Clock
    component_names: tuple[str, ...]
    options: RuntimeOptions
    logger: _jax_logging.LoggerLike | None = None

    def __post_init__(self) -> None:
        """Copy registered component names into an immutable tuple."""

        object.__setattr__(
            self,
            "component_names",
            _freeze_name_sequence(
                self.component_names,
                label="ExecutionContext.component_names",
            ),
        )


@runtime_checkable
class RuntimeDriver(Protocol):
    """Advance exact plans from a custom backend's active chunk.

    The driver maintains an identity-based ordered ledger. A backend must pass
    each supplied :class:`StepPlan` object exactly once; equal-but-forged,
    repeated, reordered, skipped, and out-of-chunk plans are rejected. Incoming
    and returned states are checked against the prepared component, grid,
    store, route, payload, dtype, shape, and mask schemas.
    """

    def run_step(self, state: _state.RunState, plan: StepPlan) -> _state.RunState:
        """Execute ``plan.components`` and return the resulting runtime state.

        Args:
            state: Schema-compatible state entering this planned clock step.
            plan: The next exact plan object from the active execution chunk.

        Returns:
            The validated state after ordered receive, component step, and send
            dispatch for every component in ``plan``.
        """
        ...


@runtime_checkable
class ExecutionBackend(Protocol):
    """Execute one core-defined chunk solely through :class:`RuntimeDriver`.

    The core owns workflow validation, chunking, lifecycle, output cadence and
    writes, cancellation, and chunk-result validation. The backend owns only
    how it invokes the supplied driver for the current chunk.
    """

    def execute(
        self,
        state: _state.RunState,
        *,
        context: ExecutionContext,
        chunk: ExecutionChunk,
        driver: RuntimeDriver,
    ) -> _state.RunState:
        """Consume every chunk plan exactly once and return a ``RunState``.

        Args:
            state: Validated state entering the chunk.
            context: Public metadata for the complete run.
            chunk: Core-authored contiguous plans for this invocation.
            driver: Identity-checking ordered plan executor.

        Returns:
            State after all plans in ``chunk`` have been executed.
        """
        ...


__all__ = [
    "ExecutionBackend",
    "ExecutionChunk",
    "ExecutionContext",
    "ExecutionPlan",
    "RuntimeDriver",
    "RuntimeOptions",
    "SequentialWorkflow",
    "StepPlan",
    "Workflow",
    "WorkflowContext",
]
