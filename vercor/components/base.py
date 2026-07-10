from __future__ import annotations

from abc import ABC
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from vercor.components.contracts import (
    FieldImportPolicy,
    LifecycleHooks,
    ComponentStepReturn,
    ComponentSpec,
    _AuthorStepCallable,
)
from vercor.components._callable_wrappers import (
    _CallableRuntimeMixin,
    callable_component_options,
)
from vercor.components._field_authoring import ComponentFieldAuthoringMixin
from vercor.components._lifecycle_api import ComponentLifecycleMixin
from vercor.components.contexts import StepContext
from vercor.grids import RectilinearGrid
from vercor.output import OutputConfig
from vercor.settings import Settings
from vercor.types import RuntimeArray

__all__ = [
    "Component",
]


@dataclass(init=False)
class Component(
    ComponentFieldAuthoringMixin,
    ComponentLifecycleMixin,
    ABC,
):
    """Active differentiable component-author contract for VerCOR model adapters.

    Component instances own mutable setup-time metadata: name, grid, seed data,
    and component-specific settings. During coupling, the coupler copies those
    seed fields into immutable runtime state containers so JAX can trace the
    integration. Active differentiable components implement :meth:`step`, which
    receives read-only field mappings and returns field updates or
    :class:`vercor.components.StepResult`. Data-only forcing adapters should
    inherit :class:`vercor.components.DataComponent`; non-differentiable adapters
    should inherit :class:`vercor.components.HostComponent`.

    Common exchange-field conventions:
        - fields use SI units
        - surface fluxes are positive downward and negative upward
        - data fields use canonical trailing horizontal dimensions:
          (nLat, nLon), (nTime, nLat, nLon), (nLev, nLat, nLon), or
          (nTime, nLev, nLat, nLon)

    Attributes:
        name: component name
        grid: component grid
        _data: internal storage for component data arrays to/from which fields
            seed the runtime state during initialization
        settings: component-specific settings
        _setup_metadata: non-runtime setup metadata for adapter provenance or
            diagnostics that must not enter runtime field validation
    """

    name: str
    grid: RectilinearGrid
    _data: dict[str, RuntimeArray] = field(default_factory=dict)
    settings: Settings = field(default_factory=Settings)
    _setup_metadata: dict[str, Any] = field(default_factory=dict)
    _spec: ComponentSpec = field(
        default_factory=ComponentSpec,
        init=False,
        repr=False,
    )
    _import_policy: FieldImportPolicy = field(
        default_factory=FieldImportPolicy,
        init=False,
        repr=False,
    )
    _lifecycle_hooks: LifecycleHooks = field(
        default_factory=LifecycleHooks,
        init=False,
        repr=False,
    )

    def __init__(
        self,
        name: str,
        grid: RectilinearGrid,
        *,
        settings: Settings | None = None,
        spec: ComponentSpec | None = None,
    ) -> None:
        """Create a component configuration shell for setup-time authoring."""

        self.name = name
        self.grid = grid
        self._data = {}
        self.settings = Settings() if settings is None else settings
        self._setup_metadata = {}
        self._spec = ComponentSpec() if spec is None else spec
        self._import_policy = FieldImportPolicy()
        self._lifecycle_hooks = self._spec.lifecycle

    @property
    def output(self) -> OutputConfig:
        """Return the output extension configuration from ``spec``."""

        return self._spec.output

    @property
    def import_policy(self) -> FieldImportPolicy:
        """Return this component's data import policy."""

        return self._import_policy

    @classmethod
    def from_step(
        cls,
        name: str,
        grid: RectilinearGrid,
        step: Callable[..., ComponentStepReturn],
        *,
        spec: ComponentSpec | None = None,
        payload: Any | None = None,
        settings: Settings | None = None,
    ) -> "Component":
        """Create a differentiable component from a user step callable.

        ``spec.inputs`` declares fields the model reads, ``spec.outputs``
        declares fields the model writes, and ``spec.defaults`` declares
        concrete runtime defaults.
        Scalar default values expand to this component's grid shape.
        """

        options = callable_component_options(
            step,
            spec=spec,
            payload=payload,
        )
        return _CallableComponent(
            name=name,
            grid=grid,
            step=options.step,
            payload=options.payload,
            settings=settings,
            spec=options.spec,
            lifecycle_hooks=options.lifecycle_hooks,
        )

    @property
    def data(self) -> None:
        """Block removed public setup-data access."""

        raise AttributeError(
            "Component.data is not public API; use seed_field()/seed_fields() "
            "for setup fields or RunState.component(...).field(...) for results."
        )

    @data.setter
    def data(self, value: object) -> None:
        _ = value
        raise AttributeError(
            "Component.data is not public API; use seed_field()/seed_fields() "
            "for setup fields or RunState.component(...).field(...) for results."
        )

    @property
    def setup_metadata(self) -> None:
        """Block removed public setup-metadata access."""

        raise AttributeError(
            "Component.setup_metadata is not public API; setup metadata is "
            "adapter-private."
        )

    @setup_metadata.setter
    def setup_metadata(self, value: object) -> None:
        _ = value
        raise AttributeError(
            "Component.setup_metadata is not public API; setup metadata is "
            "adapter-private."
        )

    def step(
        self,
        fields: Mapping[str, RuntimeArray],
        context: "StepContext",
        payload: Any | None = None,
    ) -> ComponentStepReturn:
        """Return runtime field updates for one component step."""

        _ = fields, context, payload
        raise NotImplementedError(
            f"Component '{self.name}' must implement step(...) or be created "
            "with Component.from_step(...)."
        )

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}:\n"
            f" ├── Name: {self.name}\n"
            f" ├── Runtime fields: Configured by Coupler runtime contract\n"
            f" └── Grid name: {self.grid.name}\n"
            f"     └── Shape: {self.grid.shape}\n"
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, grid={repr(self.grid)})"


class _CallableComponent(_CallableRuntimeMixin, Component):
    """Differentiable component backed by an author-provided step callable."""

    def __init__(
        self,
        name: str,
        grid: RectilinearGrid,
        *,
        step: _AuthorStepCallable,
        payload: Any | None,
        settings: Settings | None,
        spec: ComponentSpec,
        lifecycle_hooks: LifecycleHooks,
    ) -> None:
        if settings is None:
            Component.__init__(self, name=name, grid=grid, spec=spec)
        else:
            Component.__init__(
                self,
                name=name,
                grid=grid,
                settings=settings,
                spec=spec,
            )
        self._initialize_callable_runtime(
            step=step,
            payload=payload,
            spec=spec,
            lifecycle_hooks=lifecycle_hooks,
        )

    def step(
        self,
        fields: Mapping[str, RuntimeArray],
        context: StepContext,
        payload: Any | None = None,
    ) -> ComponentStepReturn:
        """Return field updates from the callable-backed component step."""

        return self._step(fields, context, payload)
