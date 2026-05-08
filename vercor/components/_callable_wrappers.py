from __future__ import annotations

from inspect import Parameter, signature
from typing import TYPE_CHECKING, Any, Mapping, cast

from vercor.components._contracts import (
    AuthorFieldValues,
    AuthorStepCallable,
    ComponentFieldSpec,
    ComponentStepCallable,
    ComponentStepResult,
    ComponentStepReturn,
)
from vercor.exceptions import ComponentError
from vercor.grid import RectilinearGrid
from vercor.runtime.contexts import RuntimeStepContext
from vercor.settings import VercorSettings
from vercor.types import RuntimeArray
from vercor.components.base import Component, HostRuntimeComponent

if TYPE_CHECKING:
    from vercor.runtime import RuntimeComponentState


def apply_callable_step_result(
    component: "Component",
    component_state: "RuntimeComponentState",
    result: ComponentStepReturn,
) -> "RuntimeComponentState":
    """Apply a callable wrapper result to runtime state."""

    if isinstance(result, ComponentStepResult):
        updated_state = component.with_runtime_fields(component_state, result.fields)
        return updated_state.with_runtime_payload(result.payload)

    return component.with_runtime_fields(component_state, result)


def normalize_component_step_callable(
    step: AuthorStepCallable,
) -> ComponentStepCallable:
    """Adapt supported author step signatures to the runtime wrapper shape."""

    try:
        step_signature = signature(step)
    except (TypeError, ValueError) as exc:
        raise ComponentError(
            "Component step callable must expose an inspectable signature that "
            "accepts 1, 2, or 3 positional arguments: fields, optional context, "
            "and optional payload."
        ) from exc

    parameters = tuple(step_signature.parameters.values())
    positional_parameters = tuple(
        parameter
        for parameter in parameters
        if parameter.kind
        in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
    )
    required_positional_parameters = tuple(
        parameter
        for parameter in positional_parameters
        if parameter.default is Parameter.empty
    )
    required_keyword_only_parameters = tuple(
        parameter
        for parameter in parameters
        if parameter.kind == Parameter.KEYWORD_ONLY
        and parameter.default is Parameter.empty
    )
    has_varargs = any(
        parameter.kind == Parameter.VAR_POSITIONAL for parameter in parameters
    )

    if required_keyword_only_parameters:
        required_names = ", ".join(
            parameter.name for parameter in required_keyword_only_parameters
        )
        raise ComponentError(
            "Component step callable has required keyword-only argument(s) "
            f"{required_names}; use 1, 2, or 3 positional arguments instead."
        )

    if has_varargs:
        if len(required_positional_parameters) > 3:
            raise _component_step_signature_error()
        arity = 3
    else:
        if (
            len(positional_parameters) < 1
            or len(positional_parameters) > 3
            or len(required_positional_parameters) > 3
        ):
            raise _component_step_signature_error()
        arity = len(positional_parameters)

    if arity == 1:

        def step_fields_only(
            fields: Mapping[str, RuntimeArray],
            context: RuntimeStepContext,
            payload: Any | None,
        ) -> ComponentStepReturn:
            _ = context, payload
            return step(fields)

        return step_fields_only

    if arity == 2:

        def step_fields_and_context(
            fields: Mapping[str, RuntimeArray],
            context: RuntimeStepContext,
            payload: Any | None,
        ) -> ComponentStepReturn:
            _ = payload
            return step(fields, context)

        return step_fields_and_context

    def step_fields_context_payload(
        fields: Mapping[str, RuntimeArray],
        context: RuntimeStepContext,
        payload: Any | None,
    ) -> ComponentStepReturn:
        return step(fields, context, payload)

    return step_fields_context_payload


def _component_step_signature_error() -> ComponentError:
    """Return a consistent author-facing error for unsupported step signatures."""

    return ComponentError(
        "Component step callable must accept 1, 2, or 3 positional arguments: "
        "fields, optional context, and optional payload."
    )


class _CallableRuntimeMixin:
    """Shared metadata hooks for callable-backed component wrappers."""

    _payload: Any | None

    def _initialize_callable_runtime(
        self,
        *,
        payload: Any | None,
        field_spec: ComponentFieldSpec,
    ) -> None:
        self._payload = payload
        component = cast("Component", self)
        component.declare_fields(field_spec)

    def create_runtime_payload(self) -> Any | None:
        """Return the payload supplied to the callable component factory."""

        return self._payload


class _CallableComponent(_CallableRuntimeMixin, Component):
    """Differentiable component backed by a user-provided step callable."""

    _step: ComponentStepCallable
    _payload: Any | None

    def __init__(
        self,
        name: str,
        grid: RectilinearGrid,
        *,
        step: AuthorStepCallable,
        initial_fields: AuthorFieldValues = None,
        payload: Any | None = None,
        settings: VercorSettings | None = None,
        field_spec: ComponentFieldSpec | None = None,
    ) -> None:
        if settings is None:
            super().__init__(name=name, grid=grid)
        else:
            super().__init__(name=name, grid=grid, settings=settings)
        self._step = normalize_component_step_callable(step)
        self._initialize_callable_runtime(
            payload=payload,
            field_spec=field_spec or ComponentFieldSpec(),
        )
        if initial_fields is not None:
            self.seed_fields(initial_fields)

    def step_runtime_state(
        self,
        component_state: "RuntimeComponentState",
        context: RuntimeStepContext,
    ) -> "RuntimeComponentState":
        """Advance this callable-backed differentiable component one step."""

        return apply_callable_step_result(
            self,
            component_state,
            self._step(
                self.runtime_fields(component_state),
                context,
                component_state.runtime_payload,
            ),
        )


class _CallableHostRuntimeComponent(
    _CallableRuntimeMixin,
    HostRuntimeComponent,
):
    """Host-runtime component backed by a user-provided step callable."""

    _step: ComponentStepCallable
    _payload: Any | None

    def __init__(
        self,
        name: str,
        grid: RectilinearGrid,
        *,
        step: AuthorStepCallable,
        initial_fields: AuthorFieldValues = None,
        payload: Any | None = None,
        settings: VercorSettings | None = None,
        field_spec: ComponentFieldSpec | None = None,
    ) -> None:
        if settings is None:
            super().__init__(name=name, grid=grid)
        else:
            super().__init__(name=name, grid=grid, settings=settings)
        self._step = normalize_component_step_callable(step)
        self._initialize_callable_runtime(
            payload=payload,
            field_spec=field_spec or ComponentFieldSpec(),
        )
        if initial_fields is not None:
            self.seed_fields(initial_fields)

    def step_host_runtime_state(
        self,
        component_state: "RuntimeComponentState",
        context: RuntimeStepContext,
    ) -> "RuntimeComponentState":
        """Advance this callable-backed host component one step."""

        return apply_callable_step_result(
            self,
            component_state,
            self._step(
                self.runtime_fields(component_state),
                context,
                component_state.runtime_payload,
            ),
        )


def make_callable_component(
    name: str,
    grid: RectilinearGrid,
    *,
    step: AuthorStepCallable,
    initial_fields: AuthorFieldValues = None,
    payload: Any | None = None,
    settings: VercorSettings | None = None,
    field_spec: ComponentFieldSpec | None = None,
) -> "Component":
    """Create a differentiable callable-backed component."""

    return _CallableComponent(
        name=name,
        grid=grid,
        step=step,
        initial_fields=initial_fields,
        payload=payload,
        settings=settings,
        field_spec=field_spec,
    )


def make_callable_host_component(
    name: str,
    grid: RectilinearGrid,
    *,
    step: AuthorStepCallable,
    initial_fields: AuthorFieldValues = None,
    payload: Any | None = None,
    settings: VercorSettings | None = None,
    field_spec: ComponentFieldSpec | None = None,
) -> "HostRuntimeComponent":
    """Create a host-runtime callable-backed component."""

    return _CallableHostRuntimeComponent(
        name=name,
        grid=grid,
        step=step,
        initial_fields=initial_fields,
        payload=payload,
        settings=settings,
        field_spec=field_spec,
    )
