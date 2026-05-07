from __future__ import annotations

from inspect import Parameter, signature
from typing import TYPE_CHECKING, Any, Mapping, cast

from vercor.components._contracts import (
    AuthorStepCallable,
    ComponentFieldSpec,
    ComponentStepCallable,
    ComponentStepResult,
    ComponentStepReturn,
    FieldDefaults,
    FieldNames,
    unique_field_names,
)
from vercor.dtypes import jax_zeros
from vercor.exceptions import ComponentError
from vercor.field_layout import validate_component_data_layout
from vercor.grid import RectilinearGrid
from vercor.runtime.contexts import RuntimeStepContext
from vercor.settings import VercorSettings
from vercor.types import RuntimeArray
from vercor.components.base import Component, HostRuntimeComponent

if TYPE_CHECKING:
    from vercor.runtime import RuntimeComponentContract, RuntimeComponentState


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
    _required_fields: tuple[str, ...]
    _prefill_fields: tuple[str, ...]
    _field_defaults: dict[str, RuntimeArray]

    def _initialize_callable_runtime(
        self,
        *,
        payload: Any | None,
        required_fields: FieldNames,
        prefill_fields: FieldNames,
        field_defaults: FieldDefaults,
    ) -> None:
        self._payload = payload
        self._prefill_fields = unique_field_names(prefill_fields)
        self._field_defaults = dict(field_defaults or {})
        component = cast("Component", self)
        validate_component_data_layout(
            component_name=component.name,
            grid_shape=component.grid.shape,
            data=self._field_defaults,
        )
        self._required_fields = unique_field_names(
            (
                *tuple(required_fields),
                *self._prefill_fields,
                *tuple(self._field_defaults),
            )
        )
        component._field_spec = ComponentFieldSpec(
            outputs=self._prefill_fields,
            required_fields=required_fields,
            default_fields=self._field_defaults,
        )

    def create_runtime_payload(self) -> Any | None:
        """Return the payload supplied to the callable component factory."""

        return self._payload

    def prefill_runtime_state_fields(
        self,
        data: dict[str, RuntimeArray],
        incoming: dict[str, RuntimeArray],
        outgoing: dict[str, RuntimeArray],
        contract: "RuntimeComponentContract",
    ) -> None:
        """Pre-seed callable runtime fields declared by wrapper metadata."""

        component = cast("Component", self)
        for field_name, field_value in self._field_defaults.items():
            data.setdefault(field_name, field_value)
        zeros = jax_zeros(component.grid.shape, component.settings)
        for field_name in self._prefill_fields:
            data.setdefault(field_name, zeros)
        _ = incoming, outgoing, contract

    def validate_runtime_state(
        self,
        component_state: "RuntimeComponentState",
        contract: "RuntimeComponentContract",
    ) -> None:
        """Validate callable runtime fields declared as required."""

        _ = contract
        cast("Component", self).require_runtime_fields(
            component_state,
            *self._required_fields,
        )


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
        fields: Mapping[str, RuntimeArray] | None = None,
        payload: Any | None = None,
        settings: VercorSettings | None = None,
        required_fields: FieldNames = (),
        prefill_fields: FieldNames = (),
        field_defaults: FieldDefaults = None,
    ) -> None:
        if settings is None:
            super().__init__(name=name, grid=grid)
        else:
            super().__init__(name=name, grid=grid, settings=settings)
        self._step = normalize_component_step_callable(step)
        self._initialize_callable_runtime(
            payload=payload,
            required_fields=required_fields,
            prefill_fields=prefill_fields,
            field_defaults=field_defaults,
        )
        if fields is not None:
            self.seed_fields(fields)

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
        fields: Mapping[str, RuntimeArray] | None = None,
        payload: Any | None = None,
        settings: VercorSettings | None = None,
        required_fields: FieldNames = (),
        prefill_fields: FieldNames = (),
        field_defaults: FieldDefaults = None,
    ) -> None:
        if settings is None:
            super().__init__(name=name, grid=grid)
        else:
            super().__init__(name=name, grid=grid, settings=settings)
        self._step = normalize_component_step_callable(step)
        self._initialize_callable_runtime(
            payload=payload,
            required_fields=required_fields,
            prefill_fields=prefill_fields,
            field_defaults=field_defaults,
        )
        if fields is not None:
            self.seed_fields(fields)

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
    fields: Mapping[str, RuntimeArray] | None = None,
    payload: Any | None = None,
    settings: VercorSettings | None = None,
    required_fields: FieldNames = (),
    prefill_fields: FieldNames = (),
    field_defaults: FieldDefaults = None,
) -> "Component":
    """Create a differentiable callable-backed component."""

    return _CallableComponent(
        name=name,
        grid=grid,
        step=step,
        fields=fields,
        payload=payload,
        settings=settings,
        required_fields=required_fields,
        prefill_fields=prefill_fields,
        field_defaults=field_defaults,
    )


def make_callable_host_component(
    name: str,
    grid: RectilinearGrid,
    *,
    step: AuthorStepCallable,
    fields: Mapping[str, RuntimeArray] | None = None,
    payload: Any | None = None,
    settings: VercorSettings | None = None,
    required_fields: FieldNames = (),
    prefill_fields: FieldNames = (),
    field_defaults: FieldDefaults = None,
) -> "HostRuntimeComponent":
    """Create a host-runtime callable-backed component."""

    return _CallableHostRuntimeComponent(
        name=name,
        grid=grid,
        step=step,
        fields=fields,
        payload=payload,
        settings=settings,
        required_fields=required_fields,
        prefill_fields=prefill_fields,
        field_defaults=field_defaults,
    )
