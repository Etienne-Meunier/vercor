from __future__ import annotations

from dataclasses import dataclass
from inspect import Parameter, signature
from typing import TYPE_CHECKING, Any, Mapping, cast

from vercor.components.contracts import (
    AuthorStepCallable,
    LifecycleHooks,
    ComponentStepCallable,
    ComponentStepReturn,
    ComponentSpec,
)
from vercor.components._runtime_fields import apply_step_result, runtime_fields
from vercor.exceptions import ComponentError
from vercor.types import RuntimeArray

if TYPE_CHECKING:
    from vercor.components.base import Component
    from vercor.components.contexts import StepContext
    from vercor._runtime.state import ComponentRuntimeState


@dataclass(frozen=True)
class CallableComponentOptions:
    """Normalized callable-backed component construction options."""

    step: AuthorStepCallable
    payload: Any | None
    spec: ComponentSpec
    lifecycle_hooks: LifecycleHooks


def callable_component_options(
    step: AuthorStepCallable,
    *,
    spec: ComponentSpec | None = None,
    payload: Any | None = None,
) -> CallableComponentOptions:
    """Normalize shared public callable component constructor options."""

    spec = ComponentSpec() if spec is None else spec
    return CallableComponentOptions(
        step=step,
        payload=payload,
        spec=spec,
        lifecycle_hooks=spec.lifecycle,
    )


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
            context: StepContext,
            payload: Any | None,
        ) -> ComponentStepReturn:
            _ = context, payload
            return step(fields)

        return step_fields_only

    if arity == 2:

        def step_fields_and_context(
            fields: Mapping[str, RuntimeArray],
            context: StepContext,
            payload: Any | None,
        ) -> ComponentStepReturn:
            _ = payload
            return step(fields, context)

        return step_fields_and_context

    def step_fields_context_and_payload(
        fields: Mapping[str, RuntimeArray],
        context: StepContext,
        payload: Any | None,
    ) -> ComponentStepReturn:
        return step(fields, context, payload)

    return step_fields_context_and_payload


def _component_step_signature_error() -> ComponentError:
    """Return a consistent author-facing error for unsupported step signatures."""

    return ComponentError(
        "Component step callable must accept 1, 2, or 3 positional arguments: "
        "fields, optional context, and optional payload."
    )


class _CallableRuntimeMixin:
    """Shared metadata hooks for callable-backed component wrappers."""

    _step: ComponentStepCallable
    _payload: Any | None

    def _initialize_callable_runtime(
        self,
        *,
        step: AuthorStepCallable,
        payload: Any | None,
        spec: ComponentSpec,
        lifecycle_hooks: LifecycleHooks,
    ) -> None:
        component = cast("Component", self)
        self._step = normalize_component_step_callable(step)
        self._payload = payload
        component._spec = spec
        component._lifecycle_hooks = lifecycle_hooks

    def _default_runtime_payload(self) -> Any | None:
        """Return the payload supplied to the callable component factory."""

        return self._payload

    def _step_callable_runtime_state(
        self,
        component_state: "ComponentRuntimeState",
        context: StepContext,
    ) -> "ComponentRuntimeState":
        """Advance callable-backed runtime state using the normalized step."""

        component = cast("Component", self)
        return apply_step_result(
            component,
            component_state,
            self._step(
                runtime_fields(component_state),
                context,
                component_state.payload,
            ),
        )
