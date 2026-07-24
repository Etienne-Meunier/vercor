"""Private callable-signature normalization for component adapters."""

from __future__ import annotations

from collections.abc import Mapping
from inspect import Parameter, signature
from typing import Any

from vercor.components.contracts import (
    _AuthorStepCallable,
    _ComponentStepCallable,
    _ComponentStepReturn,
)
from vercor.components.contexts import StepContext
from vercor.exceptions import ComponentError
from vercor.types import RuntimeArray


def normalize_component_step_callable(
    step: _AuthorStepCallable,
) -> _ComponentStepCallable:
    """Adapt a one-, two-, or three-argument author step to the protocol."""

    if not callable(step):
        raise TypeError("step must be callable")
    try:
        step_signature = signature(step)
    except (TypeError, ValueError) as exc:
        raise _component_step_signature_error() from exc

    parameters = tuple(step_signature.parameters.values())
    positional = tuple(
        parameter
        for parameter in parameters
        if parameter.kind
        in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
    )
    required = tuple(
        parameter for parameter in positional if parameter.default is Parameter.empty
    )
    required_keyword_only = tuple(
        parameter
        for parameter in parameters
        if parameter.kind == Parameter.KEYWORD_ONLY
        and parameter.default is Parameter.empty
    )
    has_varargs = any(
        parameter.kind == Parameter.VAR_POSITIONAL for parameter in parameters
    )
    if required_keyword_only:
        names = ", ".join(parameter.name for parameter in required_keyword_only)
        raise ComponentError(
            "Component step callable has required keyword-only argument(s) "
            f"{names}; use 1, 2, or 3 positional arguments instead."
        )
    if has_varargs:
        if len(required) > 3:
            raise _component_step_signature_error()
        arity = 3
    else:
        if len(positional) not in (1, 2, 3) or len(required) > 3:
            raise _component_step_signature_error()
        arity = len(positional)

    if arity == 1:

        def fields_only(
            fields: Mapping[str, RuntimeArray],
            context: StepContext,
            payload: Any | None,
        ) -> _ComponentStepReturn:
            _ = context, payload
            return step(fields)

        return fields_only

    if arity == 2:

        def fields_and_context(
            fields: Mapping[str, RuntimeArray],
            context: StepContext,
            payload: Any | None,
        ) -> _ComponentStepReturn:
            _ = payload
            return step(fields, context)

        return fields_and_context

    def fields_context_and_payload(
        fields: Mapping[str, RuntimeArray],
        context: StepContext,
        payload: Any | None,
    ) -> _ComponentStepReturn:
        return step(fields, context, payload)

    return fields_context_and_payload


def _component_step_signature_error() -> ComponentError:
    """Return a focused unsupported-step-signature error."""

    return ComponentError(
        "Component step callable must accept 1, 2, or 3 positional arguments: "
        "fields, optional context, and optional payload."
    )


__all__: list[str] = []
