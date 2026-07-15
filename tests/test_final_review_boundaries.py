"""Regression tests for final VerCOR 0.4 public-boundary review findings."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime
import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from tests._coverage_support import make_test_grid
from vercor import Clock, Coupler
from vercor.components import (
    CallableComponent,
    ComponentSpec,
    SetupContext,
    StepContext,
    ValidationContext,
)
from vercor.diagnostics import print_component_field_means_table
from vercor.exchanges import Exchange
from vercor.output import (
    OutputFrame,
    OutputSpec,
    OutputTarget,
    PeriodOutput,
)
from vercor.output._session import build_output_plan
from vercor.runtime import (
    ExecutionContext,
    RuntimeOptions,
    StepPlan,
    WorkflowContext,
)
from vercor.state import ComponentState

pytestmark = pytest.mark.fast_always

_BoundaryCall = Callable[[object], object]


def _clock() -> Clock:
    return Clock(datetime(2000, 1, 1), dt_seconds=60.0, steps=1)


def _component() -> CallableComponent:
    def step(
        fields: Mapping[str, Any],
        context: StepContext,
    ) -> Mapping[str, Any]:
        _ = context
        return {"temperature": fields["temperature"]}

    return CallableComponent(
        "ATM",
        make_test_grid(name="review-boundary"),
        step,
        spec=ComponentSpec(
            outputs=("temperature",),
            initial_fields={"temperature": 280.0},
        ),
    )


def _run_state_components(value: object) -> object:
    component = _component()
    state = Coupler(
        _clock(),
        components=(component,),
        run_order=(component.name,),
        log_level="WARNING",
    ).initial_state()
    return state.components(cast(Any, value))


def _setup_context(value: object) -> SetupContext:
    return SetupContext(
        start=datetime(2000, 1, 1),
        dt_seconds=60.0,
        run_order=cast(Any, value),
        logger=logging.getLogger("final-review-boundaries"),
    )


def _diagnostic_component_order(value: object) -> None:
    print_component_field_means_table(
        {},
        (),
        component_order=cast(Any, value),
    )


@pytest.mark.parametrize("text", ("temperature", b"temperature"))
@pytest.mark.parametrize(
    ("label", "boundary"),
    (
        (
            "Exchange.fields",
            lambda value: Exchange("ATM", "OCN", cast(Any, value)),
        ),
        (
            "WorkflowContext.component_names",
            lambda value: WorkflowContext(
                _clock(),
                cast(Any, value),
                ("ATM",),
            ),
        ),
        (
            "WorkflowContext.default_order",
            lambda value: WorkflowContext(
                _clock(),
                ("ATM",),
                cast(Any, value),
            ),
        ),
        (
            "StepPlan.components",
            lambda value: StepPlan(0, cast(Any, value)),
        ),
        (
            "ExecutionContext.component_names",
            lambda value: ExecutionContext(
                _clock(),
                cast(Any, value),
                RuntimeOptions(),
            ),
        ),
        ("SetupContext.run_order", _setup_context),
        (
            "ValidationContext.receives",
            lambda value: ValidationContext(
                ComponentState("ATM", make_test_grid()),
                receives=cast(Any, value),
            ),
        ),
        (
            "ValidationContext.sends",
            lambda value: ValidationContext(
                ComponentState("ATM", make_test_grid()),
                sends=cast(Any, value),
            ),
        ),
        (
            "OutputFrame.dimension_order",
            lambda value: OutputFrame({}, dimension_order=cast(Any, value)),
        ),
        (
            "PeriodOutput.variables",
            lambda value: PeriodOutput(variables=cast(Any, value)),
        ),
        ("RunState.components names", _run_state_components),
        ("diagnostic component_order", _diagnostic_component_order),
    ),
)
def test_public_name_sequence_boundaries_reject_text_scalars(
    label: str,
    boundary: _BoundaryCall,
    text: str | bytes,
) -> None:
    with pytest.raises(
        TypeError,
        match=rf"{label} must be a sequence, not {type(text).__name__}",
    ):
        boundary(text)


@pytest.mark.parametrize(
    ("value", "error"),
    (
        (True, TypeError),
        (False, TypeError),
        (np.bool_(True), TypeError),
        (0, ValueError),
        (0.0, ValueError),
        (-1, ValueError),
        (-1.5, ValueError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
        (float("-inf"), ValueError),
        (10**1000, ValueError),
        (1 + 2j, TypeError),
        ("31536000", TypeError),
    ),
)
def test_runtime_options_rejects_invalid_model_year_seconds(
    value: object,
    error: type[Exception],
) -> None:
    with pytest.raises(
        error,
        match="model_year_seconds must be a finite positive real number",
    ):
        RuntimeOptions(model_year_seconds=cast(Any, value))


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        (1, 1.0),
        (1.5, 1.5),
        (np.int64(2), 2.0),
        (np.float32(2.5), 2.5),
        (np.float64(3.5), 3.5),
    ),
)
def test_runtime_options_canonicalizes_valid_model_year_seconds(
    value: object,
    expected: float,
) -> None:
    options = RuntimeOptions(model_year_seconds=cast(Any, value))

    assert options.model_year_seconds == expected
    assert type(options.model_year_seconds) is float


def test_prepared_binding_does_not_delegate_private_markers_and_uses_spec_output(
    tmp_path: Path,
) -> None:
    class Provider:
        def sample(self, context: object) -> OutputFrame:
            _ = context
            return OutputFrame({})

    provider = Provider()
    period = PeriodOutput()
    component = _component()
    component.spec = ComponentSpec(
        outputs=component.spec.outputs,
        initial_fields=component.spec.initial_fields,
        output=OutputSpec(provider=provider, period=period),
    )
    setattr(component, "_private_output_marker", object())
    coupler = Coupler(
        _clock(),
        components=(component,),
        run_order=(component.name,),
        log_level="WARNING",
    )

    prepared = coupler._ensure_prepared()
    binding = prepared.components[component.name]

    with pytest.raises(AttributeError, match="_private_output_marker"):
        getattr(binding, "_private_output_marker")

    plan = build_output_plan(
        prepared.components,
        prepared.clock,
        OutputTarget(tmp_path),
    )
    assert len(plan.schemas) == 1
    assert plan.schemas[0].provider is provider
    assert plan.schemas[0].period is period
