"""Focused coverage for the 0.4 component adapters and private runtime bridge."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from tests._coverage_support import make_test_grid
from vercor.clock import Clock
from vercor.components import (
    CallableComponent,
    ComponentSpec,
    LifecycleHooks,
    PrefillResult,
    SetupResult,
    StepResult,
)
from vercor.coupler import Coupler
from vercor.dtypes import DTypePolicy
from vercor.exceptions import ComponentError
from vercor.runtime import RuntimeOptions
from vercor._runtime.contracts import ExchangeContract


def _coupler(component: Any, *, steps: int = 1) -> Coupler:
    return Coupler(
        Clock(datetime(2000, 1, 1), 60.0, steps),
        components=(component,),
        run_order=(component.name,),
    )


@pytest.mark.parametrize("arity", [1, 2, 3])
def test_callable_component_adapts_supported_step_arities(arity: int) -> None:
    def one(fields: Any) -> Any:
        return {"value": fields["value"] + 1.0}

    def two(fields: Any, context: Any) -> Any:
        return {"value": fields["value"] + context.dt_seconds / 60.0}

    def three(fields: Any, context: Any, payload: Any) -> Any:
        _ = context
        return {"value": fields["value"] + payload}

    setup = None
    step = (one, two, three)[arity - 1]
    if arity == 3:
        setup = LifecycleHooks(
            setup=lambda component, context: SetupResult(payload=jnp.asarray(1.0))
        )
    component = CallableComponent(
        "MODEL",
        make_test_grid(),
        step,
        spec=ComponentSpec(
            outputs=("value",),
            initial_fields={"value": 0.0},
            lifecycle=setup,
        ),
    )

    result = _coupler(component).run()

    np.testing.assert_array_equal(
        result.component("MODEL").field("value"), np.ones((2, 2))
    )


@pytest.mark.parametrize(
    "step",
    [lambda: {}, lambda a, b, c, d: {}, lambda fields, *, required: {}],
)
def test_callable_component_rejects_invalid_step_signatures(step: Any) -> None:
    with pytest.raises(ComponentError, match="1, 2, or 3 positional"):
        CallableComponent("MODEL", make_test_grid(), step)


def test_step_result_can_preserve_replace_and_clear_payload() -> None:
    def step(fields: Any, context: Any, payload: Any) -> StepResult:
        _ = fields, context
        if payload == 1:
            return StepResult(payload=2)
        return StepResult(payload=None)

    component = CallableComponent(
        "MODEL",
        make_test_grid(),
        step,
        spec=ComponentSpec(
            execution="host",
            lifecycle=LifecycleHooks(
                setup=lambda component, context: SetupResult(payload=1)
            ),
        ),
    )

    state = _coupler(component).run()
    assert state._component_state("MODEL").payload == 2

    state = _coupler(component).run(state)
    assert state._component_state("MODEL").payload is None


def test_setup_and_prefill_reject_wrong_result_types() -> None:
    setup_component = CallableComponent(
        "SETUP",
        make_test_grid(),
        lambda fields: {},
        spec=ComponentSpec(
            lifecycle=LifecycleHooks(setup=lambda component, context: {})  # type: ignore[arg-type, return-value]
        ),
    )
    with pytest.raises(ComponentError, match="setup must return SetupResult"):
        _coupler(setup_component).initial_state()

    prefill_component = CallableComponent(
        "PREFILL",
        make_test_grid(),
        lambda fields: {},
        spec=ComponentSpec(
            lifecycle=LifecycleHooks(prefill=lambda component, context: {})  # type: ignore[arg-type, return-value]
        ),
    )
    with pytest.raises(ComponentError, match="prefill must return PrefillResult"):
        _coupler(prefill_component).initial_state()


def test_prefill_result_populates_declared_runtime_fields() -> None:
    component = CallableComponent(
        "MODEL",
        make_test_grid(),
        lambda fields: {},
        spec=ComponentSpec(
            outputs=("value",),
            lifecycle=LifecycleHooks(
                prefill=lambda component, context: PrefillResult(
                    fields={"value": jnp.full((2, 2), 4.0)}
                )
            ),
        ),
    )

    state = _coupler(component).initial_state()
    np.testing.assert_array_equal(
        state.component("MODEL").field("value"), np.full((2, 2), 4.0)
    )


def test_step_result_rejects_shape_changes() -> None:
    component = CallableComponent(
        "MODEL",
        make_test_grid(),
        lambda fields: {"value": jnp.zeros((3, 3))},
        spec=ComponentSpec(outputs=("value",)),
    )

    with pytest.raises(ComponentError, match="invalid step field update"):
        _coupler(component).run()


def _prepared_prefill_binding(result: PrefillResult) -> Any:
    component = CallableComponent(
        "MODEL",
        make_test_grid(),
        lambda fields: {},
        spec=ComponentSpec(
            inputs=("incoming",),
            outputs=("outgoing",),
            initial_fields={"outgoing": 0.0},
            lifecycle=LifecycleHooks(prefill=lambda component, context: result),
        ),
    )
    coupler = Coupler(
        Clock(datetime(2000, 1, 1), 60.0, 1),
        components=(component,),
        run_order=(component.name,),
        runtime=RuntimeOptions(dtype=DTypePolicy(enable_x64=False)),
    )
    return coupler._ensure_prepared().components["MODEL"]


@pytest.mark.parametrize("store_name", ["received", "sent"])
def test_prefill_rejects_fields_absent_from_exchange_contract(
    store_name: str,
) -> None:
    result = PrefillResult(**{store_name: {"secret": 1.0}})
    binding = _prepared_prefill_binding(result)

    with pytest.raises(
        ComponentError,
        match=rf"prefill {store_name}.*'secret'.*exchange contract",
    ):
        binding._prefill_runtime_state_fields(
            dict(binding._data),
            {},
            {},
            ExchangeContract(receives=("incoming",), sends=("outgoing",)),
        )


@pytest.mark.parametrize("store_name", ["received", "sent"])
def test_prefill_rejects_non_grid_runtime_store_shapes(store_name: str) -> None:
    field_name = "incoming" if store_name == "received" else "outgoing"
    result = PrefillResult(**{store_name: {field_name: np.zeros((3, 3))}})
    binding = _prepared_prefill_binding(result)

    with pytest.raises(
        ComponentError,
        match=rf"prefill {store_name} field '{field_name}'.*shape \(3, 3\).*\(2, 2\)",
    ):
        binding._prefill_runtime_state_fields(
            dict(binding._data),
            {},
            {},
            ExchangeContract(receives=("incoming",), sends=("outgoing",)),
        )


def test_prefill_normalizes_all_store_scalars_and_dtypes_before_update() -> None:
    binding = _prepared_prefill_binding(
        PrefillResult(
            fields={"outgoing": np.int64(3)},
            received={"incoming": np.int64(4)},
            sent={"outgoing": np.int64(5)},
        )
    )
    data = dict(binding._data)
    received: dict[str, Any] = {}
    sent: dict[str, Any] = {}

    binding._prefill_runtime_state_fields(
        data,
        received,
        sent,
        ExchangeContract(receives=("incoming",), sends=("outgoing",)),
    )

    for mapping, field_name, expected in (
        (data, "outgoing", 3.0),
        (received, "incoming", 4.0),
        (sent, "outgoing", 5.0),
    ):
        value = mapping[field_name]
        assert value.shape == binding.grid.shape
        assert value.dtype == jnp.float32
        np.testing.assert_array_equal(value, np.full((2, 2), expected))


def test_prefill_rejects_nonnumeric_store_values_with_component_error() -> None:
    binding = _prepared_prefill_binding(PrefillResult(received={"incoming": object()}))

    with pytest.raises(
        ComponentError,
        match="prefill received field 'incoming'.*real numeric",
    ):
        binding._prefill_runtime_state_fields(
            dict(binding._data),
            {},
            {},
            ExchangeContract(receives=("incoming",), sends=("outgoing",)),
        )
