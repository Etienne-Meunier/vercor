from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
import importlib
import importlib.util
from pathlib import Path
from typing import Any, cast

import h5netcdf
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

import vercor.components as components_module
import vercor.components.base as base_module
import vercor.components.contracts as contracts_module
import vercor.components.data as data_module
import vercor.components.host as host_module
from vercor.components._contracts import merge_component_outputs
from vercor.components._runtime_fields import (
    apply_step_result,
    has_runtime_field,
    prefill_runtime_fields,
    runtime_field,
    runtime_field_or,
    runtime_field_or_zeros_like,
    runtime_fields,
    with_runtime_fields,
)
from vercor.components._runtime_validation import require_runtime_fields
from vercor.components.runtime_execution import step_component_runtime_state
from tests._coverage_support import DummyComponent, make_test_grid
from tests._runtime_helpers import run_scanned_coupler
from tests.assertions import assert_allclose_compact
from vercor.forcing_data import read_forcing
from vercor.setups._data.era5_atmosphere import make_era5_atmosphere
from vercor.clock import Clock
from vercor.components.contexts import SetupContext, StepContext
from vercor.coupler import Coupler
from vercor.exceptions import ComponentError, CouplerError
from vercor.output._runtime import write_runtime_component_view_to_netcdf
from vercor._runtime.contracts import ExchangeContract
from vercor._runtime.component_state import create_runtime_component_state
from vercor._runtime.field_transfer import (
    receive_runtime_fields,
    send_runtime_fields,
)
from vercor._runtime.state import ComponentRuntimeState
from vercor._runtime.stores import FieldStore
from vercor._runtime.validation import (
    check_not_empty_import_export_lists,
    validate_exchange_fields_declared,
    validate_component_runtime_contract_fields,
)
from vercor._runtime.time import scalar_runtime_step_info
from vercor.state import ComponentState
from vercor.settings import Settings
from vercor.types import RuntimeArray


def _step_runtime_state(
    component: base_module.Component,
    component_state: ComponentRuntimeState,
    context: StepContext,
    *,
    allow_host_runtime: bool = False,
) -> ComponentRuntimeState:
    return step_component_runtime_state(
        component,
        component_state,
        context,
        allow_host_runtime=allow_host_runtime,
    )


class _RuntimeOnlyComponent(base_module.Component):
    def step(
        self,
        fields: Mapping[str, RuntimeArray],
        context: StepContext,
        payload: Any | None = None,
    ) -> Mapping[str, RuntimeArray]:
        _ = payload
        return {"temperature": fields["temperature"] + context.dt_seconds}


class _MissingSetupComponent(base_module.Component):
    def __init__(self) -> None:
        pass

    def step(
        self,
        fields: Mapping[str, RuntimeArray],
        context: StepContext,
        payload: Any | None = None,
    ) -> Mapping[str, RuntimeArray]:
        _ = fields, context, payload
        return {}


class _HostStepOnlyComponent(host_module.HostComponent):
    def step(
        self,
        fields: Mapping[str, RuntimeArray],
        context: StepContext,
        payload: Any | None = None,
    ) -> Mapping[str, RuntimeArray]:
        _ = fields, context, payload
        return {}


def test_component_runtime_execution_policy_helpers_detect_host_components() -> None:
    assert importlib.util.find_spec("vercor.components.runtime_execution") is not None
    runtime_execution = cast(
        Any,
        importlib.import_module("vercor.components.runtime_execution"),
    )
    pure_component = _RuntimeOnlyComponent(name="ATM", grid=make_test_grid())
    host_component = _HostStepOnlyComponent(name="OCN", grid=make_test_grid())

    assert not hasattr(runtime_execution, "component_requires_host_runtime")
    assert runtime_execution.host_component_names(
        {"ATM": pure_component, "OCN": host_component}
    ) == ["OCN"]


def test_component_runtime_execution_policy_steps_selected_runtime_path() -> None:
    assert importlib.util.find_spec("vercor.components.runtime_execution") is not None
    runtime_execution = cast(
        Any,
        importlib.import_module("vercor.components.runtime_execution"),
    )

    class PureMarkerComponent(base_module.Component):
        def step(
            self,
            fields: Mapping[str, RuntimeArray],
            context: StepContext,
            payload: Any | None = None,
        ) -> Mapping[str, RuntimeArray]:
            _ = context, payload
            return {"marker": fields["marker"] + 1.0}

    class HostMarkerComponent(host_module.HostComponent):
        def step(
            self,
            fields: Mapping[str, RuntimeArray],
            context: StepContext,
            payload: Any | None = None,
        ) -> Mapping[str, RuntimeArray]:
            _ = context, payload
            return {"marker": fields["marker"] + 2.0}

    grid = make_test_grid()
    context = StepContext(dt_seconds=1.0, settings=Settings())
    state = ComponentRuntimeState(
        fields=FieldStore.from_mapping({"marker": jnp.asarray(0.0)}),
        received=FieldStore.empty(),
        sent=FieldStore.empty(),
    )

    pure_state = runtime_execution.step_component_runtime_state(
        PureMarkerComponent(name="ATM", grid=grid),
        state,
        context,
        allow_host_runtime=False,
    )
    host_state = runtime_execution.step_component_runtime_state(
        HostMarkerComponent(name="OCN", grid=grid),
        state,
        context,
        allow_host_runtime=True,
    )

    assert_allclose_compact(pure_state.fields.get("marker"), np.asarray(1.0))
    assert_allclose_compact(host_state.fields.get("marker"), np.asarray(2.0))
    with pytest.raises(ComponentError, match="host-backed"):
        runtime_execution.step_component_runtime_state(
            HostMarkerComponent(name="LND", grid=grid),
            state,
            context,
            allow_host_runtime=False,
        )


@pytest.mark.fast_always
def test_active_component_requires_explicit_runtime_step() -> None:
    class MissingRuntimeStep(base_module.Component):
        pass

    component = MissingRuntimeStep(name="ATM", grid=make_test_grid())
    state = ComponentRuntimeState(
        fields=FieldStore.empty(),
        received=FieldStore.empty(),
        sent=FieldStore.empty(),
    )

    with pytest.raises(NotImplementedError, match="step"):
        _step_runtime_state(
            component,
            state,
            StepContext(dt_seconds=1.0, settings=Settings()),
        )


@pytest.mark.fast_always
def test_host_runtime_component_requires_explicit_host_step() -> None:
    class MissingHostStep(host_module.HostComponent):
        pass

    component = MissingHostStep(name="ATM", grid=make_test_grid())
    state = ComponentRuntimeState(
        fields=FieldStore.empty(),
        received=FieldStore.empty(),
        sent=FieldStore.empty(),
    )

    with pytest.raises(ComponentError, match="must implement step"):
        _step_runtime_state(
            component,
            state,
            StepContext(dt_seconds=1.0, settings=Settings()),
            allow_host_runtime=True,
        )


@pytest.mark.fast_always
def test_data_component_uses_explicit_noop_runtime_step() -> None:
    class StaticForcingComponent(data_module.DataComponent):
        pass

    grid = make_test_grid(name="data")
    component = StaticForcingComponent(name="OCN", grid=grid)
    component._data["sea_surface_temperature"] = jnp.full(grid.shape, 280.0)
    contract = ExchangeContract(sends=("sea_surface_temperature",))
    state = create_runtime_component_state(
        component,
        prefill_missing=True,
        contract=contract,
    )

    stepped = _step_runtime_state(
        component,
        state,
        StepContext(dt_seconds=60.0, settings=Settings()),
    )

    assert stepped.fields is state.fields
    sent = send_runtime_fields(component, stepped, contract=contract)
    assert_allclose_compact(
        sent.sent.get("sea_surface_temperature"),
        np.full(grid.shape, 280.0),
    )


@pytest.mark.fast_always
def test_data_component_seeds_canonical_fields() -> None:
    grid = make_test_grid(name="factory-data")
    component = data_module.DataComponent.from_fields(
        name="OBS",
        grid=grid,
        fields={"temperature": jnp.full(grid.shape, 281.0)},
    )

    assert isinstance(component, data_module.DataComponent)
    assert_allclose_compact(
        component._data["temperature"],
        np.full(grid.shape, 281.0),
    )


@pytest.mark.fast_always
def test_data_component_from_fields_accepts_lifecycle_hooks() -> None:
    grid = make_test_grid(name="data-facade-hooks")
    calls: list[str] = []

    def initialize(component: base_module.Component, context: SetupContext) -> None:
        calls.append(f"initialize:{component.name}:{context.dt_seconds}")
        component.seed_field("temperature", 280.0)

    def create_runtime_payload(component: base_module.Component) -> dict[str, str]:
        calls.append(f"payload:{component.name}")
        return {"component": component.name}

    def prefill_runtime_state_fields(
        component: base_module.Component,
        context: contracts_module.PrefillContext,
    ) -> contracts_module.PrefillResult:
        _ = context
        calls.append(f"prefill:{component.name}")
        return contracts_module.PrefillResult(
            fields={"humidity": jnp.full(component.grid.shape, 0.5)}
        )

    def validate_runtime_state(
        component: base_module.Component,
        context: contracts_module.ValidationContext,
    ) -> None:
        calls.append(
            f"validate:{component.name}:{'humidity' in context.state.fields()}"
        )

    component = data_module.DataComponent.from_fields(
        name="OBS",
        grid=grid,
        spec=contracts_module.ComponentSpec(
            lifecycle=contracts_module.LifecycleHooks(
                initialize=initialize,
                create_payload=create_runtime_payload,
                prefill=prefill_runtime_state_fields,
                validate=validate_runtime_state,
            ),
        ),
    )
    context = SetupContext(
        start=datetime(2000, 1, 1),
        dt_seconds=60.0,
        logger=cast(Any, None),
        settings=Settings(),
        run_order=("OBS",),
    )
    component.initialize(context)
    state = create_runtime_component_state(
        component,
        prefill_missing=True,
        contract=ExchangeContract(),
    )
    component._validate_runtime_state(state, ExchangeContract())

    assert calls == [
        "initialize:OBS:60.0",
        "prefill:OBS",
        "payload:OBS",
        "validate:OBS:True",
    ]
    assert_allclose_compact(component._data["temperature"], np.full(grid.shape, 280.0))
    assert_allclose_compact(state.fields.get("humidity"), np.full(grid.shape, 0.5))
    assert state.payload == {"component": "OBS"}


@pytest.mark.fast_always
def test_removed_wrapper_entrypoints_stay_absent() -> None:
    assert not hasattr(base_module.Component, "wrap")
    assert not hasattr(data_module.DataComponent, "wrap")
    assert not hasattr(host_module.HostComponent, "wrap")
    assert not hasattr(base_module, "make_data_component")
    assert not hasattr(base_module, "make_differentiable_component")
    assert not hasattr(base_module, "make_host_component")


@pytest.mark.fast_always
def test_from_fields_and_from_step_facade_expand_scalar_defaults() -> None:
    grid = make_test_grid(name="facade")

    data_component = data_module.DataComponent.from_fields(
        name="OBS",
        grid=grid,
        fields={"temperature": 281.0},
    )
    assert isinstance(data_component, data_module.DataComponent)
    assert_allclose_compact(
        data_component._data["temperature"],
        np.full(grid.shape, 281.0),
    )

    def step(
        fields: Mapping[str, RuntimeArray],
        context: StepContext,
        payload: Any | None,
    ) -> Mapping[str, RuntimeArray]:
        _ = payload
        return {
            "temperature": fields["temperature"] + fields["forcing"],
            "tendency": fields["tendency"] + context.dt_seconds,
        }

    component = base_module.Component.from_step(
        name="ATM",
        grid=grid,
        step=step,
        spec=contracts_module.ComponentSpec(
            outputs=("temperature", "tendency"),
            defaults={"temperature": 280.0, "forcing": 2.0},
        ),
    )
    state = create_runtime_component_state(
        component,
        prefill_missing=True,
        contract=ExchangeContract(),
    )

    component._validate_runtime_state(state, ExchangeContract())
    assert_allclose_compact(state.fields.get("temperature"), np.full(grid.shape, 280.0))
    assert_allclose_compact(state.fields.get("forcing"), np.full(grid.shape, 2.0))
    assert_allclose_compact(state.fields.get("tendency"), np.zeros(grid.shape))

    stepped = _step_runtime_state(
        component,
        state,
        StepContext(dt_seconds=3.0, settings=Settings()),
    )
    assert_allclose_compact(
        stepped.fields.get("temperature"),
        np.full(grid.shape, 282.0),
    )
    assert_allclose_compact(
        stepped.fields.get("tendency"),
        np.full(grid.shape, 3.0),
    )


@pytest.mark.fast_always
def test_data_component_from_fields_normalizes_author_fields_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid = make_test_grid(name="facade-normalize-once")
    field_authoring_module = importlib.import_module(
        "vercor.components._field_authoring"
    )
    real_normalize = field_authoring_module._normalize_author_field_values
    call_count = 0

    def counting_normalize(*args: Any, **kwargs: Any) -> dict[str, RuntimeArray] | None:
        nonlocal call_count
        call_count += 1
        return cast("dict[str, RuntimeArray] | None", real_normalize(*args, **kwargs))

    monkeypatch.setattr(
        field_authoring_module,
        "_normalize_author_field_values",
        counting_normalize,
    )

    component = data_module.DataComponent.from_fields(
        name="OBS",
        grid=grid,
        fields={"temperature": 281.0},
    )

    assert call_count == 1
    assert component.spec.outputs == ("temperature",)
    assert_allclose_compact(
        component._data["temperature"],
        np.full(grid.shape, 281.0),
    )


@pytest.mark.fast_always
def test_callable_facade_accepts_one_two_and_three_argument_steps() -> None:
    grid = make_test_grid(name="flex-step")

    def fields_only(fields: Mapping[str, RuntimeArray]) -> Mapping[str, RuntimeArray]:
        return {"temperature": fields["temperature"] + 1.0}

    def fields_and_context(
        fields: Mapping[str, RuntimeArray],
        context: StepContext,
    ) -> Mapping[str, RuntimeArray]:
        return {"temperature": fields["temperature"] + context.dt_seconds}

    def fields_context_payload(
        fields: Mapping[str, RuntimeArray],
        context: StepContext,
        payload: Any | None,
    ) -> Mapping[str, RuntimeArray]:
        assert isinstance(payload, Mapping)
        return {
            "temperature": (
                fields["temperature"] + context.dt_seconds + payload["offset"]
            )
        }

    components = (
        base_module.Component.from_step(
            name="ONE",
            grid=grid,
            step=fields_only,
            spec=contracts_module.ComponentSpec(
                outputs=("temperature",),
                defaults={"temperature": 280.0},
            ),
        ),
        base_module.Component.from_step(
            name="TWO",
            grid=grid,
            step=fields_and_context,
            spec=contracts_module.ComponentSpec(
                outputs=("temperature",),
                defaults={"temperature": 280.0},
            ),
        ),
        base_module.Component.from_step(
            name="THREE",
            grid=grid,
            step=fields_context_payload,
            payload={"offset": 3.0},
            spec=contracts_module.ComponentSpec(
                outputs=("temperature",),
                defaults={"temperature": 280.0},
            ),
        ),
    )

    for component, expected_temperature in zip(
        components,
        (281.0, 282.0, 285.0),
        strict=True,
    ):
        state = create_runtime_component_state(
            component,
            prefill_missing=True,
            contract=ExchangeContract(),
        )
        stepped = _step_runtime_state(
            component,
            state,
            StepContext(dt_seconds=2.0, settings=Settings()),
        )
        assert_allclose_compact(
            stepped.fields.get("temperature"),
            np.full(grid.shape, expected_temperature),
        )


@pytest.mark.fast_always
def test_callable_facade_rejects_unsupported_step_signature() -> None:
    grid = make_test_grid(name="bad-step")

    def too_many_arguments(
        fields: Mapping[str, RuntimeArray],
        context: StepContext,
        payload: Any | None,
        extra: object,
    ) -> Mapping[str, RuntimeArray]:
        _ = fields, context, payload, extra
        return {}

    with pytest.raises(
        ComponentError,
        match="step callable.*1, 2, or 3 positional arguments",
    ):
        base_module.Component.from_step(
            name="ATM",
            grid=grid,
            step=too_many_arguments,
        )


@pytest.mark.fast_always
def test_callable_facade_rejects_removed_field_seed_keyword() -> None:
    grid = make_test_grid(name="removed-field-seed")

    def step(fields: Mapping[str, RuntimeArray]) -> Mapping[str, RuntimeArray]:
        return {"temperature": fields["temperature"]}

    removed_keyword = "initial" + "_fields"
    with pytest.raises(TypeError, match=removed_keyword):
        cast(Any, base_module.Component.from_step)(
            name="ATM",
            grid=grid,
            step=step,
            **{removed_keyword: {"temperature": 280.0}},
            outputs=("temperature",),
        )


@pytest.mark.fast_always
def test_seed_declared_defaults_and_field_names_expose_author_state() -> None:
    grid = make_test_grid(name="declared-defaults")
    component = _RuntimeOnlyComponent(name="ATM", grid=grid)
    component.declare_fields(
        outputs=("temperature", "humidity"),
        defaults={"temperature": 280.0, "humidity": 0.5},
    )

    returned = component.seed_declared_defaults()

    assert returned is component
    assert component.field_names == ("temperature", "humidity")
    assert_allclose_compact(component._data["temperature"], np.full(grid.shape, 280.0))
    assert_allclose_compact(component._data["humidity"], np.full(grid.shape, 0.5))


@pytest.mark.fast_always
def test_base_initialize_seeds_declared_defaults() -> None:
    grid = make_test_grid(name="base-initialize")
    component = _RuntimeOnlyComponent(name="ATM", grid=grid)
    component.declare_fields(
        outputs=("temperature", "humidity"),
        defaults={"temperature": 280.0, "humidity": 0.5},
    )

    component.initialize(
        SetupContext(
            start=datetime(2000, 1, 1),
            dt_seconds=60.0,
            logger=cast(Any, None),
            settings=Settings(),
            run_order=("ATM",),
        )
    )

    assert component.field_names == ("temperature", "humidity")
    assert_allclose_compact(component._data["temperature"], np.full(grid.shape, 280.0))
    assert_allclose_compact(component._data["humidity"], np.full(grid.shape, 0.5))


@pytest.mark.fast_always
def test_data_component_import_policy_is_declared_at_component_boundary() -> None:
    component = data_module.DataComponent.from_fields(
        name="ATM",
        grid=make_test_grid(),
        import_policy=contracts_module.FieldImportPolicy(
            time_interpolation=True,
        ),
    )

    assert component.import_policy.time_interpolation
    assert not hasattr(component.spec, "import_policy")


@pytest.mark.fast_always
def test_grid_field_defaults_expands_default_value_and_overrides() -> None:
    grid = make_test_grid(name="grid-defaults")
    component = _RuntimeOnlyComponent(name="ATM", grid=grid)

    defaults = component.grid_field_defaults(
        ("temperature", "humidity", "pressure"),
        value=0.0,
        overrides={"temperature": 280.0, "humidity": np.full(grid.shape, 0.5)},
    )

    assert tuple(defaults) == ("temperature", "humidity", "pressure")
    assert_allclose_compact(defaults["temperature"], np.full(grid.shape, 280.0))
    assert_allclose_compact(defaults["humidity"], np.full(grid.shape, 0.5))
    assert_allclose_compact(defaults["pressure"], np.zeros(grid.shape))


@pytest.mark.fast_always
def test_apply_step_result_updates_fields_and_payload() -> None:
    grid = make_test_grid(name="apply-step-result")
    component = _RuntimeOnlyComponent(name="ATM", grid=grid)
    component.seed_field("temperature", 280.0)
    state = create_runtime_component_state(
        component,
        contract=ExchangeContract(),
    )

    updated = apply_step_result(
        component,
        state,
        contracts_module.StepResult(
            fields={"temperature": jnp.full(grid.shape, 281.0)},
            payload={"counter": 1},
        ),
    )

    assert_allclose_compact(
        updated.fields.get("temperature"), np.full(grid.shape, 281.0)
    )
    assert updated.payload == {"counter": 1}


@pytest.mark.fast_always
def test_data_component_seeding_updates_declared_outputs() -> None:
    grid = make_test_grid(name="data-outputs")
    component = data_module.DataComponent.from_fields(
        name="OBS",
        grid=grid,
        fields={"temperature": 281.0},
    )

    component.seed_field("humidity", 0.5)
    component.seed_fields({"pressure": 101325.0})

    assert component.spec.outputs == ("temperature", "humidity", "pressure")
    assert component.field_names == ("temperature", "humidity", "pressure")


@pytest.mark.fast_always
def test_merge_component_outputs_is_pure_and_preserves_contract_details() -> None:
    component_spec = contracts_module.ComponentSpec(
        inputs=("forcing",),
        outputs=("temperature",),
        defaults={"temperature": 280.0},
    )

    merged = merge_component_outputs(component_spec, ("humidity", "temperature"))

    assert merged is not component_spec
    assert component_spec.outputs == ("temperature",)
    assert merged.inputs == ("forcing",)
    assert merged.outputs == ("temperature", "humidity")
    assert merged.defaults == {"temperature": 280.0}


@pytest.mark.fast_always
def test_data_component_seeding_preserves_inputs_and_defaults() -> None:
    grid = make_test_grid(name="data-contract-preserve")
    component = data_module.DataComponent(name="DATA", grid=grid)
    component.declare_fields(
        inputs=("forcing",),
        defaults={"temperature": 280.0},
    )

    component.seed_fields({"humidity": 0.5})

    assert component.spec.inputs == ("forcing",)
    assert component.spec.outputs == ("humidity",)
    assert "temperature" in component.spec.defaults


@pytest.mark.fast_always
def test_constructor_lifecycle_hooks_are_owned_by_component_spec() -> None:
    grid = make_test_grid(name="lifecycle-container")
    events: list[str] = []

    def step(fields: Mapping[str, RuntimeArray]) -> Mapping[str, RuntimeArray]:
        return {"temperature": fields["temperature"] + 1.0}

    def initialize(component: Any, context: SetupContext) -> None:
        _ = context
        events.append(f"initialize:{component.name}")

    def create_runtime_payload(component: Any) -> dict[str, int]:
        events.append(f"payload:{component.name}")
        return {"counter": 1}

    def prefill(
        component: Any,
        context: contracts_module.PrefillContext,
    ) -> contracts_module.PrefillResult:
        events.append(f"prefill:{component.name}")
        data = dict(context.fields)
        prefill_runtime_fields(component, data, outputs=("temperature",))
        return contracts_module.PrefillResult(fields=data)

    def validate(
        component: Any,
        context: contracts_module.ValidationContext,
    ) -> None:
        events.append(f"validate:{component.name}")
        assert context.state.field("temperature").shape == component.grid.shape

    factories = (
        data_module.DataComponent.from_fields(
            name="DATA",
            grid=grid,
            spec=contracts_module.ComponentSpec(
                lifecycle=contracts_module.LifecycleHooks(
                    initialize=initialize,
                    create_payload=create_runtime_payload,
                    prefill=prefill,
                    validate=validate,
                ),
            ),
        ),
        base_module.Component.from_step(
            name="ATM",
            grid=grid,
            step=step,
            spec=contracts_module.ComponentSpec(
                lifecycle=contracts_module.LifecycleHooks(
                    initialize=initialize,
                    create_payload=create_runtime_payload,
                    prefill=prefill,
                    validate=validate,
                ),
            ),
        ),
        host_module.HostComponent.from_step(
            name="HOST",
            grid=grid,
            step=step,
            spec=contracts_module.ComponentSpec(
                lifecycle=contracts_module.LifecycleHooks(
                    initialize=initialize,
                    create_payload=create_runtime_payload,
                    prefill=prefill,
                    validate=validate,
                ),
            ),
        ),
        base_module.Component.from_step(
            name="DIRECT",
            grid=grid,
            step=step,
            spec=contracts_module.ComponentSpec(
                lifecycle=contracts_module.LifecycleHooks(
                    initialize=initialize,
                    create_payload=create_runtime_payload,
                    prefill=prefill,
                    validate=validate,
                ),
            ),
        ),
        host_module.HostComponent.from_step(
            name="DIRECT_HOST",
            grid=grid,
            step=step,
            spec=contracts_module.ComponentSpec(
                lifecycle=contracts_module.LifecycleHooks(
                    initialize=initialize,
                    create_payload=create_runtime_payload,
                    prefill=prefill,
                    validate=validate,
                ),
            ),
        ),
    )

    for component in factories:
        assert isinstance(component.spec.lifecycle, contracts_module.LifecycleHooks)
        assert not hasattr(component, "_lifecycle_hooks")
        assert not hasattr(component, "_initialize_hook")
        assert not hasattr(component, "_create_runtime_payload_hook")
        component.initialize(
            SetupContext(
                start=datetime(2000, 1, 1),
                dt_seconds=60.0,
                logger=cast(Any, None),
                settings=Settings(),
                run_order=(component.name,),
            )
        )
        state = create_runtime_component_state(
            component,
            prefill_missing=True,
            contract=ExchangeContract(),
        )
        component._validate_runtime_state(state, ExchangeContract())

    assert events == [
        "initialize:DATA",
        "prefill:DATA",
        "payload:DATA",
        "validate:DATA",
        "initialize:ATM",
        "prefill:ATM",
        "payload:ATM",
        "validate:ATM",
        "initialize:HOST",
        "prefill:HOST",
        "payload:HOST",
        "validate:HOST",
        "initialize:DIRECT",
        "prefill:DIRECT",
        "payload:DIRECT",
        "validate:DIRECT",
        "initialize:DIRECT_HOST",
        "prefill:DIRECT_HOST",
        "payload:DIRECT_HOST",
        "validate:DIRECT_HOST",
    ]


@pytest.mark.fast_always
def test_configure_requires_component_spec_and_updates_authoritative_lifecycle() -> (
    None
):
    grid = make_test_grid(name="configure-contract")
    events: list[str] = []

    component = base_module.Component.from_step(
        "MODEL",
        grid,
        lambda fields: fields,
        spec=contracts_module.ComponentSpec(
            lifecycle=contracts_module.LifecycleHooks(
                initialize=lambda owner, context: events.append("old")
            )
        ),
    )
    replacement = contracts_module.ComponentSpec(
        lifecycle=contracts_module.LifecycleHooks(
            initialize=lambda owner, context: events.append("new")
        )
    )

    with pytest.raises(ComponentError, match="spec.*ComponentSpec"):
        component.configure(cast(Any, object()))

    assert component.configure(replacement) is component
    component.initialize(
        SetupContext(
            start=datetime(2000, 1, 1),
            dt_seconds=60.0,
            run_order=("MODEL",),
            settings=Settings(),
            logger=cast(Any, None),
        )
    )

    assert component.spec is replacement
    assert events == ["new"]


@pytest.mark.fast_always
def test_host_configure_preserves_host_execution_and_other_spec_fields() -> None:
    grid = make_test_grid(name="host-configure-contract")
    hooks = contracts_module.LifecycleHooks(initialize=lambda owner, context: None)
    output = contracts_module.OutputConfig()
    requested = contracts_module.ComponentSpec(
        inputs=("forcing",),
        outputs=("temperature",),
        defaults={"temperature": 280.0},
        execution="jax",
        lifecycle=hooks,
        output=output,
    )
    component = _HostStepOnlyComponent(name="HOST", grid=grid)

    assert component.configure(requested) is component
    assert component.spec.execution == "host"
    assert component.spec.inputs == requested.inputs
    assert component.spec.outputs == requested.outputs
    assert component.spec.defaults == requested.defaults
    assert component.spec.lifecycle is hooks
    assert component.spec.output is output


@pytest.mark.fast_always
def test_seed_helpers_accept_scalar_author_values_and_expose_spec() -> None:
    grid = make_test_grid(name="scalar-seed")
    component = _RuntimeOnlyComponent(name="ATM", grid=grid)

    returned_component = component.declare_fields(
        inputs=("forcing",),
        outputs=("temperature",),
        defaults={"pressure": 101325.0},
    )
    assert returned_component is component
    assert component.spec.inputs == ("forcing",)
    assert component.spec.outputs == ("temperature",)
    assert not hasattr(component.spec, "required_fields")
    assert "pressure" in component.spec.defaults
    with pytest.raises(AttributeError):
        component.spec = contracts_module.ComponentSpec()  # type: ignore[misc]

    component.seed_field("temperature", 280.0)
    component.seed_fields({"humidity": 0.5, "forcing": jnp.ones(grid.shape)})
    state = create_runtime_component_state(
        component,
        prefill_missing=True,
        contract=ExchangeContract(),
    )

    assert_allclose_compact(state.fields.get("temperature"), np.full(grid.shape, 280.0))
    assert_allclose_compact(state.fields.get("humidity"), np.full(grid.shape, 0.5))
    assert_allclose_compact(state.fields.get("forcing"), np.ones(grid.shape))
    assert_allclose_compact(state.fields.get("pressure"), np.full(grid.shape, 101325.0))


@pytest.mark.fast_always
def test_seeded_component_arrays_follow_float32_policy_with_global_x64_enabled() -> (
    None
):
    grid = make_test_grid(name="seeded-policy")
    component = data_module.DataComponent.from_fields(
        name="DATA",
        grid=grid,
        fields={
            "temperature": jnp.asarray([[280.0, 281.0], [282.0, 283.0]]),
        },
        settings=Settings(enable_x64=False),
    )

    assert component._data["temperature"].dtype == jnp.float32


@pytest.mark.fast_always
def test_required_fields_declaration_api_is_removed() -> None:
    grid = make_test_grid(name="removed-required-fields")

    def step(fields: Mapping[str, RuntimeArray]) -> Mapping[str, RuntimeArray]:
        return {"temperature": fields["temperature"]}

    rejected_callables: tuple[tuple[Any, dict[str, Any]], ...] = (
        (contracts_module.ComponentSpec, {}),
        (
            base_module.Component.from_step,
            {"name": "ATM", "grid": grid, "step": step},
        ),
        (
            host_module.HostComponent.from_step,
            {"name": "HOST", "grid": grid, "step": step},
        ),
        (
            base_module.Component.from_step,
            {"name": "ATM", "grid": grid, "step": step},
        ),
        (
            host_module.HostComponent.from_step,
            {"name": "HOST", "grid": grid, "step": step},
        ),
    )
    for callable_factory, kwargs in rejected_callables:
        with pytest.raises(TypeError, match="required_fields"):
            cast(Any, callable_factory)(**kwargs, required_fields=("humidity",))

    component = _RuntimeOnlyComponent(name="ATM", grid=grid)
    with pytest.raises(TypeError, match="required_fields"):
        cast(Any, component.declare_fields)(required_fields=("humidity",))


@pytest.mark.fast_always
def test_from_step_inputs_validate_missing_fields_without_zero_prefill() -> None:
    grid = make_test_grid(name="facade-inputs")

    def step(
        fields: Mapping[str, RuntimeArray],
        context: StepContext,
        payload: Any | None,
    ) -> Mapping[str, RuntimeArray]:
        _ = context, payload
        return {"temperature": fields["temperature"] + fields["forcing"]}

    component = base_module.Component.from_step(
        name="ATM",
        grid=grid,
        step=step,
        spec=contracts_module.ComponentSpec(
            inputs=("forcing",),
            outputs=("temperature",),
            defaults={"temperature": 280.0},
        ),
    )
    state = create_runtime_component_state(
        component,
        prefill_missing=True,
        contract=ExchangeContract(),
    )

    with pytest.raises(
        CouplerError,
        match="Runtime missing required data field 'forcing' for component 'ATM'",
    ):
        component._validate_runtime_state(state, ExchangeContract())


@pytest.mark.fast_always
def test_host_runtime_component_from_step_uses_author_friendly_names() -> None:
    grid = make_test_grid(name="facade-host")

    def step(
        fields: Mapping[str, RuntimeArray],
        context: StepContext,
        payload: Any | None,
    ) -> Mapping[str, RuntimeArray]:
        _ = payload
        return {"temperature": fields["temperature"] + context.dt_seconds}

    component = host_module.HostComponent.from_step(
        name="HOST",
        grid=grid,
        step=step,
        spec=contracts_module.ComponentSpec(
            outputs=("temperature",),
            defaults={"temperature": 1.0},
        ),
    )
    state = create_runtime_component_state(
        component,
        prefill_missing=True,
        contract=ExchangeContract(),
    )

    stepped = _step_runtime_state(
        component,
        state,
        StepContext(dt_seconds=5.0, settings=Settings()),
        allow_host_runtime=True,
    )
    assert_allclose_compact(
        stepped.fields.get("temperature"),
        np.full(grid.shape, 6.0),
    )


@pytest.mark.fast_always
def test_subclasses_can_declare_fields_with_author_spec() -> None:
    grid = make_test_grid(name="declared")

    class DeclaredComponent(base_module.Component):
        def __init__(self, name: str, grid: Any) -> None:
            super().__init__(name, grid)
            self.declare_fields(
                inputs=("forcing",),
                outputs=("temperature",),
                defaults={"temperature": 280.0},
            )

        def step(
            self,
            fields: Mapping[str, RuntimeArray],
            context: StepContext,
            payload: Any | None = None,
        ) -> Mapping[str, RuntimeArray]:
            _ = context, payload
            return {"temperature": fields["temperature"] + fields["forcing"]}

    component = DeclaredComponent(name="ATM", grid=grid)
    missing_input_state = create_runtime_component_state(
        component,
        prefill_missing=True,
        contract=ExchangeContract(),
    )
    with pytest.raises(
        CouplerError,
        match="Runtime missing required data field 'forcing' for component 'ATM'",
    ):
        component._validate_runtime_state(missing_input_state, ExchangeContract())

    contract = ExchangeContract(receives=("forcing",))
    state = create_runtime_component_state(
        component,
        prefill_missing=True,
        contract=contract,
    )
    component._validate_runtime_state(state, contract)
    assert_allclose_compact(state.fields.get("temperature"), np.full(grid.shape, 280.0))
    assert_allclose_compact(state.fields.get("forcing"), np.zeros(grid.shape))


@pytest.mark.fast_always
def test_runtime_field_optional_helpers_return_defaults() -> None:
    grid = make_test_grid(name="field-defaults")
    component = _RuntimeOnlyComponent(name="ATM", grid=grid)
    component.seed_field("temperature", jnp.full(grid.shape, 280.0))
    state = create_runtime_component_state(
        component,
        contract=ExchangeContract(),
    )

    assert has_runtime_field(state, "temperature")
    assert not has_runtime_field(state, "missing")
    assert_allclose_compact(
        runtime_field_or(component, state, "temperature", 1.0),
        np.full(grid.shape, 280.0),
    )
    assert_allclose_compact(
        runtime_field_or(component, state, "missing", 2.0),
        np.full(grid.shape, 2.0),
    )
    assert_allclose_compact(
        runtime_field_or_zeros_like(component, state, "missing", "temperature"),
        np.zeros(grid.shape),
    )


@pytest.mark.fast_always
def test_callable_component_prefills_and_validates_declared_fields() -> None:
    grid = make_test_grid(name="facade-prefill")

    def step(
        fields: Mapping[str, RuntimeArray],
        context: StepContext,
        payload: Any | None,
    ) -> Mapping[str, RuntimeArray]:
        _ = payload
        return {
            "temperature": fields["temperature"] + fields["wind"] + context.dt_seconds,
            "wind": fields["wind"],
        }

    component = base_module.Component.from_step(
        name="ATM",
        grid=grid,
        step=step,
        spec=contracts_module.ComponentSpec(
            outputs=("temperature", "wind"),
            defaults={"temperature": jnp.full(grid.shape, 280.0)},
        ),
    )
    assert not hasattr(component, "_required_fields")
    assert not hasattr(component, "_prefill_fields")
    assert not hasattr(component, "_field_defaults")
    state = create_runtime_component_state(
        component,
        prefill_missing=True,
        contract=ExchangeContract(),
    )

    component._validate_runtime_state(state, ExchangeContract())
    assert_allclose_compact(state.fields.get("temperature"), np.full(grid.shape, 280.0))
    assert_allclose_compact(state.fields.get("wind"), np.zeros(grid.shape))

    stepped = _step_runtime_state(
        component,
        state,
        StepContext(dt_seconds=2.0, settings=Settings()),
    )
    assert_allclose_compact(
        stepped.fields.get("temperature"),
        np.full(grid.shape, 282.0),
    )


@pytest.mark.fast_always
def test_callable_component_reports_missing_declared_inputs() -> None:
    grid = make_test_grid(name="facade-required")

    def step(
        fields: Mapping[str, RuntimeArray],
        context: StepContext,
        payload: Any | None,
    ) -> Mapping[str, RuntimeArray]:
        _ = context, payload
        return {"temperature": fields["temperature"]}

    component = base_module.Component.from_step(
        name="ATM",
        grid=grid,
        step=step,
        spec=contracts_module.ComponentSpec(inputs=("temperature",)),
    )
    state = create_runtime_component_state(
        component,
        prefill_missing=True,
        contract=ExchangeContract(),
    )

    with pytest.raises(
        CouplerError,
        match="Runtime missing required data field 'temperature' for component 'ATM'",
    ):
        component._validate_runtime_state(state, ExchangeContract())


@pytest.mark.fast_always
def test_component_seed_fields_and_required_field_validator() -> None:
    grid = make_test_grid(name="seed-defaults")
    component = _RuntimeOnlyComponent(name="ATM", grid=grid)

    component.seed_field("temperature", 0.0)
    component.seed_fields({"u_velocity": 0.0, "v_velocity": 0.0})
    component.seed_field("humidity", 0.5)
    state = create_runtime_component_state(
        component,
        contract=ExchangeContract(),
    )

    require_runtime_fields(
        component,
        state,
        "temperature",
        "u_velocity",
        "v_velocity",
        "humidity",
    )
    assert_allclose_compact(state.fields.get("temperature"), np.zeros(grid.shape))
    assert_allclose_compact(state.fields.get("u_velocity"), np.zeros(grid.shape))
    assert_allclose_compact(state.fields.get("v_velocity"), np.zeros(grid.shape))
    assert_allclose_compact(state.fields.get("humidity"), np.full(grid.shape, 0.5))

    with pytest.raises(
        CouplerError,
        match="Runtime missing required data field 'missing' for component 'ATM'",
    ):
        require_runtime_fields(component, state, "missing")


@pytest.mark.fast_always
def test_required_field_validator_accepts_time_dependent_canonical_data() -> None:
    grid = make_test_grid(name="time-dependent-required")
    component = _RuntimeOnlyComponent(name="OCN", grid=grid)
    monthly_sst = jnp.zeros((12, *grid.shape), dtype=jnp.float64)
    state = ComponentRuntimeState(
        fields=FieldStore.from_mapping({"sea_surface_temperature": monthly_sst}),
        received=FieldStore.empty(),
        sent=FieldStore.empty(),
    )

    require_runtime_fields(component, state, "sea_surface_temperature")

    bad_state = ComponentRuntimeState(
        fields=FieldStore.from_mapping(
            {"bad_metadata": jnp.zeros((3,), dtype=jnp.float64)}
        ),
        received=FieldStore.empty(),
        sent=FieldStore.empty(),
    )
    with pytest.raises(
        CouplerError,
        match="bad_metadata.*canonical grid-field layout",
    ):
        require_runtime_fields(component, bad_state, "bad_metadata")


@pytest.mark.fast_always
def test_data_component_rejects_non_grid_fields_early() -> None:
    grid = make_test_grid(name="factory-layout")

    with pytest.raises(
        ComponentError,
        match="data field 'bad_metadata'.*canonical grid-field layout",
    ):
        data_module.DataComponent.from_fields(
            name="OBS",
            grid=grid,
            fields={"bad_metadata": jnp.zeros((3,), dtype=jnp.float64)},
        )


@pytest.mark.fast_always
def test_component_helpers_seed_and_update_runtime_fields() -> None:
    grid = make_test_grid(name="helper-fields")
    component = _RuntimeOnlyComponent(name="ATM", grid=grid)
    component.seed_field("temperature", jnp.ones(grid.shape))
    component.seed_fields({"humidity": jnp.full(grid.shape, 0.5)})
    state = create_runtime_component_state(
        component,
        contract=ExchangeContract(),
    )

    fields = runtime_fields(state)
    assert set(fields) == {"temperature", "humidity"}
    assert_allclose_compact(
        runtime_field(component, state, "humidity"),
        np.full(grid.shape, 0.5),
    )

    updated = with_runtime_fields(
        component,
        state,
        {"temperature": jnp.full(grid.shape, 284.0)},
    )

    assert_allclose_compact(
        updated.fields.get("temperature"),
        np.full(grid.shape, 284.0),
    )
    assert_allclose_compact(
        updated.fields.get("humidity"),
        np.full(grid.shape, 0.5),
    )

    base_source = Path("vercor/components/base.py").read_text(encoding="utf-8")
    runtime_fields_source = Path("vercor/components/_runtime_fields.py").read_text(
        encoding="utf-8"
    )
    assert "component_state.fields.to_mapping()" not in base_source
    assert "component_state.fields.replace_many(fields)" not in base_source
    assert "validate_runtime_component_data_field" not in base_source
    assert "component_state.fields.to_mapping()" in runtime_fields_source
    assert "component_state.fields.replace_many(fields)" in runtime_fields_source
    assert "from vercor._runtime.validation import" not in runtime_fields_source


@pytest.mark.fast_always
def test_public_runtime_field_mapping_and_membership_helpers_are_stable() -> None:
    grid = make_test_grid(name="public-runtime-fields")
    component = _RuntimeOnlyComponent(name="ATM", grid=grid)
    component.seed_fields(
        {
            "temperature": jnp.full(grid.shape, 280.0),
            "humidity": jnp.full(grid.shape, 0.25),
        }
    )
    state = create_runtime_component_state(
        component,
        contract=ExchangeContract(),
    )

    fields = runtime_fields(state)
    assert tuple(fields) == ("temperature", "humidity")
    assert has_runtime_field(state, "temperature")
    assert has_runtime_field(state, "humidity")
    assert not has_runtime_field(state, "missing")

    fields["temperature"] = jnp.zeros(grid.shape)
    assert_allclose_compact(
        state.fields.get("temperature"),
        np.full(grid.shape, 280.0),
    )


@pytest.mark.fast_always
def test_differentiable_component_applies_callable_field_updates() -> None:
    grid = make_test_grid(name="factory-active")

    def step(
        fields: Mapping[str, RuntimeArray],
        context: StepContext,
        payload: Any | None,
    ) -> Mapping[str, RuntimeArray]:
        assert payload is None
        return {"temperature": fields["temperature"] + context.dt_seconds}

    component = base_module.Component.from_step(
        name="ATM",
        grid=grid,
        step=step,
        spec=contracts_module.ComponentSpec(
            outputs=("temperature",),
            defaults={"temperature": jnp.ones(grid.shape)},
        ),
    )
    state = create_runtime_component_state(
        component,
        prefill_missing=True,
        contract=ExchangeContract(),
    )

    stepped = _step_runtime_state(
        component,
        state,
        StepContext(dt_seconds=3.0, settings=Settings()),
    )

    assert_allclose_compact(
        stepped.fields.get("temperature"),
        np.full(grid.shape, 4.0),
    )


@pytest.mark.fast_always
def test_callable_component_preserves_and_replaces_payload() -> None:
    grid = make_test_grid(name="factory-payload")

    def preserve_payload(
        fields: Mapping[str, RuntimeArray],
        context: StepContext,
        payload: Any | None,
    ) -> Mapping[str, RuntimeArray]:
        _ = context
        assert isinstance(payload, Mapping)
        return {"temperature": fields["temperature"] + payload["offset"]}

    preserve_component = base_module.Component.from_step(
        name="ATM",
        grid=grid,
        payload={"offset": jnp.asarray(2.0)},
        step=preserve_payload,
        spec=contracts_module.ComponentSpec(
            outputs=("temperature",),
            defaults={"temperature": jnp.ones(grid.shape)},
        ),
    )
    preserve_state = create_runtime_component_state(
        preserve_component,
        prefill_missing=True,
        contract=ExchangeContract(),
    )
    preserved = _step_runtime_state(
        preserve_component,
        preserve_state,
        StepContext(dt_seconds=1.0, settings=Settings()),
    )

    assert preserved.payload is preserve_state.payload
    assert_allclose_compact(
        preserved.fields.get("temperature"),
        np.full(grid.shape, 3.0),
    )

    def replace_payload(
        fields: Mapping[str, RuntimeArray],
        context: StepContext,
        payload: Any | None,
    ) -> contracts_module.StepResult:
        _ = context
        assert isinstance(payload, Mapping)
        return contracts_module.StepResult(
            fields={"temperature": fields["temperature"] + 1.0},
            payload={"offset": payload["offset"] + 1.0},
        )

    replace_component = base_module.Component.from_step(
        name="ATM",
        grid=grid,
        payload={"offset": jnp.asarray(2.0)},
        step=replace_payload,
        spec=contracts_module.ComponentSpec(
            outputs=("temperature",),
            defaults={"temperature": jnp.ones(grid.shape)},
        ),
    )
    replace_state = create_runtime_component_state(
        replace_component,
        prefill_missing=True,
        contract=ExchangeContract(),
    )
    replaced = _step_runtime_state(
        replace_component,
        replace_state,
        StepContext(dt_seconds=1.0, settings=Settings()),
    )

    assert replaced.payload is not replace_state.payload
    assert_allclose_compact(
        replaced.fields.get("temperature"),
        np.full(grid.shape, 2.0),
    )
    assert isinstance(replaced.payload, Mapping)
    assert_allclose_compact(replaced.payload["offset"], np.asarray(3.0))


@pytest.mark.fast_always
def test_callable_payload_default_can_be_overridden_by_lifecycle_hook() -> None:
    grid = make_test_grid(name="factory-payload-hook")

    def step(fields: Mapping[str, RuntimeArray]) -> Mapping[str, RuntimeArray]:
        return {"temperature": fields["temperature"] + 1.0}

    payload = {"offset": 2}
    component = base_module.Component.from_step(
        name="ATM",
        grid=grid,
        payload=payload,
        step=step,
        spec=contracts_module.ComponentSpec(
            outputs=("temperature",),
            defaults={"temperature": jnp.ones(grid.shape)},
        ),
    )

    assert component._create_runtime_payload() is payload

    def create_runtime_payload(owner: Any) -> dict[str, str]:
        return {"owner": owner.name}

    hooked_component = base_module.Component.from_step(
        name="HOOKED",
        grid=grid,
        payload=payload,
        step=step,
        spec=contracts_module.ComponentSpec(
            outputs=("temperature",),
            defaults={"temperature": jnp.ones(grid.shape)},
            lifecycle=contracts_module.LifecycleHooks(
                create_payload=create_runtime_payload
            ),
        ),
    )

    assert hooked_component._create_runtime_payload() == {"owner": "HOOKED"}


@pytest.mark.fast_always
def test_host_component_runs_through_coupler_host_runtime() -> None:
    grid = make_test_grid(name="factory-host")

    def step(
        fields: Mapping[str, RuntimeArray],
        context: StepContext,
        payload: Any | None,
    ) -> Mapping[str, RuntimeArray]:
        _ = payload
        return {"temperature": fields["temperature"] + context.dt_seconds}

    component = host_module.HostComponent.from_step(
        name="HOST",
        grid=grid,
        step=step,
        spec=contracts_module.ComponentSpec(
            outputs=("temperature",),
            defaults={"temperature": jnp.ones(grid.shape)},
        ),
    )
    coupler = Coupler(clock=Clock(start=datetime(2000, 1, 1), dt_seconds=5.0, steps=1))
    coupler.add_component(component)
    coupler.set_run_order(("HOST",))

    final_state = coupler.run()

    assert_allclose_compact(
        final_state._component_state("HOST").fields.get("temperature"),
        np.full(grid.shape, 6.0),
    )


@pytest.mark.fast_always
def test_callable_component_rejects_unseeded_field_updates() -> None:
    grid = make_test_grid(name="factory-missing")

    def step(
        fields: Mapping[str, RuntimeArray],
        context: StepContext,
        payload: Any | None,
    ) -> Mapping[str, RuntimeArray]:
        _ = fields, context, payload
        return {"created_during_step": jnp.zeros(grid.shape)}

    component = base_module.Component.from_step(
        name="ATM",
        grid=grid,
        step=step,
        spec=contracts_module.ComponentSpec(
            defaults={"temperature": jnp.ones(grid.shape)}
        ),
    )
    state = create_runtime_component_state(
        component,
        prefill_missing=True,
        contract=ExchangeContract(),
    )

    with pytest.raises(
        ComponentError,
        match="created_during_step.*missing from runtime data.*seed_field",
    ):
        _step_runtime_state(
            component,
            state,
            StepContext(dt_seconds=1.0, settings=Settings()),
        )


@pytest.mark.fast_always
def test_era5_atmosphere_uses_data_component_runtime_contract() -> None:

    assert callable(make_era5_atmosphere)
    assert issubclass(data_module.DataComponent, base_module.Component)


@pytest.mark.fast_always
def test_component_setup_validation_reports_missing_required_attributes() -> None:
    component = _MissingSetupComponent()
    contract = ExchangeContract(sends=("temperature",))

    with pytest.raises(
        ComponentError,
        match="missing required setup attribute.*name.*grid.*data.*settings",
    ):
        create_runtime_component_state(component, contract=contract)


@pytest.mark.fast_always
def test_coupler_register_validates_component_setup_before_name_lookup() -> None:
    coupler = Coupler(clock=Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1))

    with pytest.raises(
        ComponentError,
        match="missing required setup attribute.*name.*grid.*data.*settings",
    ):
        coupler.add_component(cast(Any, _MissingSetupComponent()))


@pytest.mark.fast_always
def test_coupler_initialize_validates_component_setup_before_precision_sync() -> None:
    coupler = Coupler(clock=Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1))
    coupler._components["ATM"] = cast(Any, _MissingSetupComponent())
    coupler.set_run_order(("ATM",))

    with pytest.raises(
        ComponentError,
        match="missing required setup attribute.*name.*grid.*data.*settings",
    ):
        coupler._initialize_runtime()


@pytest.mark.fast_always
def test_initialize_storage_mutation_is_reported_by_setup_validation() -> None:
    class InvalidStorageComponent(_RuntimeOnlyComponent):
        def initialize(self, context: SetupContext) -> None:
            _ = context
            self._data = cast(Any, object())

    component = InvalidStorageComponent(
        name="MODEL",
        grid=make_test_grid(name="initialize-storage-mutation"),
    )
    coupler = Coupler(
        Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(component,),
        run_order=("MODEL",),
    )

    with pytest.raises(ComponentError, match="invalid setup attribute '_data'"):
        coupler.initial_state()


@pytest.mark.fast_always
def test_component_data_layout_validation_accepts_canonical_grid_fields() -> None:
    grid = make_test_grid(
        name="layout",
        longitude=np.asarray([0.0, 1.0, 2.0]),
        latitude=np.asarray([-1.0, 1.0]),
    )
    component = DummyComponent(name="ATM", grid=grid)
    component._data = {
        "snapshot_2d": jnp.zeros(grid.shape, dtype=jnp.float64),
        "time_surface_3d": jnp.zeros((12, *grid.shape), dtype=jnp.float64),
        "level_snapshot_3d": jnp.zeros((4, *grid.shape), dtype=jnp.float64),
        "time_level_4d": jnp.zeros((12, 4, *grid.shape), dtype=jnp.float64),
    }

    state = create_runtime_component_state(component, contract=ExchangeContract())

    assert state.fields.get("snapshot_2d").shape == grid.shape
    assert state.fields.get("time_surface_3d").shape == (12, *grid.shape)
    assert state.fields.get("level_snapshot_3d").shape == (4, *grid.shape)
    assert state.fields.get("time_level_4d").shape == (12, 4, *grid.shape)


@pytest.mark.fast_always
def test_component_data_layout_validation_rejects_non_grid_data_fields() -> None:
    grid = make_test_grid(
        name="layout",
        longitude=np.asarray([0.0, 1.0, 2.0]),
        latitude=np.asarray([-1.0, 1.0]),
    )
    component = DummyComponent(name="ATM", grid=grid)
    component._data = {
        "noncanonical_monthly_temperature": jnp.zeros((3, 2, 12), dtype=jnp.float64),
        "hyai": jnp.zeros((4,), dtype=jnp.float64),
    }

    with pytest.raises(
        ComponentError,
        match=(
            "Component 'ATM' data field 'noncanonical_monthly_temperature'.*"
            r"shape \(3, 2, 12\).*canonical.*"
            r"\(nTime, nLat, nLon\)"
        ),
    ):
        create_runtime_component_state(component, contract=ExchangeContract())


@pytest.mark.fast_always
def test_host_component_rejects_scanned_runtime_with_clear_error() -> None:
    grid = make_test_grid(name="host")
    component = _HostStepOnlyComponent(name="ATM", grid=grid)
    component._data["temperature"] = jnp.ones(grid.shape)
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=3600.0, steps=1),
        components=(component,),
        run_order=("ATM",),
    )
    state = coupler.initial_state()

    with pytest.raises(ComponentError, match="host-backed.*Coupler.run"):
        run_scanned_coupler(coupler, state)


@pytest.mark.fast_always
def test_removed_component_api_stays_absent() -> None:
    component = DummyComponent(name="ATM", grid=make_test_grid())

    assert not hasattr(components_module, "Shared")
    assert not hasattr(components_module, "TimedNamedArray")
    assert not hasattr(components_module, "ComponentInitContext")
    assert not hasattr(components_module, "RuntimeStepContext")
    assert not hasattr(base_module, "Shared")
    assert not hasattr(base_module, "TimedNamedArray")
    assert not hasattr(base_module, "write_shared_to_netcdf")
    assert not hasattr(base_module, "write_runtime_component_to_netcdf")
    assert not hasattr(base_module, "write_runtime_component_view_to_netcdf")
    assert not hasattr(base_module, "ComponentForcingData")
    assert not hasattr(components_module, "ComponentForcingData")
    assert not hasattr(components_module, "Atmosphere")
    assert not hasattr(components_module, "Ocean")
    assert not hasattr(components_module, "SeaIce")
    assert not hasattr(components_module, "Land")
    assert not hasattr(components_module, "ERA5Atmosphere")
    assert not hasattr(components_module, "ERA5Ocean")
    assert not hasattr(components_module, "ERAInterimOcean")
    assert not hasattr(components_module, "ERA5Land")
    assert not hasattr(components_module, "JCMLand")
    assert not hasattr(components_module, "JAXGCM")
    assert not hasattr(components_module, "VerosGCM")
    assert not hasattr(components_module, "CAMulatorGCM")
    assert not hasattr(components_module, "CAMulatorLand")
    assert not hasattr(components_module, "write_runtime_component_to_netcdf")
    assert not hasattr(components_module, "write_runtime_component_view_to_netcdf")
    assert not hasattr(component, "incoming_fields")
    assert not hasattr(component, "outgoing_fields")
    assert not hasattr(component, "commit_runtime_state")
    assert not hasattr(component, "merge_incoming_outgoing_fields")
    assert not hasattr(component, "get")
    assert hasattr(component, "step")
    assert not hasattr(component, "step_runtime_state")
    assert not hasattr(component, "to_runtime_component_state")
    assert not hasattr(component, "receive_runtime_fields")
    assert not hasattr(component, "send_runtime_fields")
    assert not hasattr(component, "check_not_empty_import_export_lists")
    assert not hasattr(component, "check_valid_exchange_field_names")
    assert not hasattr(component, "_validate_runtime_grid_data_field")
    assert not hasattr(component, "_sync_data_from_runtime_state")


def test_runtime_state_creation_receive_and_send() -> None:
    grid = make_test_grid(name="atm")
    component = _RuntimeOnlyComponent(name="ATM", grid=grid)
    contract = ExchangeContract(
        receives=("temperature",),
        sends=("sensible_heat_flux",),
    )
    component._data["sensible_heat_flux"] = jnp.full(grid.shape, 2.0)

    state = create_runtime_component_state(
        component,
        prefill_missing=True,
        contract=contract,
    )
    assert set(state.received.field_names) == {"temperature"}
    assert set(state.sent.field_names) == {"sensible_heat_flux"}
    assert isinstance(state.received.get("temperature"), jax.Array)

    incoming = state.received.set("temperature", jnp.full(grid.shape, 5.0))
    state = receive_runtime_fields(
        state.with_received(incoming),
        contract,
    )
    assert_allclose_compact(state.fields.get("temperature"), np.full(grid.shape, 5.0))

    stepped = _step_runtime_state(
        component,
        state,
        StepContext(
            dt_seconds=3.0,
            settings=Settings(),
        ),
    )
    assert_allclose_compact(stepped.fields.get("temperature"), np.full(grid.shape, 8.0))

    sent = send_runtime_fields(component, stepped, contract=contract)
    assert_allclose_compact(
        sent.sent.get("sensible_heat_flux"),
        np.full(grid.shape, 2.0),
    )


def test_component_validation_and_runtime_receive_delegate() -> None:
    component = DummyComponent(name="ATM", grid=make_test_grid())

    with pytest.raises(ComponentError, match="no runtime fields defined"):
        check_not_empty_import_export_lists(component, ExchangeContract())

    import_only = ExchangeContract(receives=("temperature",))
    check_not_empty_import_export_lists(component, import_only)

    overlapping = ExchangeContract(
        receives=("temperature",),
        sends=("temperature",),
    )
    check_not_empty_import_export_lists(component, overlapping)

    invalid = ExchangeContract(
        receives=("temperature",),
        sends=("not_supported",),
    )
    with pytest.raises(ComponentError, match="not_supported.*ATM.*not declared"):
        validate_exchange_fields_declared(component, invalid)

    contract = ExchangeContract(
        receives=("temperature",),
        sends=("sensible_heat_flux",),
    )
    state = create_runtime_component_state(
        component,
        prefill_missing=True,
        contract=contract,
    )
    state = state.with_received(
        state.received.set("temperature", np.ones(component.grid.shape))
    )
    received = receive_runtime_fields(state, contract)
    assert_allclose_compact(
        received.fields.get("temperature"), np.ones(component.grid.shape)
    )


def test_runtime_validation_uses_component_grid_shape_without_shape_argument() -> None:
    grid = make_test_grid(
        name="atm",
        longitude=np.asarray([0.0, 1.0, 2.0]),
        latitude=np.asarray([-1.0, 1.0]),
    )
    component = DummyComponent(name="ATM", grid=grid)
    contract = ExchangeContract(
        receives=("temperature",),
        sends=("sensible_heat_flux",),
    )
    valid_state = ComponentRuntimeState(
        fields=FieldStore.from_mapping(
            {
                "temperature": jnp.ones(grid.shape),
                "sensible_heat_flux": jnp.zeros(grid.shape),
            }
        ),
        received=FieldStore.from_mapping({"temperature": jnp.ones(grid.shape)}),
        sent=FieldStore.from_mapping({"sensible_heat_flux": jnp.zeros(grid.shape)}),
    )

    validate_component_runtime_contract_fields(component, valid_state, contract)
    component._validate_runtime_state(valid_state, contract)

    bad_state = valid_state.with_received(
        FieldStore.from_mapping({"temperature": jnp.ones((1, 3))})
    )
    with pytest.raises(
        CouplerError,
        match=r"has shape \(1, 3\), expected \(2, 3\)",
    ):
        validate_component_runtime_contract_fields(component, bad_state, contract)


def test_send_runtime_fields_updates_outgoing_store() -> None:
    grid = make_test_grid()
    component = DummyComponent(name="ATM", grid=grid)
    timestamp = datetime(2000, 1, 1)
    contract = ExchangeContract(sends=("temperature",))
    component._data["temperature"] = jnp.full(grid.shape, 1.0)

    component_state = send_runtime_fields(
        component,
        create_runtime_component_state(component, contract=ExchangeContract()),
        contract=contract,
    )
    assert_allclose_compact(
        component_state.sent.get("temperature"),
        np.full(grid.shape, 1.0),
    )
    assert isinstance(component_state.sent.get("temperature"), jax.Array)

    runtime_coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=3600.0, steps=1)
    )
    monthly = jnp.zeros((12, *grid.shape), dtype=jnp.float64)
    monthly = monthly.at[0].set(jnp.asarray([[1.0, 2.0], [3.0, 4.0]]))
    component.configure(
        contracts_module.ComponentSpec(
            outputs=("temperature",),
        )
    )
    component._import_policy = contracts_module.FieldImportPolicy(
        time_interpolation=True,
    )
    component._data["temperature"] = monthly
    component_state = send_runtime_fields(
        component,
        create_runtime_component_state(component, contract=ExchangeContract()),
        scalar_runtime_step_info(
            timestamp,
            runtime_coupler.clock,
            runtime_coupler.settings,
            model_year_seconds=runtime_coupler.runtime.model_year_seconds,
        ),
        contract=contract,
    )
    assert_allclose_compact(
        component_state.sent.get("temperature"),
        np.asarray(monthly[0]),
    )

    runtime_coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 3), dt_seconds=3600.0, steps=1)
    )
    daily = jnp.arange(5 * 2 * 2, dtype=jnp.float64).reshape((5, *grid.shape))
    component.configure(
        contracts_module.ComponentSpec(
            outputs=("temperature",),
        )
    )
    component._import_policy = contracts_module.FieldImportPolicy(
        daily_selection=True,
    )
    component._data["temperature"] = daily
    component_state = send_runtime_fields(
        component,
        create_runtime_component_state(component, contract=ExchangeContract()),
        scalar_runtime_step_info(
            runtime_coupler.clock.start,
            runtime_coupler.clock,
            runtime_coupler.settings,
            model_year_seconds=runtime_coupler.runtime.model_year_seconds,
        ),
        contract=contract,
    )
    assert_allclose_compact(
        component_state.sent.get("temperature"),
        np.asarray(daily[2]),
    )


def test_read_forcing_and_runtime_write_round_trip(
    tmp_path: Path,
) -> None:
    path = tmp_path / "forcing.nc"
    source = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    xr.Dataset({"foo": (("x", "y"), source)}).to_netcdf(path)

    data_files = {"sample": str(path)}
    normal_read = read_forcing(data_files, "foo", "sample")
    flipped_read = read_forcing(
        data_files,
        "foo",
        "sample",
        flip_y=True,
    )

    assert isinstance(normal_read, jax.Array)
    assert isinstance(flipped_read, jax.Array)
    assert_allclose_compact(normal_read, source.T)
    assert_allclose_compact(
        flipped_read,
        np.flip(source.T, axis=1),
    )

    with pytest.raises(KeyError, match="Provided 'where' key 'missing'"):
        read_forcing(data_files, "foo", "missing")

    with pytest.raises(KeyError, match="Variable 'bar' not found"):
        read_forcing(data_files, "bar", "sample")

    broken = tmp_path / "broken.nc"
    broken.write_text("not-a-netcdf-file", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Error reading variable 'foo'"):
        read_forcing({"broken": str(broken)}, "foo", "broken")

    state = ComponentRuntimeState(
        fields=FieldStore.empty(),
        received=FieldStore.from_mapping(
            {"temperature": jnp.asarray([[10.0, 11.0], [12.0, 13.0]])}
        ),
        sent=FieldStore.from_mapping(
            {"humidity": jnp.asarray([[0.1, 0.2], [0.3, 0.4]])}
        ),
    )
    output = tmp_path / "runtime.nc"

    write_runtime_component_view_to_netcdf(
        ComponentState._from_runtime("ATM", make_test_grid(), state),
        output,
        masks={"fmask_OCN_ATM_bilinear": jnp.ones((2, 2))},
    )

    with h5netcdf.File(output, "r") as dataset:
        assert_allclose_compact(
            np.asarray(dataset.variables["received_temperature"]),
            state.received.get("temperature"),
        )
        assert_allclose_compact(
            np.asarray(dataset.variables["sent_humidity"]),
            state.sent.get("humidity"),
        )
        assert_allclose_compact(
            np.asarray(dataset.variables["latitude"]),
            np.asarray([-1.0, 1.0]),
        )
        assert_allclose_compact(
            np.asarray(dataset.variables["longitude"]),
            np.asarray([0.0, 1.0]),
        )
        assert dataset.variables["received_temperature"].attrs["component"] == "ATM"
        assert (
            dataset.variables["received_temperature"].attrs["runtime_store"]
            == "received"
        )
        assert "fmask_OCN_ATM_bilinear" in dataset.variables

    view_output = tmp_path / "runtime-view.nc"
    write_runtime_component_view_to_netcdf(
        ComponentState._from_runtime(
            "ATM",
            make_test_grid(),
            state,
        ),
        view_output,
    )
    with h5netcdf.File(view_output, "r") as dataset:
        assert dataset.variables["sent_humidity"].attrs["component"] == "ATM"
