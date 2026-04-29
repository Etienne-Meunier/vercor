from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tests._coverage_support import make_test_grid
from tests.assertions import assert_allclose_compact
from vercor.clock import ModelDateTime
from vercor.components.base import Component, ComponentInitContext
from vercor.settings import ComponentSettings
from vercor.components.external.jax_gcm import JAXGCMRuntimePayload
from vercor.runtime import (
    RuntimeComponentContract,
    RuntimeComponentState,
    RuntimeCouplerState,
    RuntimeFieldStore,
    RuntimeStepInfo,
)
from vercor.runtime_components import send_runtime_fields


class _RuntimeSendComponent(Component):
    def __init__(self, settings: ComponentSettings) -> None:
        super().__init__("DATA", make_test_grid(name="runtime-send"))
        self.settings = settings

    def initialize(self, context: ComponentInitContext) -> None:
        _ = context

    def step(
        self,
        dt: timedelta,
        time: datetime | ModelDateTime,
        coupler: Any,
    ) -> None:
        _ = dt, time, coupler


def test_runtime_module_does_not_own_component_specific_steps() -> None:
    runtime_source = Path("vercor/runtime.py").read_text(encoding="utf-8")
    runtime_components_source = Path("vercor/runtime_components.py").read_text(
        encoding="utf-8"
    )
    runtime_driver_source = Path("vercor/runtime_driver.py").read_text(encoding="utf-8")
    runtime_time_source = Path("vercor/runtime_time.py").read_text(encoding="utf-8")
    coupler_source = Path("vercor/coupler.py").read_text(encoding="utf-8")
    base_source = Path("vercor/components/base.py").read_text(encoding="utf-8")
    components_source = Path("vercor/components/__init__.py").read_text(
        encoding="utf-8"
    )
    forcing_data_source = Path("vercor/forcing_data.py").read_text(encoding="utf-8")
    flux_source = Path("vercor/fluxes/bulk_formula_cesm.py").read_text(encoding="utf-8")
    diagnostics_source = Path("vercor/diagnostics.py").read_text(encoding="utf-8")
    jax_gcm_source = Path("vercor/components/external/jax_gcm.py").read_text(
        encoding="utf-8"
    )
    veros_source = Path("vercor/components/external/veros_gcm.py").read_text(
        encoding="utf-8"
    )
    camulator_source = Path("vercor/components/external/camulator.py").read_text(
        encoding="utf-8"
    )
    camulator_land_source = Path("vercor/components/data/camulator_land.py").read_text(
        encoding="utf-8"
    )
    veros_runtime_settings_source = Path(
        "vercor/components/external/veros_runtime_settings.py"
    ).read_text(encoding="utf-8")
    windpp_source = Path("vercor/components/external/windpp.py").read_text(
        encoding="utf-8"
    )

    forbidden_component_markers = (
        "step_slab_component_state",
        "is_supported_differentiable_component",
        "receive_component_fields",
        "send_component_fields",
        "step_component_state",
        "JAXGCMRuntimePayload",
        "VerosGCM",
        "CAMulatorGCM",
        "CAMulatorLand",
    )
    for marker in forbidden_component_markers:
        assert marker not in runtime_source
    assert "import_fields" not in coupler_source
    assert 'hasattr(component, "step_host_runtime_state")' not in coupler_source
    assert "isinstance(component, HostRuntimeComponent)" not in coupler_source
    assert "isinstance(component, HostRuntimeComponent)" in runtime_driver_source
    assert "time is not None and isinstance" not in runtime_driver_source
    assert "def _step_runtime_component" not in coupler_source
    assert "def _runtime_step_info_from_times" not in coupler_source
    assert "def _runtime_daily_index" not in coupler_source
    assert "def _build_runtime_contracts" not in coupler_source
    assert "def build_runtime_contracts" in runtime_source
    assert not Path("vercor/runtime_contracts.py").exists()
    assert "def runtime_step_info_from_times" in runtime_time_source
    assert "def step_runtime_component_pure" in runtime_driver_source
    assert "def step_runtime_component_host_enabled" in runtime_driver_source
    assert "_sync_data_from_runtime_state" not in base_source
    assert "_fields2import" not in base_source
    assert "_fields2export" not in base_source
    assert "_fields2import" not in coupler_source
    assert "_fields2export" not in coupler_source
    assert "def to_runtime_component_state" not in base_source
    assert "def receive_runtime_fields" not in base_source
    assert "def send_runtime_fields" not in base_source
    assert "def check_not_empty_import_export_lists" not in base_source
    assert "def check_valid_exchange_field_names" not in base_source
    assert "ComponentForcingData" not in base_source
    assert "h5netcdf" not in base_source
    assert "import numpy" not in base_source
    assert "class ComponentForcingData" in forcing_data_source
    assert "ComponentForcingData" not in components_source
    assert "def create_runtime_component_state" in runtime_components_source
    assert "def receive_runtime_fields" in runtime_components_source
    assert "def send_runtime_fields" in runtime_components_source
    assert "def validate_component_runtime_state" in runtime_components_source
    assert "RuntimeComponentContract" in runtime_source
    assert 'def empty(cls) -> "RuntimeComponentContract"' not in runtime_source
    assert "RuntimeComponentContract.empty" not in coupler_source
    assert "RuntimeComponentContract.empty" not in runtime_driver_source
    assert "def build_runtime_contracts_for_components" not in runtime_source
    assert "build_runtime_contracts_for_components" not in coupler_source
    assert "RuntimeDispatchContext" in runtime_driver_source
    assert "dispatch_context: RuntimeDispatchContext" in runtime_driver_source
    assert "contracts.get(" not in runtime_driver_source
    assert "_runtime_contracts.get(" not in coupler_source
    assert "def subset(" not in runtime_source
    assert "def to_mapping(" not in runtime_source
    assert "def merge(" not in runtime_source
    assert "_runtime_contracts" in coupler_source
    assert "ComponentInitContext" in base_source
    assert "RuntimeStepContext" in base_source
    assert "component.initialize(self)" not in coupler_source
    assert "dt_seconds: float,\n        runtime_settings" not in base_source
    assert "def write_runtime_component_to_netcdf" not in base_source
    assert "write_runtime_component_to_netcdf" not in components_source
    assert "write_runtime_component_view_to_netcdf" not in components_source
    assert not Path("vercor/tools.py").exists()
    assert "class RuntimeComponentView" not in diagnostics_source
    assert "RuntimeComponentView =" not in diagnostics_source
    assert "RuntimeComponentView" in diagnostics_source
    assert 'hasattr(store, "field_names")' not in diagnostics_source
    assert "elif field_name in store" not in diagnostics_source
    assert "def runtime_contract" not in runtime_components_source
    assert "RuntimeComponentContract | None" not in runtime_components_source
    assert "def step_runtime_state" in jax_gcm_source
    assert "def step_host_runtime_state" in veros_source
    assert "def step_host_runtime_state" in camulator_source
    assert "def step_host_runtime_state" in camulator_land_source
    assert "load_camulator_forcing_context" in camulator_land_source
    assert "initialize_camulator" not in camulator_land_source
    assert "vercor.components.external.camulator import" not in camulator_land_source
    assert (
        "from vercor.components.external.veros_runtime_settings import *"
        not in veros_source
    )
    assert (
        "from vercor.components.external.veros_runtime_settings import configure_veros_runtime"
        in veros_source
    )
    assert veros_source.index("configure_veros_runtime()") < veros_source.index(
        "from veros.setups.global_4deg import GlobalFourDegreeSetup"
    )
    assert "def configure_veros_runtime" in veros_runtime_settings_source
    assert "def _step_host_runtime_state" not in base_source
    assert "_step_host_runtime_state" not in runtime_driver_source
    for source in (veros_source, camulator_source, camulator_land_source):
        signature = source.split("def step_host_runtime_state", 1)[1].split(") ->", 1)[
            0
        ]
        assert "coupler" not in signature
        assert "context" in signature
        assert "logger" not in signature
        assert "runtime_settings" not in signature
    assert "def step_runtime_state" not in veros_source
    assert "def step_runtime_state" not in camulator_source
    assert "def step_runtime_state" not in camulator_land_source
    assert "component_state.data.to_mapping()" not in veros_source
    assert "component_state.data.to_mapping()" not in camulator_source
    assert "component_state.data.to_mapping()" not in camulator_land_source
    assert "post_process_wind_artifacts_deprecated" not in windpp_source
    assert "old_flux_atmOcn" not in flux_source
    assert "new_flux_atmOcn" not in flux_source
    assert "def compute_ocean_surface_fluxes" in flux_source


def test_examples_use_coupler_runtime_component_view_factory() -> None:
    slab_driver_source = Path("examples/run_slab_driver.py").read_text(encoding="utf-8")
    data_driver_source = Path("examples/run_data_driver.py").read_text(encoding="utf-8")
    jcm_slab_source = Path("examples/run_jcm_with_slab.py").read_text(encoding="utf-8")

    for source in (slab_driver_source, data_driver_source, jcm_slab_source):
        assert "RuntimeComponentView.from_coupler_state" not in source
        assert "cpl.runtime_component_view(final_state," in source


def test_examples_import_concrete_components_directly() -> None:
    for path in Path("examples").glob("run_*.py"):
        source = path.read_text(encoding="utf-8")
        assert "from vercor.components import" not in source


def test_runtime_field_store_is_immutable_pytree() -> None:
    store = RuntimeFieldStore.from_mapping(
        {"temperature": jnp.asarray([[1.0, 2.0], [3.0, 4.0]])}
    )

    updated = store.set("temperature", store.get("temperature") + 1.0)

    assert store.field_names == ("temperature",)
    assert_allclose_compact(
        store.get("temperature"), np.asarray([[1.0, 2.0], [3.0, 4.0]])
    )
    assert_allclose_compact(
        updated.get("temperature"), np.asarray([[2.0, 3.0], [4.0, 5.0]])
    )

    leaves, treedef = jax.tree_util.tree_flatten(updated)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)

    assert isinstance(restored, RuntimeFieldStore)
    assert restored.field_names == ("temperature",)
    assert_allclose_compact(restored.get("temperature"), updated.get("temperature"))


def test_runtime_field_store_supports_jit_updates_and_mapping_roundtrip() -> None:
    store = RuntimeFieldStore.from_mapping(
        {
            "a": jnp.asarray([1.0, 2.0]),
            "b": jnp.asarray([3.0, 4.0]),
        }
    )

    def update(value: RuntimeFieldStore) -> RuntimeFieldStore:
        return value.set("a", value.get("a") * 2.0).set("b", value.get("b") + 1.0)

    updated = jax.jit(update)(store)

    assert updated.field_names == ("a", "b")
    assert_allclose_compact(updated.get("a"), np.asarray([2.0, 4.0]))
    assert_allclose_compact(updated.get("b"), np.asarray([4.0, 5.0]))


def test_runtime_component_and_coupler_state_are_pytrees() -> None:
    component = RuntimeComponentState(
        data=RuntimeFieldStore.from_mapping({"temperature": jnp.ones((2, 2))}),
        incoming=RuntimeFieldStore.from_mapping(
            {"sea_surface_temperature": jnp.zeros((2, 2))}
        ),
        outgoing=RuntimeFieldStore.from_mapping({"temperature": jnp.ones((2, 2))}),
    )
    assert not hasattr(component, "name")
    assert not hasattr(component, "fields_to_import")
    assert not hasattr(component, "fields_to_export")
    state = RuntimeCouplerState(
        component_names=("ATM",),
        components=(component,),
        fractional_masks=RuntimeFieldStore.from_mapping(
            {"OCN|ATM|bilinear": jnp.ones((2, 2))}
        ),
        binary_masks=RuntimeFieldStore.empty(),
    )

    def update(value: RuntimeCouplerState) -> RuntimeCouplerState:
        atm = value.get_component_state("ATM")
        atm = atm.with_data(
            atm.data.set("temperature", atm.data.get("temperature") + 2.0)
        )
        return value.set_component_state("ATM", atm)

    updated = jax.jit(update)(state)

    assert updated.component_names == ("ATM",)
    assert_allclose_compact(
        updated.get_component_state("ATM").data.get("temperature"),
        np.full((2, 2), 3.0),
    )


def test_runtime_component_state_preserves_optional_payload_under_jit() -> None:
    payload = JAXGCMRuntimePayload(
        jcm_state={"metadata": jnp.asarray(1.0)},
        forcing={"surface_temperature": jnp.asarray([[2.0, 3.0]])},
    )
    component = RuntimeComponentState(
        data=RuntimeFieldStore.from_mapping({"temperature": jnp.ones((1, 2))}),
        incoming=RuntimeFieldStore.empty(),
        outgoing=RuntimeFieldStore.empty(),
        runtime_payload=payload,
    )

    def update(value: RuntimeComponentState) -> RuntimeComponentState:
        runtime_payload = value.runtime_payload
        assert isinstance(runtime_payload, JAXGCMRuntimePayload)
        return value.with_runtime_payload(
            JAXGCMRuntimePayload(
                jcm_state={"metadata": runtime_payload.jcm_state["metadata"] + 1.0},
                forcing=runtime_payload.forcing,
            )
        )

    updated = jax.jit(update)(component)

    assert isinstance(updated.runtime_payload, JAXGCMRuntimePayload)
    assert_allclose_compact(
        updated.runtime_payload.jcm_state["metadata"],
        np.asarray(2.0),
    )


def test_runtime_send_applies_monthly_interpolation_under_jit_and_grad() -> None:
    component = _RuntimeSendComponent(ComponentSettings(apply_time_interpolation=True))
    contract = RuntimeComponentContract(exports=("temperature",))
    step_info = jax.tree_util.tree_map(
        lambda value: value[0],
        RuntimeStepInfo.from_sequences([0], [1], [0.75], [0.25], [0]),
    )
    forcing = jnp.zeros((2, 3, 12), dtype=jnp.float64)
    forcing = forcing.at[:, :, 0].set(4.0)
    forcing = forcing.at[:, :, 1].set(8.0)

    def send_loss(field: jax.Array) -> jax.Array:
        state = RuntimeComponentState(
            data=RuntimeFieldStore.from_mapping({"temperature": field}),
            incoming=RuntimeFieldStore.empty(),
            outgoing=RuntimeFieldStore.empty(),
        )
        sent = send_runtime_fields(component, state, step_info, contract=contract)
        return jnp.sum(sent.outgoing.get("temperature"))

    sent_state = jax.jit(
        lambda field: send_runtime_fields(
            component,
            RuntimeComponentState(
                data=RuntimeFieldStore.from_mapping({"temperature": field}),
                incoming=RuntimeFieldStore.empty(),
                outgoing=RuntimeFieldStore.empty(),
            ),
            step_info,
            contract=contract,
        )
    )(forcing)
    out = sent_state.outgoing.get("temperature")
    gradient = jax.grad(send_loss)(forcing)

    assert out.shape == (3, 2)
    assert_allclose_compact(out, np.full((3, 2), 5.0))
    assert_allclose_compact(gradient[:, :, 0], np.full((2, 3), 0.75))
    assert_allclose_compact(gradient[:, :, 1], np.full((2, 3), 0.25))
    assert_allclose_compact(gradient[:, :, 2:], np.zeros((2, 3, 10)))


def test_runtime_send_applies_daily_time_slice_under_jit_and_grad() -> None:
    component = _RuntimeSendComponent(ComponentSettings(get_field_time_slice=True))
    contract = RuntimeComponentContract(exports=("temperature",))
    step_info = jax.tree_util.tree_map(
        lambda value: value[0],
        RuntimeStepInfo.from_sequences([0], [1], [1.0], [0.0], [2]),
    )
    forcing = jnp.arange(5 * 2 * 2, dtype=jnp.float64).reshape((5, 2, 2))

    def send_loss(field: jax.Array) -> jax.Array:
        state = RuntimeComponentState(
            data=RuntimeFieldStore.from_mapping({"temperature": field}),
            incoming=RuntimeFieldStore.empty(),
            outgoing=RuntimeFieldStore.empty(),
        )
        sent = send_runtime_fields(component, state, step_info, contract=contract)
        return jnp.sum(sent.outgoing.get("temperature"))

    state = RuntimeComponentState(
        data=RuntimeFieldStore.from_mapping({"temperature": forcing}),
        incoming=RuntimeFieldStore.empty(),
        outgoing=RuntimeFieldStore.empty(),
    )
    sent_state = jax.jit(
        lambda value: send_runtime_fields(
            component,
            value,
            step_info,
            contract=contract,
        )
    )(state)
    gradient = jax.grad(send_loss)(forcing)

    assert_allclose_compact(
        sent_state.outgoing.get("temperature"), np.asarray(forcing[2])
    )
    assert_allclose_compact(gradient[2], np.ones((2, 2)))
    assert_allclose_compact(gradient[:2], np.zeros((2, 2, 2)))
    assert_allclose_compact(gradient[3:], np.zeros((2, 2, 2)))
