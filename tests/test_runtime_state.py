from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from tests.assertions import assert_allclose_compact
from vercor.settings import ComponentSettings
from vercor.components.external.jax_gcm import JAXGCMRuntimePayload
from vercor.runtime import (
    RuntimeComponentState,
    RuntimeCouplerState,
    RuntimeFieldStore,
    RuntimeStepInfo,
    send_component_fields,
)


class _RuntimeSendComponent:
    def __init__(self, settings: ComponentSettings) -> None:
        self.settings = settings


def test_runtime_module_does_not_own_component_specific_steps() -> None:
    runtime_source = Path("vercor/runtime.py").read_text(encoding="utf-8")

    forbidden_component_markers = (
        "step_slab_component_state",
        "is_supported_differentiable_component",
        "JAXGCMRuntimePayload",
        "VerosGCM",
        "CAMulatorGCM",
        "CAMulatorLand",
    )
    for marker in forbidden_component_markers:
        assert marker not in runtime_source


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
        return value.set("a", value.get("a") * 2.0).merge(
            RuntimeFieldStore.from_mapping({"b": value.get("b") + 1.0})
        )

    updated = jax.jit(update)(store)

    assert updated.field_names == ("a", "b")
    assert_allclose_compact(updated.get("a"), np.asarray([2.0, 4.0]))
    assert_allclose_compact(updated.to_mapping()["b"], np.asarray([4.0, 5.0]))


def test_runtime_component_and_coupler_state_are_pytrees() -> None:
    component = RuntimeComponentState(
        name="ATM",
        data=RuntimeFieldStore.from_mapping({"temperature": jnp.ones((2, 2))}),
        incoming=RuntimeFieldStore.from_mapping(
            {"sea_surface_temperature": jnp.zeros((2, 2))}
        ),
        outgoing=RuntimeFieldStore.from_mapping({"temperature": jnp.ones((2, 2))}),
        fields_to_import=("sea_surface_temperature",),
        fields_to_export=("temperature",),
    )
    state = RuntimeCouplerState(
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
        return value.set_component_state(atm)

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
        name="ATM",
        data=RuntimeFieldStore.from_mapping({"temperature": jnp.ones((1, 2))}),
        incoming=RuntimeFieldStore.empty(),
        outgoing=RuntimeFieldStore.empty(),
        fields_to_import=(),
        fields_to_export=(),
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
    step_info = jax.tree_util.tree_map(
        lambda value: value[0],
        RuntimeStepInfo.from_sequences([0], [1], [0.75], [0.25], [0]),
    )
    forcing = jnp.zeros((2, 3, 12), dtype=jnp.float64)
    forcing = forcing.at[:, :, 0].set(4.0)
    forcing = forcing.at[:, :, 1].set(8.0)

    def send_loss(field: jax.Array) -> jax.Array:
        state = RuntimeComponentState(
            name="DATA",
            data=RuntimeFieldStore.from_mapping({"temperature": field}),
            incoming=RuntimeFieldStore.empty(),
            outgoing=RuntimeFieldStore.empty(),
            fields_to_import=(),
            fields_to_export=("temperature",),
        )
        sent = send_component_fields(state, component, step_info)
        return jnp.sum(sent.outgoing.get("temperature"))

    sent_state = jax.jit(
        lambda field: send_component_fields(
            RuntimeComponentState(
                name="DATA",
                data=RuntimeFieldStore.from_mapping({"temperature": field}),
                incoming=RuntimeFieldStore.empty(),
                outgoing=RuntimeFieldStore.empty(),
                fields_to_import=(),
                fields_to_export=("temperature",),
            ),
            component,
            step_info,
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
    step_info = jax.tree_util.tree_map(
        lambda value: value[0],
        RuntimeStepInfo.from_sequences([0], [1], [1.0], [0.0], [2]),
    )
    forcing = jnp.arange(5 * 2 * 2, dtype=jnp.float64).reshape((5, 2, 2))

    def send_loss(field: jax.Array) -> jax.Array:
        state = RuntimeComponentState(
            name="DATA",
            data=RuntimeFieldStore.from_mapping({"temperature": field}),
            incoming=RuntimeFieldStore.empty(),
            outgoing=RuntimeFieldStore.empty(),
            fields_to_import=(),
            fields_to_export=("temperature",),
        )
        sent = send_component_fields(state, component, step_info)
        return jnp.sum(sent.outgoing.get("temperature"))

    state = RuntimeComponentState(
        name="DATA",
        data=RuntimeFieldStore.from_mapping({"temperature": forcing}),
        incoming=RuntimeFieldStore.empty(),
        outgoing=RuntimeFieldStore.empty(),
        fields_to_import=(),
        fields_to_export=("temperature",),
    )
    sent_state = jax.jit(
        lambda value: send_component_fields(value, component, step_info)
    )(state)
    gradient = jax.grad(send_loss)(forcing)

    assert_allclose_compact(
        sent_state.outgoing.get("temperature"), np.asarray(forcing[2])
    )
    assert_allclose_compact(gradient[2], np.ones((2, 2)))
    assert_allclose_compact(gradient[:2], np.zeros((2, 2, 2)))
    assert_allclose_compact(gradient[3:], np.zeros((2, 2, 2)))
