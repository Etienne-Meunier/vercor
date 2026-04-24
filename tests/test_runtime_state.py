from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tests.assertions import assert_allclose_compact
from vercor.runtime import (
    RuntimeComponentState,
    RuntimeCouplerState,
    RuntimeFieldStore,
)


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
