from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax

from vercor.pytree import PyTreeNodeMixin
from vercor._runtime.stores import FieldStore


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ComponentRuntimeState(PyTreeNodeMixin):
    """Immutable runtime state for one component."""

    pytree_children = ("data", "incoming", "outgoing", "runtime_payload")

    data: FieldStore
    incoming: FieldStore
    outgoing: FieldStore
    runtime_payload: Any | None = None

    def with_data(self, data: FieldStore) -> "ComponentRuntimeState":
        """Return this component state with replaced data."""

        return ComponentRuntimeState(
            data=data,
            incoming=self.incoming,
            outgoing=self.outgoing,
            runtime_payload=self.runtime_payload,
        )

    def with_incoming(self, incoming: FieldStore) -> "ComponentRuntimeState":
        """Return this component state with replaced incoming fields."""

        return ComponentRuntimeState(
            data=self.data,
            incoming=incoming,
            outgoing=self.outgoing,
            runtime_payload=self.runtime_payload,
        )

    def with_outgoing(self, outgoing: FieldStore) -> "ComponentRuntimeState":
        """Return this component state with replaced outgoing fields."""

        return ComponentRuntimeState(
            data=self.data,
            incoming=self.incoming,
            outgoing=outgoing,
            runtime_payload=self.runtime_payload,
        )

    def with_runtime_payload(
        self, runtime_payload: Any | None
    ) -> "ComponentRuntimeState":
        """Return this component state with replaced runtime payload."""

        return ComponentRuntimeState(
            data=self.data,
            incoming=self.incoming,
            outgoing=self.outgoing,
            runtime_payload=runtime_payload,
        )


__all__ = ["ComponentRuntimeState"]
