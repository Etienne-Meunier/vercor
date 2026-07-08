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

    pytree_children = ("fields", "received", "sent", "payload")

    fields: FieldStore
    received: FieldStore
    sent: FieldStore
    payload: Any | None = None

    def with_fields(self, fields: FieldStore) -> "ComponentRuntimeState":
        """Return this component state with replaced fields."""

        return ComponentRuntimeState(
            fields=fields,
            received=self.received,
            sent=self.sent,
            payload=self.payload,
        )

    def with_received(self, received: FieldStore) -> "ComponentRuntimeState":
        """Return this component state with replaced received fields."""

        return ComponentRuntimeState(
            fields=self.fields,
            received=received,
            sent=self.sent,
            payload=self.payload,
        )

    def with_sent(self, sent: FieldStore) -> "ComponentRuntimeState":
        """Return this component state with replaced sent fields."""

        return ComponentRuntimeState(
            fields=self.fields,
            received=self.received,
            sent=sent,
            payload=self.payload,
        )

    def with_payload(self, payload: Any | None) -> "ComponentRuntimeState":
        """Return this component state with replaced payload."""

        return ComponentRuntimeState(
            fields=self.fields,
            received=self.received,
            sent=self.sent,
            payload=payload,
        )


__all__ = ["ComponentRuntimeState"]
