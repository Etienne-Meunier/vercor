from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax

from vercor.pytree import PyTreeNodeMixin
from vercor.runtime.stores import RuntimeFieldStore


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class RuntimeComponentState(PyTreeNodeMixin):
    """Immutable runtime state for one component."""

    pytree_children = ("data", "incoming", "outgoing", "runtime_payload")

    data: RuntimeFieldStore
    incoming: RuntimeFieldStore
    outgoing: RuntimeFieldStore
    runtime_payload: Any | None = None

    def with_data(self, data: RuntimeFieldStore) -> "RuntimeComponentState":
        """Return this component state with replaced data."""

        return RuntimeComponentState(
            data=data,
            incoming=self.incoming,
            outgoing=self.outgoing,
            runtime_payload=self.runtime_payload,
        )

    def with_incoming(self, incoming: RuntimeFieldStore) -> "RuntimeComponentState":
        """Return this component state with replaced incoming fields."""

        return RuntimeComponentState(
            data=self.data,
            incoming=incoming,
            outgoing=self.outgoing,
            runtime_payload=self.runtime_payload,
        )

    def with_outgoing(self, outgoing: RuntimeFieldStore) -> "RuntimeComponentState":
        """Return this component state with replaced outgoing fields."""

        return RuntimeComponentState(
            data=self.data,
            incoming=self.incoming,
            outgoing=outgoing,
            runtime_payload=self.runtime_payload,
        )

    def with_runtime_payload(
        self, runtime_payload: Any | None
    ) -> "RuntimeComponentState":
        """Return this component state with replaced runtime payload."""

        return RuntimeComponentState(
            data=self.data,
            incoming=self.incoming,
            outgoing=self.outgoing,
            runtime_payload=runtime_payload,
        )


__all__ = ["RuntimeComponentState"]
