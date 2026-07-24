from __future__ import annotations

from typing import Any, ClassVar, Self


class PyTreeNodeMixin:
    """Declarative base for immutable classes registered as JAX PyTrees."""

    pytree_children: ClassVar[tuple[str, ...]] = ()
    pytree_aux_data: ClassVar[tuple[str, ...]] = ()

    def tree_flatten(self) -> tuple[tuple[Any, ...], tuple[Any, ...] | None]:
        """Return traced children and static metadata in declared field order."""

        children = tuple(getattr(self, name) for name in self.pytree_children)
        if not self.pytree_aux_data:
            return children, None

        aux_data = tuple(getattr(self, name) for name in self.pytree_aux_data)
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls, aux_data: tuple[Any, ...] | None, children: tuple[Any, ...]
    ) -> Self:
        """Reconstruct an immutable PyTree object without rerunning ``__init__``."""

        obj = object.__new__(cls)
        for name, value in zip(cls.pytree_children, children, strict=True):
            object.__setattr__(obj, name, value)

        if aux_data is not None:
            for name, value in zip(cls.pytree_aux_data, aux_data, strict=True):
                object.__setattr__(obj, name, value)

        obj._pytree_post_unflatten()
        return obj

    def _pytree_post_unflatten(self) -> None:
        """Restore derived attributes or validate invariants after unflattening."""
