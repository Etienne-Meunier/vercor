from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, final

from vercor.components.contracts import (
    AuthorFieldValues,
    ComponentHooks,
    FieldNames,
)
from vercor.components._contracts import (
    merge_component_outputs,
)
from vercor.components._constructor_options import normalize_lifecycle_hooks
from vercor.components.base import Component
from vercor.dtypes import PrecisionPolicy
from vercor._grid import RectilinearGrid
from vercor.settings import Settings

if TYPE_CHECKING:
    from vercor.components.contexts import StepContext
    from vercor.runtime.state import RuntimeComponentState


class DataComponent(Component):
    """Base class for data-only components that intentionally do not step.

    Use this for forcing and boundary-condition adapters whose runtime behavior is
    limited to importing/exporting seeded fields through the coupler contract.
    Data components must not own active runtime stepping behavior; compute
    plotting-only diagnostics outside runtime state. Active differentiable models
    should inherit :class:`Component` and implement
    :meth:`Component.step_runtime_state` instead.
    """

    @classmethod
    def from_fields(
        cls,
        name: str,
        grid: RectilinearGrid,
        *,
        fields: AuthorFieldValues = None,
        settings: Settings | None = None,
        outputs: FieldNames = (),
        hooks: ComponentHooks | None = None,
    ) -> "DataComponent":
        """Create a data-only component from user-provided grid fields.

        Scalar field values expand to grid-shaped constants and seeded field
        names are exposed as declared outputs. Optional lifecycle hooks mirror
        the callable component constructors for setup and runtime customization.
        """

        if settings is None:
            component = cls(name=name, grid=grid)
        else:
            component = cls(name=name, grid=grid, settings=settings)
        if fields is not None:
            component.seed_fields(fields)
        if tuple(outputs):
            component._field_spec = merge_component_outputs(
                component.field_spec,
                outputs,
            )
        component._lifecycle_hooks = normalize_lifecycle_hooks(hooks=hooks)
        return component

    def seed_fields(
        self,
        fields: Mapping[str, object],
        policy: PrecisionPolicy = None,
    ) -> "DataComponent":
        """Seed data fields and expose their names as declared outputs."""

        super().seed_fields(fields, policy=policy)
        self._field_spec = merge_component_outputs(self.field_spec, fields.keys())
        return self

    @final
    def step_runtime_state(
        self,
        component_state: "RuntimeComponentState",
        context: StepContext,
    ) -> "RuntimeComponentState":
        """Return the runtime state unchanged for data-only components."""

        _ = context
        return component_state


__all__ = ["DataComponent"]
