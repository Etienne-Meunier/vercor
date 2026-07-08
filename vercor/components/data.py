from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from vercor.components.contracts import (
    AuthorFieldValues,
    ComponentHooks,
    ComponentStepReturn,
    FieldSpec,
)
from vercor.components._contracts import (
    merge_component_outputs,
)
from vercor.components._constructor_options import normalize_lifecycle_hooks
from vercor.components.base import Component
from vercor.dtypes import PrecisionPolicy
from vercor.grids import RectilinearGrid
from vercor.output.adapters import OutputSpec
from vercor.settings import Settings

if TYPE_CHECKING:
    from vercor.components.contexts import StepContext
    from vercor.types import RuntimeArray


class DataComponent(Component):
    """Base class for data-only components that intentionally do not step.

    Use this for forcing and boundary-condition adapters whose runtime behavior is
    limited to importing/exporting seeded fields through the coupler contract.
    Data components must not own active runtime stepping behavior; compute
    plotting-only diagnostics outside runtime state. Active differentiable models
    should inherit :class:`Component` and implement :meth:`Component.step`.
    """

    @classmethod
    def from_fields(
        cls,
        name: str,
        grid: RectilinearGrid,
        fields: AuthorFieldValues = None,
        *,
        settings: Settings | None = None,
        spec: FieldSpec | None = None,
        hooks: ComponentHooks | None = None,
        output: OutputSpec | None = None,
    ) -> "DataComponent":
        """Create a data-only component from user-provided grid fields.

        Scalar field values expand to grid-shaped constants and seeded field
        names are exposed as declared outputs. Optional lifecycle hooks mirror
        the callable component constructors for setup and runtime customization.
        """

        if settings is None:
            component = cls(name=name, grid=grid, output=output)
        else:
            component = cls(name=name, grid=grid, settings=settings, output=output)
        if spec is not None:
            component._field_spec = spec
        if fields is not None:
            component.seed_fields(fields)
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

    def step(
        self,
        fields: Mapping[str, "RuntimeArray"],
        context: "StepContext",
        payload: Any | None = None,
    ) -> ComponentStepReturn:
        """Return no updates for data-only components."""

        _ = fields, context, payload
        return {}


__all__ = ["DataComponent"]
