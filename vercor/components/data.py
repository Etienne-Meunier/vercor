from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, NoReturn

from vercor.components.contracts import (
    ComponentSpec,
    ComponentStepReturn,
    FieldImportPolicy,
)
from vercor.components._contracts import (
    merge_component_outputs,
)
from vercor.components.base import Component
from vercor.components.contexts import StepContext
from vercor.dtypes import PrecisionPolicy
from vercor.grids import RectilinearGrid
from vercor.settings import Settings
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
    def from_step(
        cls,
        name: str,
        grid: RectilinearGrid,
        step: Callable[..., ComponentStepReturn],
        *,
        spec: ComponentSpec | None = None,
        payload: Any | None = None,
        settings: Settings | None = None,
    ) -> NoReturn:
        """Reject active stepping for data-only components."""

        _ = cls, name, grid, step, spec, payload, settings
        raise TypeError(
            "DataComponent.from_step is unavailable because data-only "
            "components do not execute steps. Use DataComponent.from_fields "
            "for data, Component.from_step for differentiable models, or "
            "HostComponent.from_step for host-side models."
        )

    @classmethod
    def from_fields(
        cls,
        name: str,
        grid: RectilinearGrid,
        fields: Mapping[str, object] | None = None,
        *,
        settings: Settings | None = None,
        spec: ComponentSpec | None = None,
        import_policy: FieldImportPolicy | None = None,
    ) -> "DataComponent":
        """Create a data-only component from user-provided grid fields.

        Scalar field values expand to grid-shaped constants and seeded field
        names are exposed as declared outputs. Optional lifecycle hooks mirror
        the callable component constructors for setup and runtime customization.
        """

        spec = ComponentSpec() if spec is None else spec
        if settings is None:
            component = cls(name=name, grid=grid, spec=spec)
        else:
            component = cls(
                name=name,
                grid=grid,
                settings=settings,
                spec=spec,
            )
        if fields is not None:
            component.seed_fields(fields)
        component._import_policy = (
            FieldImportPolicy() if import_policy is None else import_policy
        )
        return component

    def seed_fields(
        self,
        fields: Mapping[str, object],
        policy: PrecisionPolicy = None,
    ) -> "DataComponent":
        """Seed data fields and expose their names as declared outputs."""

        super().seed_fields(fields, policy=policy)
        self._spec = merge_component_outputs(self.spec, fields.keys())
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
