from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Self

from vercor.components.contracts import (
    ComponentSpec,
)
from vercor.components._constructor_options import normalize_component_spec
from vercor.components._contracts import (
    normalize_author_field_values as _normalize_author_field_values,
)
from vercor._field_names import unique_field_names as _unique_field_names
from vercor.dtypes import PrecisionPolicy
from vercor.exceptions import ComponentError
from vercor.grids import RectilinearGrid
from vercor.settings import Settings
from vercor.types import RuntimeArray


class ComponentFieldAuthoringMixin:
    """Author-facing field declaration, setup seeding, and settings helpers."""

    name: str
    grid: RectilinearGrid
    _data: dict[str, RuntimeArray]
    settings: Settings
    _spec: ComponentSpec

    def configure(self: Self, spec: ComponentSpec) -> Self:
        """Replace this component's public runtime field contract."""

        if not isinstance(spec, ComponentSpec):
            raise ComponentError("Component spec must be a ComponentSpec instance.")
        self._spec = spec
        return self

    def declare_fields(
        self: Self,
        *,
        inputs: Iterable[str] = (),
        outputs: Iterable[str] = (),
        defaults: Mapping[str, object] | None = None,
    ) -> Self:
        """Declare runtime data fields for subclasses using author-facing names."""

        declared = normalize_component_spec(
            inputs=inputs,
            outputs=outputs,
            defaults=defaults,
        )
        self._spec = ComponentSpec(
            inputs=declared.inputs,
            outputs=declared.outputs,
            defaults=_normalize_author_field_values(
                component_name=self.name,
                grid=self.grid,
                fields=declared.defaults,
                policy=self.settings,
            )
            or {},
            execution=self._spec.execution,
            lifecycle=self._spec.lifecycle,
            output=self._spec.output,
        )
        return self

    @property
    def spec(self) -> ComponentSpec:
        """Return this component's declared author-facing runtime field contract."""

        return self._spec

    @property
    def field_names(self) -> tuple[str, ...]:
        """Return setup-time field names in insertion order."""

        return tuple(self._data)

    def initial_fields(self) -> Mapping[str, RuntimeArray]:
        """Return setup-time grid fields as a plain field mapping."""

        return dict(self._data)

    def update_settings(self, **values: object) -> Self:
        """Update component settings by name and return this component."""

        for setting_name, setting_value in values.items():
            self.settings.set(setting_name, setting_value)
        return self

    def grid_field_defaults(
        self,
        names: Iterable[str],
        value: object = 0.0,
        overrides: Mapping[str, object] | None = None,
        policy: PrecisionPolicy = None,
    ) -> dict[str, RuntimeArray]:
        """Return grid-shaped default fields for named runtime data fields."""

        field_names = _unique_field_names(names)
        defaults: dict[str, object] = {field_name: value for field_name in field_names}
        for field_name, field_value in (overrides or {}).items():
            if field_name not in defaults:
                raise ComponentError(
                    f"Default override field '{field_name}' is not declared for "
                    f"component '{self.name}'."
                )
            defaults[field_name] = field_value

        return (
            _normalize_author_field_values(
                component_name=self.name,
                grid=self.grid,
                fields=defaults,
                policy=self.settings if policy is None else policy,
            )
            or {}
        )

    def seed_field(
        self: Self,
        name: str,
        value: object,
        policy: PrecisionPolicy = None,
    ) -> Self:
        """Seed one setup-time grid field and return this component."""

        return self.seed_fields({name: value}, policy=policy)

    def seed_fields(
        self: Self,
        fields: Mapping[str, object],
        policy: PrecisionPolicy = None,
    ) -> Self:
        """Seed setup-time grid fields and return this component."""

        field_updates = _normalize_author_field_values(
            component_name=self.name,
            grid=self.grid,
            fields=fields,
            policy=self.settings if policy is None else policy,
        )
        self._data.update(field_updates or {})
        return self

    def seed_declared_defaults(
        self: Self,
        policy: PrecisionPolicy = None,
    ) -> Self:
        """Seed this component's declared default fields and return itself."""

        defaults = self._spec.defaults
        if defaults:
            self.seed_fields(defaults, policy=policy)
        return self


__all__ = ["ComponentFieldAuthoringMixin"]
