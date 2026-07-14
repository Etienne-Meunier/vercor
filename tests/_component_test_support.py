"""Test-only adapters for exercising runtime behavior without public mixins."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Self

import jax.numpy as jnp

from vercor._field_names import unique_field_names
from vercor.components import ComponentSpec, SetupResult
from vercor.components.contexts import SetupContext, StepContext
from vercor.dtypes import DTypePolicy
from vercor.grids import RectilinearGrid
from vercor.settings import Settings


class LegacyTestComponent:
    """Mutable fixture builder normalized to the v4 structural protocol."""

    _author_step: Any

    def __init__(
        self,
        name: str,
        grid: RectilinearGrid,
        *,
        settings: Settings | None = None,
        spec: ComponentSpec | None = None,
    ) -> None:
        self.name = name
        self.grid = grid
        self.settings = Settings() if settings is None else settings
        self._dtype_policy = DTypePolicy.from_jax_config()
        self._data: dict[str, Any] = dict({} if spec is None else spec.initial_fields)
        self._declared_spec = ComponentSpec() if spec is None else spec
        self._cached_spec: ComponentSpec | None = None
        self._legacy_payload: Any | None = None

    @property
    def spec(self) -> ComponentSpec:
        if self._cached_spec is None:
            declaration = self._declared_spec
            outputs = unique_field_names((*declaration.outputs, *tuple(self._data)))
            lifecycle = declaration.lifecycle

            def setup(component: Any, context: SetupContext) -> SetupResult:
                _ = component
                initialize = getattr(self, "initialize", None)
                if callable(initialize):
                    initialize(context)
                hook_result = (
                    None if lifecycle.setup is None else lifecycle.setup(self, context)
                )
                fields = dict(self._data)
                payload = self._legacy_payload
                if hook_result is not None:
                    fields.update(hook_result.fields)
                    payload = hook_result.payload
                return SetupResult(fields=fields, payload=payload)

            from vercor.components import LifecycleHooks

            self._cached_spec = ComponentSpec(
                inputs=declaration.inputs,
                outputs=outputs,
                initial_fields={name: value for name, value in self._data.items()},
                execution=declaration.execution,
                lifecycle=LifecycleHooks(
                    setup=setup,
                    prefill=lifecycle.prefill,
                    validate=lifecycle.validate,
                ),
                transfer=declaration.transfer,
                output=declaration.output,
            )
        return self._cached_spec

    @classmethod
    def from_step(
        cls,
        name: str,
        grid: RectilinearGrid,
        step: Any,
        *,
        spec: ComponentSpec | None = None,
        payload: Any | None = None,
        settings: Settings | None = None,
    ) -> "LegacyTestComponent":
        component = cls(name, grid, spec=spec, settings=settings)
        component._author_step = step
        component._legacy_payload = payload
        component.step = step  # type: ignore[method-assign]
        return component

    def configure(self, spec: ComponentSpec) -> Self:
        self._declared_spec = spec
        self._data = dict(spec.initial_fields)
        self._cached_spec = None
        return self

    def declare_fields(
        self,
        *,
        inputs: Iterable[str] = (),
        outputs: Iterable[str] = (),
        initial_fields: Mapping[str, object] | None = None,
        defaults: Mapping[str, object] | None = None,
    ) -> Self:
        values = initial_fields if initial_fields is not None else defaults
        self._declared_spec = ComponentSpec(
            inputs=inputs,
            outputs=outputs,
            initial_fields=values,
            execution=self._declared_spec.execution,
            lifecycle=self._declared_spec.lifecycle,
            transfer=self._declared_spec.transfer,
            output=self._declared_spec.output,
        )
        self._data.update(values or {})
        self._cached_spec = None
        return self

    def grid_field_defaults(
        self,
        names: Iterable[str],
        value: object = 0.0,
        overrides: Mapping[str, object] | None = None,
        policy: Any = None,
    ) -> dict[str, Any]:
        _ = policy
        values = {name: value for name in names}
        values.update(overrides or {})
        return {
            name: jnp.full(self.grid.shape, jnp.asarray(field_value))
            for name, field_value in values.items()
        }

    def seed_field(self, name: str, value: object, policy: Any = None) -> Self:
        _ = policy
        return self.seed_fields({name: value})

    def seed_fields(self, fields: Mapping[str, object], policy: Any = None) -> Self:
        _ = policy
        for name, value in fields.items():
            array = jnp.asarray(value)
            self._data[name] = (
                jnp.full(self.grid.shape, array) if array.shape == () else array
            )
        self._cached_spec = None
        return self

    def step(
        self,
        fields: Mapping[str, Any],
        context: StepContext,
        payload: Any | None = None,
    ) -> Mapping[str, Any]:
        _ = fields, context, payload
        return {}


class LegacyTestHostComponent(LegacyTestComponent):
    """Test fixture that declares host execution through its component spec."""

    @property
    def spec(self) -> ComponentSpec:
        declaration = super().spec
        if declaration.execution == "host":
            return declaration
        self._cached_spec = ComponentSpec(
            inputs=declaration.inputs,
            outputs=declaration.outputs,
            initial_fields=declaration.initial_fields,
            execution="host",
            lifecycle=declaration.lifecycle,
            transfer=declaration.transfer,
            output=declaration.output,
        )
        return self._cached_spec


__all__ = ["LegacyTestComponent", "LegacyTestHostComponent"]
