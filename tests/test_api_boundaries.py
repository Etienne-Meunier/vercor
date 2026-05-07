from __future__ import annotations

from pathlib import Path

import pytest

import vercor
import vercor.components as components_module
from tests._coverage_support import make_test_grid
from vercor.components.base import Component, HostRuntimeComponent
from vercor.runtime.contexts import ComponentInitContext, RuntimeStepContext
from vercor.run_sequence import RunSequence
from vercor.runtime import RuntimeComponentState, RuntimeFieldStore


@pytest.mark.fast_always
def test_top_level_exports_public_orchestration_and_component_author_api() -> None:
    expected_public_names = {
        "Clock",
        "Component",
        "ComponentFieldSpec",
        "ComponentSetupContext",
        "ComponentStepContext",
        "ComponentStepResult",
        "Coupler",
        "DataComponent",
        "Exchange",
        "HostRuntimeComponent",
        "RectilinearGrid",
        "RunSequence",
        "data_component",
        "differentiable_component",
        "host_component",
        "make_data_component",
        "make_differentiable_component",
        "make_host_component",
    }
    runtime_internal_names = {
        "ComponentInitContext",
        "RuntimeComponentContract",
        "RuntimeComponentState",
        "RuntimeComponentView",
        "RuntimeCouplerState",
        "RuntimeDispatchContext",
        "RuntimeFieldStore",
        "RuntimeStepContext",
        "RuntimeStepInfo",
    }

    assert expected_public_names.issubset(set(vercor.__all__))
    assert runtime_internal_names.isdisjoint(set(vercor.__all__))

    assert vercor.Component is Component
    assert vercor.ComponentFieldSpec is components_module.ComponentFieldSpec
    assert vercor.ComponentSetupContext is ComponentInitContext
    assert vercor.ComponentStepContext is RuntimeStepContext
    assert vercor.ComponentStepResult is components_module.ComponentStepResult
    data_component_type = getattr(components_module, "DataComponent", None)
    assert data_component_type is not None
    assert getattr(vercor, "DataComponent", None) is data_component_type
    assert vercor.HostRuntimeComponent is HostRuntimeComponent
    assert vercor.data_component is components_module.data_component
    assert vercor.differentiable_component is components_module.differentiable_component
    assert vercor.host_component is components_module.host_component
    assert vercor.make_data_component is components_module.make_data_component
    assert (
        vercor.make_differentiable_component
        is components_module.make_differentiable_component
    )
    assert vercor.make_host_component is components_module.make_host_component
    assert vercor.RunSequence is RunSequence
    for name in runtime_internal_names:
        assert not hasattr(vercor, name)


@pytest.mark.fast_always
def test_components_package_exports_only_component_author_contracts() -> None:
    assert components_module.__all__ == [
        "Component",
        "ComponentFieldSpec",
        "ComponentSetupContext",
        "ComponentStepContext",
        "ComponentStepResult",
        "DataComponent",
        "HostRuntimeComponent",
        "data_component",
        "differentiable_component",
        "host_component",
        "make_data_component",
        "make_differentiable_component",
        "make_host_component",
    ]
    assert components_module.Component is Component
    assert hasattr(components_module, "ComponentFieldSpec")
    assert components_module.ComponentSetupContext is ComponentInitContext
    assert components_module.ComponentStepContext is RuntimeStepContext
    assert hasattr(components_module, "ComponentStepResult")
    assert hasattr(components_module, "DataComponent")
    assert components_module.HostRuntimeComponent is HostRuntimeComponent
    assert hasattr(components_module, "data_component")
    assert hasattr(components_module, "differentiable_component")
    assert hasattr(components_module, "host_component")
    assert hasattr(components_module, "make_data_component")
    assert hasattr(components_module, "make_differentiable_component")
    assert hasattr(components_module, "make_host_component")
    assert not hasattr(components_module, "RuntimeComponentState")
    assert not hasattr(components_module, "ComponentInitContext")
    assert not hasattr(components_module, "RuntimeStepContext")


@pytest.mark.fast_always
def test_component_base_internals_are_private_modules() -> None:
    base_source = Path("vercor/components/base.py").read_text(encoding="utf-8")
    contracts_source = Path("vercor/components/_contracts.py").read_text(
        encoding="utf-8"
    )
    callable_source = Path("vercor/components/_callable_wrappers.py").read_text(
        encoding="utf-8"
    )
    validation_source = Path("vercor/components/_validation.py").read_text(
        encoding="utf-8"
    )

    assert "class ComponentFieldSpec" in contracts_source
    assert "class ComponentStepResult" in contracts_source
    assert "def normalize_author_field_values" in contracts_source
    assert "class _CallableRuntimeMixin" in callable_source
    assert "def normalize_component_step_callable" in callable_source
    assert "def validate_component_setup" in validation_source

    private_markers = (
        "class _CallableRuntimeMixin",
        "class _CallableComponent",
        "class _CallableHostRuntimeComponent",
        "def _normalize_component_step_callable",
        "def _component_step_signature_error",
    )
    for marker in private_markers:
        assert marker not in base_source

    assert "_contracts" not in components_module.__all__
    assert "_callable_wrappers" not in components_module.__all__
    assert "_validation" not in components_module.__all__


@pytest.mark.fast_always
def test_runtime_state_is_separate_from_public_component_objects() -> None:
    assert hasattr(components_module, "DataComponent")
    component = components_module.DataComponent(
        name="ATM",
        grid=make_test_grid(name="api-boundary"),
    )
    runtime_state = RuntimeComponentState(
        data=RuntimeFieldStore.empty(),
        incoming=RuntimeFieldStore.empty(),
        outgoing=RuntimeFieldStore.empty(),
    )

    assert not isinstance(runtime_state, Component)
    assert not hasattr(runtime_state, "name")
    assert not hasattr(runtime_state, "grid")
    assert not hasattr(runtime_state, "settings")
    assert not hasattr(component, "incoming")
    assert not hasattr(component, "outgoing")
    assert not hasattr(component, "with_data")


@pytest.mark.fast_always
def test_examples_import_run_sequence_from_top_level_public_api() -> None:
    for path in Path("examples").glob("run_*.py"):
        source = path.read_text(encoding="utf-8")
        assert "from vercor.coupler import RunSequence" not in source
        if "RunSequence" in source:
            public_import_lines = [
                line
                for line in source.splitlines()
                if line.startswith("from vercor import ")
            ]
            assert any("RunSequence" in line for line in public_import_lines), path
