from __future__ import annotations

from pathlib import Path

import pytest

import vercor
import vercor.components as components_module
from tests._coverage_support import make_test_grid
from vercor.components.base import Component, HostRuntimeComponent
from vercor.run_sequence import RunSequence
from vercor.runtime import RuntimeComponentState, RuntimeFieldStore


@pytest.mark.fast_always
def test_top_level_exports_public_orchestration_and_component_author_api() -> None:
    expected_public_names = {
        "Clock",
        "Component",
        "ComponentFieldSpec",
        "ComponentStepResult",
        "Coupler",
        "DataComponent",
        "Exchange",
        "HostRuntimeComponent",
        "RectilinearGrid",
        "RunSequence",
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
    assert vercor.ComponentStepResult is components_module.ComponentStepResult
    data_component_type = getattr(components_module, "DataComponent", None)
    assert data_component_type is not None
    assert getattr(vercor, "DataComponent", None) is data_component_type
    assert vercor.HostRuntimeComponent is HostRuntimeComponent
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
        "ComponentStepResult",
        "DataComponent",
        "HostRuntimeComponent",
        "make_data_component",
        "make_differentiable_component",
        "make_host_component",
    ]
    assert components_module.Component is Component
    assert hasattr(components_module, "ComponentFieldSpec")
    assert hasattr(components_module, "ComponentStepResult")
    assert hasattr(components_module, "DataComponent")
    assert components_module.HostRuntimeComponent is HostRuntimeComponent
    assert hasattr(components_module, "make_data_component")
    assert hasattr(components_module, "make_differentiable_component")
    assert hasattr(components_module, "make_host_component")
    assert not hasattr(components_module, "RuntimeComponentState")
    assert not hasattr(components_module, "ComponentInitContext")
    assert not hasattr(components_module, "RuntimeStepContext")


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
