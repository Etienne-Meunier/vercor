from __future__ import annotations

from pathlib import Path

import pytest

from tests._architecture_support import (
    package_import_cycles,
    source_for,
)
from vercor.components import Component


def test_component_runtime_helpers_do_not_keep_annotation_only_protocol_layer() -> None:
    helper_paths = (
        "vercor/components/_runtime_fields.py",
        "vercor/components/runtime_execution.py",
    )

    for path in helper_paths:
        source = source_for(path)
        assert "vercor.components._protocols" not in source, path
        assert "HostRuntimeExecutionProtocol" not in source, path
        assert "if TYPE_CHECKING:" in source, path

    callable_source = source_for("vercor/components/_callable_wrappers.py")
    assert "vercor.components._protocols" not in callable_source
    assert "normalize_component_step_callable" in callable_source
    assert not Path("vercor/components/_lifecycle_api.py").exists()
    assert not Path("vercor/components/_runtime_validation.py").exists()
    assert not Path("vercor/components/_protocols.py").exists()
    components_source = source_for("vercor/components/__init__.py")
    assert "_protocols" not in components_source


@pytest.mark.fast_always
def test_host_runtime_selection_uses_public_component_spec_execution() -> None:
    contracts_source = source_for("vercor/components/contracts.py")
    runtime_execution_source = source_for("vercor/components/runtime_execution.py")

    assert 'execution: Literal["jax", "host"] = "jax"' in contracts_source
    assert 'component.spec.execution == "host"' in runtime_execution_source
    assert "_requires_host_runtime" not in runtime_execution_source
    assert "HostRuntimeExecutionProtocol" not in runtime_execution_source


@pytest.mark.fast_always
def test_public_lifecycle_hook_types_are_owned_by_component_contracts() -> None:
    contracts_source = source_for("vercor/components/contracts.py")

    assert "class LifecycleHooks" in contracts_source
    assert Component.__module__ == "vercor.components.contracts"


@pytest.mark.fast_always
def test_lifecycle_storage_uses_component_spec_as_single_owner() -> None:
    contracts_source = source_for("vercor/components/contracts.py")
    base_source = source_for("vercor/components/base.py")
    data_source = source_for("vercor/components/data.py")

    assert "class LifecycleHooks" in contracts_source
    assert "self.spec = ComponentSpec() if spec is None else spec" in base_source
    assert "lifecycle=declaration.lifecycle" in data_source


@pytest.mark.fast_always
def test_callable_wrapper_module_does_not_need_request_dataclass() -> None:
    callable_source = source_for("vercor/components/_callable_wrappers.py")
    base_source = source_for("vercor/components/base.py")

    assert "class _CallableComponentDefinition" not in callable_source
    assert "def _callable_component_definition(" not in callable_source
    assert "lifecycle_hooks: LifecycleHooks" not in callable_source
    assert "def create_runtime_payload(" not in callable_source
    assert "component._lifecycle_hooks.create_runtime_payload" not in callable_source
    assert "spec=_ComponentComponentSpec(" not in base_source
    assert not Path("vercor/components/host.py").exists()


@pytest.mark.fast_always
def test_runtime_lifecycle_bridge_is_private_binding_surface() -> None:
    adapter_source = source_for("vercor/components/_adapter.py")
    component_state_source = source_for("vercor/_runtime/component_state.py")
    state_validation_source = source_for("vercor/_runtime/state_validation.py")

    assert "def _prefill_runtime_state_fields(" in adapter_source
    assert "def _validate_runtime_state(" in adapter_source
    assert "def prepare_component(" in adapter_source
    assert "component._prefill_runtime_state_fields(" in component_state_source
    assert "component._create_runtime_payload()" in component_state_source
    assert "component._validate_runtime_state(" in state_validation_source


def test_components_package_has_no_top_level_import_cycles() -> None:
    assert package_import_cycles("vercor/components", "vercor.components") == []
