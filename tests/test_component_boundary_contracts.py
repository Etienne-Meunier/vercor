from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests._architecture_support import (
    package_import_cycles,
    source_for,
)


def _imported_names_from(path: str, module: str) -> set[str]:
    """Return names imported from one module in a Python source file."""

    tree = ast.parse(source_for(path))
    imported_names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == module:
            imported_names.update(alias.name for alias in node.names)
    return imported_names


def test_component_runtime_helpers_do_not_keep_annotation_only_protocol_layer() -> None:
    helper_paths = (
        "vercor/components/_runtime_fields.py",
        "vercor/components/_runtime_validation.py",
        "vercor/components/_lifecycle_api.py",
        "vercor/components/_callable_wrappers.py",
        "vercor/components/runtime_execution.py",
    )

    for path in helper_paths:
        source = source_for(path)
        assert "vercor.components._protocols" not in source, path
        assert "HostRuntimeExecutionProtocol" not in source, path
        assert "if TYPE_CHECKING:" in source, path

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
    hook_names = {
        "ComponentInitializeHook",
        "ComponentCreatePayloadHook",
        "ComponentPrefillHook",
        "ComponentValidateHook",
    }
    contracts_source = source_for("vercor/components/contracts.py")

    for hook_name in hook_names:
        assert f"{hook_name} =" in contracts_source
    assert "class LifecycleHooks" in contracts_source
    assert "class ComponentLifecycleHooks" not in contracts_source
    assert not Path("vercor/components/_lifecycle.py").exists()

    for path in (
        "vercor/components/base.py",
        "vercor/components/host.py",
    ):
        private_imports = _imported_names_from(path, "vercor.components._lifecycle")
        public_imports = _imported_names_from(path, "vercor.components.contracts")
        assert hook_names.isdisjoint(private_imports), path
        assert "LifecycleHooks" not in public_imports, path
        assert hook_names.isdisjoint(public_imports), path


@pytest.mark.fast_always
def test_lifecycle_storage_uses_component_spec_as_single_owner() -> None:
    contracts_source = source_for("vercor/components/contracts.py")
    base_source = source_for("vercor/components/base.py")
    callable_source = source_for("vercor/components/_callable_wrappers.py")
    data_source = source_for("vercor/components/data.py")

    assert "class LifecycleHooks" in contracts_source
    assert "class ComponentLifecycleHooks" not in contracts_source
    assert "_lifecycle_hooks" not in base_source
    assert "_lifecycle_hooks" not in callable_source
    assert "normalize_lifecycle_hooks" not in data_source
    assert "_lifecycle_hooks" not in data_source
    assert not Path("vercor/components/_protocols.py").exists()


@pytest.mark.fast_always
def test_callable_wrapper_module_does_not_need_request_dataclass() -> None:
    callable_source = source_for("vercor/components/_callable_wrappers.py")
    base_source = source_for("vercor/components/base.py")
    host_source = source_for("vercor/components/host.py")

    assert "class _CallableComponentDefinition" not in callable_source
    assert "def _callable_component_definition(" not in callable_source
    assert "lifecycle_hooks: LifecycleHooks" not in callable_source
    assert "def create_runtime_payload(" not in callable_source
    assert "component._lifecycle_hooks.create_runtime_payload" not in callable_source
    assert "spec=_ComponentComponentSpec(" not in base_source
    assert "spec=ComponentComponentSpec(" not in host_source


@pytest.mark.fast_always
def test_runtime_lifecycle_bridge_is_private_inherited_surface() -> None:
    lifecycle_source = source_for("vercor/components/_lifecycle_api.py")
    component_state_source = source_for("vercor/_runtime/component_state.py")
    state_validation_source = source_for("vercor/_runtime/state_validation.py")

    public_bridge_names = (
        "create_runtime_payload",
        "prefill_runtime_state_fields",
        "validate_runtime_state",
    )
    for method_name in public_bridge_names:
        assert not hasattr(__import__("vercor").Component, method_name)
        assert f"def {method_name}(" not in lifecycle_source

    assert "def _create_runtime_payload(" in lifecycle_source
    assert "def _prefill_runtime_state_fields(" in lifecycle_source
    assert "def _validate_runtime_state(" in lifecycle_source
    assert "component._prefill_runtime_state_fields(" in component_state_source
    assert "component._create_runtime_payload()" in component_state_source
    assert "component._validate_runtime_state(" in state_validation_source


def test_components_package_has_no_top_level_import_cycles() -> None:
    assert package_import_cycles("vercor/components", "vercor.components") == []
