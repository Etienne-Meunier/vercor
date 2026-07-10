"""Boundary tests for lazy bundled setup imports and configuration."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import cast

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPTIONAL_MODULE_ROOTS = {"credit", "dinosaur", "jcm", "tensorflow", "torch", "veros"}
RUNTIME_ENVIRONMENT_KEYS = (
    "TF_CPP_MIN_LOG_LEVEL",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
)
ENVIRONMENT_SENTINEL = "vercor-import-boundary"


def _run_setup_probe(statement: str) -> dict[str, object]:
    script = f"""
import json
import os
import sys

for key in {RUNTIME_ENVIRONMENT_KEYS!r}:
    os.environ[key] = {ENVIRONMENT_SENTINEL!r}

{statement}

print(json.dumps({{
    "optional_modules": sorted(
        name for name in sys.modules
        if name.partition(".")[0] in {OPTIONAL_MODULE_ROOTS!r}
    ),
    "runtime_environment": {{
        key: os.environ.get(key) for key in {RUNTIME_ENVIRONMENT_KEYS!r}
    }},
    "setup_modules": sorted(
        name for name in sys.modules if name.startswith("vercor.setups")
    ),
}}))
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return cast(dict[str, object], json.loads(completed.stdout.splitlines()[-1]))


def _run_missing_dependency_probe(
    *,
    factory_name: str,
    dependency_root: str,
    invocation: str,
) -> dict[str, str]:
    script = f"""
import builtins
import json
import vercor.setups as setups

factory = setups.{factory_name}
real_import = builtins.__import__

def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == {dependency_root!r} or name.startswith({dependency_root!r} + "."):
        raise ModuleNotFoundError("blocked optional dependency: " + name)
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = blocked_import
try:
    {invocation}
except Exception as error:
    print(json.dumps({{"type": type(error).__name__, "message": str(error)}}))
else:
    print(json.dumps({{"type": "", "message": ""}}))
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return cast(dict[str, str], json.loads(completed.stdout.splitlines()[-1]))


@pytest.mark.fast_always
@pytest.mark.parametrize(
    "statement",
    (
        "import vercor",
        "import vercor.setups",
        "import vercor.setups as setups; dir(setups)",
        (
            "from vercor.setups import (CAMulatorConfig, JAXGCMConfig, "
            "JCMLandAtmosphereConfig, Spinup, VerosConfig); "
            "Spinup(); JAXGCMConfig(); VerosConfig(); "
            "CAMulatorConfig(config_path='config.yml'); JCMLandAtmosphereConfig()"
        ),
    ),
)
def test_public_import_and_config_access_are_optional_dependency_free(
    statement: str,
) -> None:
    result = _run_setup_probe(statement)

    assert result["optional_modules"] == []
    assert result["runtime_environment"] == {
        key: ENVIRONMENT_SENTINEL for key in RUNTIME_ENVIRONMENT_KEYS
    }


@pytest.mark.fast_always
@pytest.mark.parametrize(
    ("factory_name", "factory_module"),
    (
        ("make_jax_gcm", "vercor.setups._external.jax_gcm"),
        ("make_veros_gcm", "vercor.setups._external.veros_gcm"),
        ("make_camulator_gcm", "vercor.setups._external.camulator"),
    ),
)
def test_lazy_factory_attribute_access_loads_only_lightweight_factory_module(
    factory_name: str,
    factory_module: str,
) -> None:
    result = _run_setup_probe(
        f"import vercor.setups as setups; factory = setups.{factory_name}; "
        "assert callable(factory)"
    )

    setup_modules = result["setup_modules"]
    assert isinstance(setup_modules, list)
    assert factory_module in setup_modules
    assert result["optional_modules"] == []
    assert result["runtime_environment"] == {
        key: ENVIRONMENT_SENTINEL for key in RUNTIME_ENVIRONMENT_KEYS
    }


@pytest.mark.fast_always
def test_vercor_setups_is_the_only_setup_lazy_export_registry() -> None:
    import vercor.setups as setups
    import vercor.setups._data as data_setups
    import vercor.setups._external as external_setups

    assert "_LAZY_EXPORTS" in vars(setups)
    for private_package in (data_setups, external_setups):
        assert "_LAZY_EXPORTS" not in vars(private_package)
        assert "__getattr__" not in vars(private_package)
        assert not any(name.startswith("make_") for name in dir(private_package))


@pytest.mark.fast_always
@pytest.mark.parametrize(
    ("factory_name", "dependency_root", "invocation", "message"),
    (
        (
            "make_jax_gcm",
            "jcm",
            "factory(None, None)",
            "requires the jcm package.*pip install jcm",
        ),
        (
            "make_veros_gcm",
            "veros",
            "factory()",
            "requires the Veros package.*pip install veros",
        ),
        (
            "make_camulator_gcm",
            "credit",
            (
                "factory(config=setups.CAMulatorConfig("
                "config_path='missing.yml', device='cpu'))"
            ),
            "CREDIT modules are required.*credit is installed",
        ),
    ),
)
def test_missing_optional_dependencies_fail_at_factory_invocation(
    factory_name: str,
    dependency_root: str,
    invocation: str,
    message: str,
) -> None:
    result = _run_missing_dependency_probe(
        factory_name=factory_name,
        dependency_root=dependency_root,
        invocation=invocation,
    )

    assert result["type"] == "ImportError"
    assert re.search(message, result["message"]) is not None


@pytest.mark.fast_always
def test_camulator_enabled_spinup_fails_before_runtime_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vercor.setups as setups
    import vercor.setups._external.camulator_runtime_settings as runtime_settings

    configuration_calls: list[None] = []

    def unexpected_runtime_configuration() -> None:
        configuration_calls.append(None)
        pytest.fail("CAMulator runtime must not be configured for unsupported spinup")

    monkeypatch.setattr(
        runtime_settings,
        "configure_camulator_runtime",
        unexpected_runtime_configuration,
    )

    with pytest.raises(
        ValueError,
        match="CAMulator spinup is not implemented.*Spinup\\(enabled=False\\)",
    ):
        setups.make_camulator_gcm(
            config=setups.CAMulatorConfig(
                config_path="unused.yml",
                spinup=setups.Spinup(enabled=True),
            )
        )

    assert configuration_calls == []
