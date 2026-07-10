"""Static and artifact-level tests for VerCOR distribution boundaries."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tomllib
import zipfile

import pytest

from tests._distribution_support import build_distributions, install_local_target

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_ROOT = PROJECT_ROOT / "tests" / "fixtures" / "public_plugin"


@pytest.mark.fast_always
def test_runtime_metadata_separates_test_and_development_dependencies() -> None:
    metadata = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    project = metadata["project"]
    runtime_dependencies = tuple(project["dependencies"])
    extras = project["optional-dependencies"]

    assert project["version"] == "3.0.0"
    assert not any(
        dependency.lower().startswith("pytest") for dependency in runtime_dependencies
    )
    assert {"jcm", "veros", "test", "dev"}.issubset(extras)
    assert any(dependency.lower().startswith("pytest") for dependency in extras["test"])
    for tool in ("black", "build", "flake8", "mypy"):
        assert any(dependency.lower().startswith(tool) for dependency in extras["dev"])


@pytest.mark.fast_always
def test_pep561_marker_and_public_plugin_fixture_are_present() -> None:
    assert (PROJECT_ROOT / "vercor" / "py.typed").is_file()
    required_plugin_files = (
        "pyproject.toml",
        "src/vercor_public_plugin/__init__.py",
        "src/vercor_public_plugin/plugin.py",
        "src/vercor_public_plugin/smoke.py",
        "src/vercor_public_plugin/py.typed",
        "use_site.py",
    )
    for relative_path in required_plugin_files:
        assert (PLUGIN_ROOT / relative_path).is_file(), relative_path


@pytest.mark.fast_always
def test_public_plugin_fixture_never_imports_private_vercor_modules() -> None:
    python_paths = sorted(PLUGIN_ROOT.rglob("*.py"))
    assert python_paths

    for path in python_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                modules.append(node.module)
            for module in modules:
                if module == "vercor" or module.startswith("vercor."):
                    assert not any(
                        part.startswith("_") for part in module.split(".")[1:]
                    ), f"{path} imports private VerCOR module {module}"


@pytest.mark.fast_always
def test_ci_validates_installed_artifacts_across_supported_environments() -> None:
    workflow = (PROJECT_ROOT / ".github/workflows/python-package.yml").read_text(
        encoding="utf-8"
    )

    for python_version in ("3.12", "3.13"):
        assert python_version in workflow
    for environment in ("base", "jcm", "veros"):
        assert environment in workflow
    assert "python -m build" in workflow
    assert "actions/upload-artifact@v4" in workflow
    assert "actions/download-artifact@v4" in workflow
    assert "tests/fixtures/public_plugin" in workflow
    assert "vercor_public_plugin.smoke" in workflow
    assert "pip install ." not in workflow


def test_built_distributions_run_public_plugin_outside_checkout(
    tmp_path: Path,
) -> None:
    distributions = build_distributions(PROJECT_ROOT, tmp_path / "dist")

    assert distributions.wheel.name == "vercor-3.0.0-py3-none-any.whl"
    assert distributions.sdist.name == "vercor-3.0.0.tar.gz"
    with zipfile.ZipFile(distributions.wheel) as wheel:
        wheel_names = set(wheel.namelist())
        metadata_name = next(
            name for name in wheel_names if name.endswith(".dist-info/METADATA")
        )
        metadata = wheel.read(metadata_name).decode("utf-8")
    assert "vercor/py.typed" in wheel_names
    assert "Version: 3.0.0" in metadata
    pytest_requirements = [
        line
        for line in metadata.splitlines()
        if line.lower().startswith("requires-dist: pytest")
    ]
    assert pytest_requirements
    assert all("extra ==" in line for line in pytest_requirements)
    assert "Provides-Extra: test" in metadata
    assert "Provides-Extra: dev" in metadata

    with tarfile.open(distributions.sdist, "r:gz") as sdist:
        sdist_names = set(sdist.getnames())
    assert "vercor-3.0.0/vercor/py.typed" in sdist_names

    target = tmp_path / "installed-target"
    install_local_target(
        wheel=distributions.wheel,
        plugin_root=PLUGIN_ROOT,
        target=target,
        build_pythonpath=distributions.build_pythonpath,
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(target)
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib.metadata, json, pathlib, vercor; "
                "print(json.dumps({'file': vercor.__file__, "
                "'version': importlib.metadata.version('vercor'), "
                "'typed': str(pathlib.Path(vercor.__file__).with_name('py.typed'))}))"
            ),
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    installed = json.loads(probe.stdout)
    assert Path(installed["file"]).is_relative_to(target)
    assert installed["version"] == "3.0.0"
    assert Path(installed["typed"]).is_file()

    smoke_output = tmp_path / "plugin-output"
    smoke = subprocess.run(
        [
            sys.executable,
            "-m",
            "vercor_public_plugin.smoke",
            "--output-dir",
            str(smoke_output),
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    evidence = json.loads(smoke.stdout.splitlines()[-1])
    assert evidence["temperature"] == 2.0
    assert evidence["host_value"] == 14.0
    assert evidence["lifecycle"] == ["user-initialize", "hook-initialize"]
    assert evidence["topology"] == ["applies", "build"]
    assert evidence["snapshot"] == {"component": "JAX", "temperature": 2.0}

    mypy_environment = environment.copy()
    mypy_environment["MYPYPATH"] = str(target)
    subprocess.run(
        [
            str(Path(sys.executable).with_name("mypy")),
            "--config-file",
            str(PLUGIN_ROOT / "pyproject.toml"),
            str(PLUGIN_ROOT / "src"),
            str(PLUGIN_ROOT / "use_site.py"),
        ],
        cwd=tmp_path,
        env=mypy_environment,
        check=True,
        capture_output=True,
        text=True,
    )
