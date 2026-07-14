"""Static and artifact-level tests for VerCOR distribution boundaries."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tomllib
import zipfile

import pytest
import yaml

import tests._distribution_support as distribution_support
from tests._distribution_support import (
    BuiltDistributions,
    EXPECTED_SDIST_NAME,
    EXPECTED_VERSION,
    EXPECTED_WHEEL_NAME,
    build_distributions,
    install_local_target,
)
from tests.test_setup_boundaries import _run_setup_probe

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_ROOT = PROJECT_ROOT / "tests" / "fixtures" / "public_plugin"
FROZEN_PLUGIN_ROOT = PROJECT_ROOT / "tests" / "fixtures" / "public_plugin_3_0"
EXPECTED_PLUGIN_WHEEL_NAME = "vercor_public_plugin-0.1.0-py3-none-any.whl"
EXPECTED_FROZEN_PLUGIN_WHEEL_NAME = "vercor_compat_plugin_3_0-0.1.0-py3-none-any.whl"


@pytest.fixture(scope="module")
def built_distributions(tmp_path_factory: pytest.TempPathFactory) -> BuiltDistributions:
    """Build once locally or reuse the explicitly supplied CI artifact bundle."""

    return build_distributions(
        PROJECT_ROOT,
        tmp_path_factory.mktemp("distribution-build") / "dist",
    )


@pytest.mark.fast_always
def test_runtime_metadata_separates_test_and_development_dependencies() -> None:
    metadata = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    project = metadata["project"]
    runtime_dependencies = tuple(project["dependencies"])
    extras = project["optional-dependencies"]

    assert project["version"] == "3.1.1"
    assert not any(
        dependency.lower().startswith("pytest") for dependency in runtime_dependencies
    )
    assert {"jcm", "veros", "test", "dev"}.issubset(extras)
    assert any(dependency.lower().startswith("pytest") for dependency in extras["test"])
    assert any(
        dependency.lower().startswith("pytest-cov") for dependency in extras["test"]
    )
    assert any(
        dependency.lower().startswith("pytest-cov") for dependency in extras["dev"]
    )
    for tool in ("black", "build", "flake8", "mypy"):
        assert any(dependency.lower().startswith(tool) for dependency in extras["dev"])

    coverage = metadata["tool"]["coverage"]
    assert coverage["run"]["branch"] is True
    assert coverage["report"]["fail_under"] == 90


@pytest.mark.fast_always
def test_pep561_markers_and_both_public_plugin_fixtures_are_present() -> None:
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

    required_frozen_plugin_files = (
        "pyproject.toml",
        "src/vercor_compat_plugin_3_0/__init__.py",
        "src/vercor_compat_plugin_3_0/plugin.py",
        "src/vercor_compat_plugin_3_0/smoke.py",
        "src/vercor_compat_plugin_3_0/py.typed",
        "use_site.py",
    )
    for relative_path in required_frozen_plugin_files:
        assert (FROZEN_PLUGIN_ROOT / relative_path).is_file(), relative_path


@pytest.mark.fast_always
def test_public_plugin_fixtures_are_isolated_and_never_import_private_modules() -> None:
    for fixture_root in (PLUGIN_ROOT, FROZEN_PLUGIN_ROOT):
        python_paths = sorted(fixture_root.rglob("*.py"))
        assert python_paths

        for path in python_paths:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
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
            if fixture_root == PLUGIN_ROOT:
                assert "vercor_compat_plugin_3_0" not in source
            else:
                assert "vercor_public_plugin" not in source


@pytest.mark.fast_always
def test_current_public_plugin_uses_canonical_owners_and_v4_workflows() -> None:
    project = tomllib.loads(
        (PLUGIN_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    source = (PLUGIN_ROOT / "src/vercor_public_plugin/plugin.py").read_text(
        encoding="utf-8"
    )

    assert project["version"] == "0.1.0"
    assert project["dependencies"] == ["vercor>=4,<5"]
    for owner in (
        "vercor.clock",
        "vercor.components",
        "vercor.coupling",
        "vercor.exchanges",
        "vercor.grids",
        "vercor.output",
        "vercor.regridding",
        "vercor.runtime",
        "vercor.state",
        "vercor.topology",
        "vercor.types",
    ):
        assert f"from {owner} import" in source, owner
    for contract in (
        "DataComponent",
        "SetupResult(",
        "Exchange(",
        "bilinear",
        "StepResult(",
        ".replace_fields(",
    ):
        assert contract in source, contract
    for removed_contract in (
        "defaults=",
        "DataComponent.from_fields",
        "create_payload=",
        "initialize=",
        "def initial_fields(",
    ):
        assert removed_contract not in source, removed_contract


@pytest.mark.fast_always
def test_frozen_plugin_uses_only_3_0_contracts_and_its_own_distribution() -> None:
    project = tomllib.loads(
        (FROZEN_PLUGIN_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    source = (FROZEN_PLUGIN_ROOT / "src/vercor_compat_plugin_3_0/plugin.py").read_text(
        encoding="utf-8"
    )

    assert project["name"] == "vercor-compat-plugin-3-0"
    assert project["version"] == "0.1.0"
    assert project["dependencies"] == ["vercor>=3.0,<4"]
    assert "vercor_public_plugin" not in source
    for newer_contract in (
        "DataComponent",
        "StepResult",
        "replace_fields",
        "TopologyPolicy",
    ):
        assert newer_contract not in source


@pytest.mark.fast_always
def test_ci_validates_installed_artifacts_across_supported_environments() -> None:
    workflow_path = PROJECT_ROOT / ".github/workflows/python-package.yml"
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    jobs = workflow["jobs"]
    build_job = jobs["build-artifacts"]
    installed_job = jobs["installed-artifact-tests"]

    build_steps = build_job["steps"]
    build_commands = "\n".join(
        step.get("run", "") for step in build_steps if isinstance(step, dict)
    )
    assert "python -m build --outdir dist" in build_commands
    assert (
        "python -m build --wheel --outdir dist tests/fixtures/public_plugin"
        in build_commands
    )
    assert (
        "python -m build --wheel --outdir dist tests/fixtures/public_plugin_3_0"
        in build_commands
    )
    upload_step = next(
        step for step in build_steps if step.get("uses") == "actions/upload-artifact@v4"
    )
    assert upload_step["with"]["path"] == "dist/"

    matrix = installed_job["strategy"]["matrix"]
    assert matrix["python-version"] == ["3.12", "3.13"]
    assert matrix["environment"] == ["base", "jcm", "veros"]
    assert len(matrix["python-version"]) * len(matrix["environment"]) == 6
    included = {item["environment"]: item for item in matrix["include"]}
    assert set(included) == {"base", "jcm", "veros"}
    assert (
        "test_make_jcm_land_atmosphere_replaces_only_missing_forcing"
        in included["jcm"]["pytest-target"]
    )
    assert (
        "test_jax_gcm_initialize_uses_provided_forcing_and_can_spin_up"
        in included["jcm"]["pytest-target"]
    )
    assert (
        "test_veros_initialize_spinup_follows_enabled_only"
        in included["veros"]["pytest-target"]
    )

    installed_steps = installed_job["steps"]
    installed_commands = "\n".join(
        step.get("run", "") for step in installed_steps if isinstance(step, dict)
    )
    download_step = next(
        step
        for step in installed_steps
        if step.get("uses") == "actions/download-artifact@v4"
    )
    assert download_step["with"]["path"] == "dist/"
    assert "python -m build" not in installed_commands
    assert "VERCOR_ARTIFACT_DIR" in installed_commands
    assert "VERCOR_PLUGIN_WHEEL_PATH" in installed_commands
    assert "VERCOR_COMPAT_PLUGIN_WHEEL_PATH" in installed_commands
    assert "VERCOR_TEST_PACKAGE_ROOT" in installed_commands
    assert EXPECTED_WHEEL_NAME in installed_commands
    assert "vercor-3.0.0-py3-none-any.whl" not in installed_commands
    assert "vercor_public_plugin.smoke" in installed_commands
    assert "vercor_compat_plugin_3_0.smoke" in installed_commands
    assert "MYPYPATH" in installed_commands
    assert "tests/fixtures/public_plugin/src" not in installed_commands
    assert (
        "pip install --no-deps tests/fixtures/public_plugin" not in installed_commands
    )
    assert 'pip install --no-deps "${PLUGIN_WHEEL_PATH}"' in installed_commands
    assert 'pip install --no-deps "${COMPAT_PLUGIN_WHEEL_PATH}"' in installed_commands
    assert "pip install ." not in installed_commands


@pytest.mark.fast_always
def test_ci_quality_job_enforces_static_full_and_coverage_gates() -> None:
    workflow = yaml.safe_load(
        (PROJECT_ROOT / ".github/workflows/python-package.yml").read_text(
            encoding="utf-8"
        )
    )
    quality = workflow["jobs"]["quality"]
    checkout = next(
        step for step in quality["steps"] if step.get("uses") == "actions/checkout@v4"
    )
    setup = next(
        step
        for step in quality["steps"]
        if step.get("uses") == "actions/setup-python@v5"
    )
    commands = "\n".join(
        step.get("run", "") for step in quality["steps"] if isinstance(step, dict)
    )

    assert checkout.get("with", {}).get("fetch-depth") == 0
    assert setup["with"]["python-version"] == "3.12"
    assert 'pip install ".[dev,jcm,veros]"' in commands
    assert "black --check vercor examples tests" in commands
    assert "flake8 ." in commands
    assert "--exit-zero" not in commands
    assert "mypy vercor examples tests" in commands
    assert "compileall" in commands
    assert "pytest tests/ -q --tb=short" in commands
    assert "--cov=vercor" in commands
    assert "--cov-branch" in commands
    assert "--cov-fail-under=90" in commands


def test_distribution_helper_reuses_explicit_artifact_directory_without_building(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir = tmp_path / "downloaded-dist"
    artifact_dir.mkdir()
    wheel = artifact_dir / EXPECTED_WHEEL_NAME
    sdist = artifact_dir / EXPECTED_SDIST_NAME
    plugin_wheel = artifact_dir / EXPECTED_PLUGIN_WHEEL_NAME
    frozen_plugin_wheel = artifact_dir / EXPECTED_FROZEN_PLUGIN_WHEEL_NAME
    wheel.touch()
    sdist.touch()
    plugin_wheel.touch()
    frozen_plugin_wheel.touch()

    def unexpected_build(*args: object, **kwargs: object) -> object:
        _ = args, kwargs
        pytest.fail("downloaded artifacts must bypass local build tooling")

    monkeypatch.setattr(subprocess, "run", unexpected_build)

    distributions = build_distributions(
        PROJECT_ROOT,
        tmp_path / "unused-build-output",
        artifact_dir=artifact_dir,
    )

    assert distributions.wheel == wheel
    assert distributions.sdist == sdist
    assert distributions.plugin_wheel == plugin_wheel
    assert distributions.frozen_plugin_wheel == frozen_plugin_wheel
    assert distributions.build_pythonpath == ""


@pytest.mark.parametrize(
    ("wheel_name", "sdist_name", "plugin_wheel_name", "frozen_plugin_wheel_name"),
    (
        (
            "vercor-3.1.0-py3-none-any.whl",
            "vercor-3.1.1.tar.gz",
            EXPECTED_PLUGIN_WHEEL_NAME,
            EXPECTED_FROZEN_PLUGIN_WHEEL_NAME,
        ),
        (
            "vercor-3.1.1-py3-none-any.whl",
            "vercor-3.1.0.tar.gz",
            EXPECTED_PLUGIN_WHEEL_NAME,
            EXPECTED_FROZEN_PLUGIN_WHEEL_NAME,
        ),
        (
            "vercor-3.1.1-py3-none-any.whl",
            "vercor-3.1.1.tar.gz",
            "vercor_public_plugin-0.2.0-py3-none-any.whl",
            EXPECTED_FROZEN_PLUGIN_WHEEL_NAME,
        ),
        (
            "vercor-3.1.1-py3-none-any.whl",
            "vercor-3.1.1.tar.gz",
            EXPECTED_PLUGIN_WHEEL_NAME,
            "vercor_compat_plugin_3_0-0.2.0-py3-none-any.whl",
        ),
    ),
)
def test_distribution_helper_rejects_wrong_artifact_version(
    tmp_path: Path,
    wheel_name: str,
    sdist_name: str,
    plugin_wheel_name: str,
    frozen_plugin_wheel_name: str,
) -> None:
    artifact_dir = tmp_path / "wrong-dist"
    artifact_dir.mkdir()
    (artifact_dir / wheel_name).touch()
    (artifact_dir / sdist_name).touch()
    (artifact_dir / plugin_wheel_name).touch()
    (artifact_dir / frozen_plugin_wheel_name).touch()

    with pytest.raises(ValueError, match=f"VerCOR {EXPECTED_VERSION}"):
        build_distributions(
            PROJECT_ROOT,
            tmp_path / "unused-build-output",
            artifact_dir=artifact_dir,
            plugin_wheel_path=artifact_dir / plugin_wheel_name,
            frozen_plugin_wheel_path=artifact_dir / frozen_plugin_wheel_name,
        )


def test_built_distributions_run_public_plugin_outside_checkout(
    built_distributions: BuiltDistributions,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    distributions = built_distributions

    assert distributions.wheel.name == EXPECTED_WHEEL_NAME
    assert distributions.sdist.name == EXPECTED_SDIST_NAME
    assert distributions.plugin_wheel.name == EXPECTED_PLUGIN_WHEEL_NAME
    assert distributions.frozen_plugin_wheel.name == EXPECTED_FROZEN_PLUGIN_WHEEL_NAME
    with zipfile.ZipFile(distributions.wheel) as wheel:
        wheel_names = set(wheel.namelist())
        metadata_name = next(
            name for name in wheel_names if name.endswith(".dist-info/METADATA")
        )
        metadata = wheel.read(metadata_name).decode("utf-8")
    assert "vercor/py.typed" in wheel_names
    assert f"Version: {EXPECTED_VERSION}" in metadata
    pytest_requirements = [
        line
        for line in metadata.splitlines()
        if line.lower().startswith("requires-dist: pytest")
    ]
    assert pytest_requirements
    assert all("extra ==" in line for line in pytest_requirements)
    assert "Provides-Extra: test" in metadata
    assert "Provides-Extra: dev" in metadata

    for plugin_wheel, requirement in (
        (distributions.plugin_wheel, "Requires-Dist: vercor>=4,<5"),
        (distributions.frozen_plugin_wheel, "Requires-Dist: vercor>=3.0,<4"),
    ):
        with zipfile.ZipFile(plugin_wheel) as plugin_archive:
            plugin_metadata_name = next(
                name
                for name in plugin_archive.namelist()
                if name.endswith(".dist-info/METADATA")
            )
            plugin_metadata = plugin_archive.read(plugin_metadata_name).decode("utf-8")
        assert "Version: 0.1.0" in plugin_metadata
        assert requirement in plugin_metadata

    with tarfile.open(distributions.sdist, "r:gz") as sdist:
        sdist_names = set(sdist.getnames())
    assert f"vercor-{EXPECTED_VERSION}/vercor/py.typed" in sdist_names

    target = tmp_path / "installed-target"
    install_local_target(
        wheel=distributions.wheel,
        plugin_wheel=distributions.plugin_wheel,
        frozen_plugin_wheel=distributions.frozen_plugin_wheel,
        target=target,
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
    assert installed["version"] == EXPECTED_VERSION
    assert Path(installed["typed"]).is_file()

    monkeypatch.setenv("VERCOR_TEST_PACKAGE_ROOT", str(target))
    setup_probe = _run_setup_probe("import vercor")
    setup_probe_path = setup_probe["vercor_file"]
    assert isinstance(setup_probe_path, str)
    assert Path(setup_probe_path).is_relative_to(target)

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
    assert evidence["temperature"] == 13.0
    assert evidence["host_value"] == 14.0
    assert evidence["exchange_forcing"] == 1.0
    assert evidence["state_replacement"] is True
    assert evidence["lifecycle"] == ["user-setup", "hook-setup"]
    assert evidence["topology"] == ["applies", "build"]
    assert evidence["snapshot"] == {"component": "JAX", "temperature": 13.0}

    frozen_smoke = subprocess.run(
        [sys.executable, "-m", "vercor_compat_plugin_3_0.smoke"],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert frozen_smoke.returncode != 0
    assert "Component.from_step" in frozen_smoke.stderr

    mypy_environment = environment.copy()
    mypy_environment["MYPYPATH"] = str(target)
    external_use_site = tmp_path / "public_plugin_use_site.py"
    shutil.copyfile(PLUGIN_ROOT / "use_site.py", external_use_site)
    mypy = subprocess.run(
        [
            str(Path(sys.executable).with_name("mypy")),
            "--strict",
            "--verbose",
            str(target / "vercor_public_plugin"),
            str(external_use_site),
        ],
        cwd=tmp_path,
        env=mypy_environment,
        check=True,
        capture_output=True,
        text=True,
    )
    mypy_evidence = mypy.stdout + mypy.stderr
    assert str(PROJECT_ROOT) not in mypy_evidence
    assert str(target) in mypy_evidence


def test_supplied_wheels_install_and_run_without_build_environment(
    built_distributions: BuiltDistributions,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir = tmp_path / "supplied-artifacts"
    artifact_dir.mkdir()
    for artifact in (
        built_distributions.wheel,
        built_distributions.sdist,
        built_distributions.plugin_wheel,
        built_distributions.frozen_plugin_wheel,
    ):
        shutil.copyfile(artifact, artifact_dir / artifact.name)

    def unavailable_build_environment() -> str:
        pytest.fail("supplied wheels must not inspect build/flit_core/Conda fallback")

    monkeypatch.setattr(
        distribution_support,
        "_cached_build_pythonpath",
        unavailable_build_environment,
    )
    monkeypatch.setenv("VERCOR_ARTIFACT_DIR", str(artifact_dir))
    monkeypatch.setenv(
        "VERCOR_PLUGIN_WHEEL_PATH",
        str(artifact_dir / built_distributions.plugin_wheel.name),
    )
    monkeypatch.setenv(
        "VERCOR_COMPAT_PLUGIN_WHEEL_PATH",
        str(artifact_dir / built_distributions.frozen_plugin_wheel.name),
    )

    supplied = build_distributions(PROJECT_ROOT, tmp_path / "must-not-build")
    target = tmp_path / "clean-installed-target"
    install_local_target(
        wheel=supplied.wheel,
        plugin_wheel=supplied.plugin_wheel,
        frozen_plugin_wheel=supplied.frozen_plugin_wheel,
        target=target,
    )

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(target)
    smoke = subprocess.run(
        [
            sys.executable,
            "-m",
            "vercor_public_plugin.smoke",
            "--output-dir",
            str(tmp_path / "clean-plugin-output"),
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    evidence = json.loads(smoke.stdout.splitlines()[-1])
    assert evidence["temperature"] == 13.0
    assert evidence["host_value"] == 14.0

    frozen_smoke = subprocess.run(
        [sys.executable, "-m", "vercor_compat_plugin_3_0.smoke"],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert frozen_smoke.returncode != 0
    assert "Component.from_step" in frozen_smoke.stderr
