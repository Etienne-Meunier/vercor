"""Offline distribution build/install helpers for artifact-boundary tests."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tomllib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_VERSION = tomllib.loads(
    (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
)["project"]["version"]
EXPECTED_WHEEL_NAME = f"vercor-{EXPECTED_VERSION}-py3-none-any.whl"
EXPECTED_SDIST_NAME = f"vercor-{EXPECTED_VERSION}.tar.gz"
EXPECTED_EXTENSION_FIXTURE_VERSION = "0.1.0"
EXPECTED_EXTENSION_FIXTURE_WHEEL_NAME = (
    "external_extension_test_fixture-"
    f"{EXPECTED_EXTENSION_FIXTURE_VERSION}-py3-none-any.whl"
)


@dataclass(frozen=True)
class BuiltDistributions:
    """Paths to the two publishable VerCOR distributions."""

    wheel: Path
    sdist: Path
    build_pythonpath: str


def _cached_build_pythonpath() -> str:
    """Return local build-backend paths, including an offline Conda cache fallback."""

    if importlib.util.find_spec("build") and importlib.util.find_spec("flit_core"):
        return ""

    environment_root = Path(sys.executable).resolve().parents[3]
    package_cache = environment_root / "pkgs"
    build_paths = sorted(package_cache.glob("python-build-*/site-packages"))
    flit_paths = sorted(package_cache.glob("flit-core-*/site-packages"))
    hook_paths = sorted(package_cache.glob("pyproject_hooks-*/site-packages"))
    if not build_paths or not flit_paths or not hook_paths:
        raise RuntimeError(
            "offline artifact build requires installed build/flit_core or cached "
            f"Conda packages under {package_cache}"
        )
    return os.pathsep.join(
        (str(build_paths[-1]), str(flit_paths[-1]), str(hook_paths[-1]))
    )


def _build_environment() -> tuple[dict[str, str], str]:
    """Return a subprocess environment and any offline backend path."""

    build_pythonpath = _cached_build_pythonpath()
    environment = os.environ.copy()
    if build_pythonpath:
        environment["PYTHONPATH"] = build_pythonpath
    return environment, build_pythonpath


def _existing_distributions(
    wheel: Path,
    sdist: Path,
    *,
    artifact_dir: Path | None = None,
) -> BuiltDistributions:
    """Validate and return externally supplied VerCOR artifacts."""

    if wheel.name != EXPECTED_WHEEL_NAME or sdist.name != EXPECTED_SDIST_NAME:
        raise ValueError(
            f"expected VerCOR {EXPECTED_VERSION} artifacts named "
            f"{EXPECTED_WHEEL_NAME!r} and {EXPECTED_SDIST_NAME!r}"
        )
    if not all(path.is_file() for path in (wheel, sdist)):
        raise ValueError(
            f"VerCOR {EXPECTED_VERSION} artifacts are missing: {wheel}, {sdist}"
        )
    if artifact_dir is not None:
        expected_inventory = {wheel, sdist}
        actual_inventory = set(artifact_dir.iterdir())
        if actual_inventory != expected_inventory:
            raise ValueError(
                f"{artifact_dir} must contain exactly "
                f"{EXPECTED_WHEEL_NAME!r} and {EXPECTED_SDIST_NAME!r}; found "
                f"{sorted(path.name for path in actual_inventory)!r}"
            )
    return BuiltDistributions(wheel, sdist, "")


def build_distributions(
    project_root: Path,
    output_dir: Path,
    *,
    artifact_dir: Path | None = None,
    wheel_path: Path | None = None,
    sdist_path: Path | None = None,
) -> BuiltDistributions:
    """Reuse supplied VerCOR artifacts or build them offline when none are set."""

    direct_configuration = any(
        path is not None for path in (artifact_dir, wheel_path, sdist_path)
    )
    if direct_configuration:
        configured_dir = artifact_dir
        configured_wheel = wheel_path
        configured_sdist = sdist_path
    else:
        configured_dir = (
            Path(os.environ["VERCOR_ARTIFACT_DIR"])
            if os.environ.get("VERCOR_ARTIFACT_DIR")
            else None
        )
        configured_wheel = (
            Path(os.environ["VERCOR_WHEEL_PATH"])
            if os.environ.get("VERCOR_WHEEL_PATH")
            else None
        )
        configured_sdist = (
            Path(os.environ["VERCOR_SDIST_PATH"])
            if os.environ.get("VERCOR_SDIST_PATH")
            else None
        )

    if configured_dir is not None:
        if configured_wheel is not None or configured_sdist is not None:
            raise ValueError(
                "configure either VERCOR_ARTIFACT_DIR or explicit VerCOR paths"
            )
        return _existing_distributions(
            configured_dir / EXPECTED_WHEEL_NAME,
            configured_dir / EXPECTED_SDIST_NAME,
            artifact_dir=configured_dir,
        )
    if configured_wheel is not None or configured_sdist is not None:
        if configured_wheel is None or configured_sdist is None:
            raise ValueError(
                f"VerCOR {EXPECTED_VERSION} wheel and sdist paths are both required"
            )
        return _existing_distributions(configured_wheel, configured_sdist)

    output_dir.mkdir(parents=True, exist_ok=True)
    environment, build_pythonpath = _build_environment()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--no-isolation",
            "--outdir",
            str(output_dir),
            str(project_root),
        ],
        check=True,
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
    )
    validated = _existing_distributions(
        output_dir / EXPECTED_WHEEL_NAME,
        output_dir / EXPECTED_SDIST_NAME,
        artifact_dir=output_dir,
    )
    return BuiltDistributions(
        wheel=validated.wheel,
        sdist=validated.sdist,
        build_pythonpath=build_pythonpath,
    )


def build_external_extension_fixture(
    project_root: Path,
    output_dir: Path,
) -> Path:
    """Build the external extension test fixture outside release artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    environment, _ = _build_environment()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--no-isolation",
            "--wheel",
            "--outdir",
            str(output_dir),
            str(
                project_root / "tests" / "fixtures" / "external_extension_test_fixture"
            ),
        ],
        check=True,
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
    )
    fixture_wheel = output_dir / EXPECTED_EXTENSION_FIXTURE_WHEEL_NAME
    if not fixture_wheel.is_file():
        raise RuntimeError(
            "external extension test fixture build did not produce " f"{fixture_wheel}"
        )
    return fixture_wheel


def install_local_target(
    *,
    wheel: Path,
    target: Path,
    extension_fixture_wheel: Path | None = None,
) -> None:
    """Install VerCOR and an optional extension fixture without dependencies."""

    environment = os.environ.copy()
    target.mkdir(parents=True, exist_ok=True)
    sources = (
        (wheel, extension_fixture_wheel)
        if extension_fixture_wheel is not None
        else (wheel,)
    )
    for source in sources:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-deps",
                "--only-binary=:all:",
                "--target",
                str(target),
                str(source),
            ],
            check=True,
            cwd=target.parent,
            env=environment,
            capture_output=True,
            text=True,
        )
