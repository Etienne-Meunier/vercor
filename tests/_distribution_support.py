"""Offline distribution build/install helpers for artifact-boundary tests."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

EXPECTED_VERSION = "3.0.0"
EXPECTED_WHEEL_NAME = f"vercor-{EXPECTED_VERSION}-py3-none-any.whl"
EXPECTED_SDIST_NAME = f"vercor-{EXPECTED_VERSION}.tar.gz"


@dataclass(frozen=True)
class BuiltDistributions:
    """Paths to one locally built wheel and source distribution."""

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


def _existing_distributions(wheel: Path, sdist: Path) -> BuiltDistributions:
    """Validate and return externally supplied VerCOR 3.0 artifacts."""

    if wheel.name != EXPECTED_WHEEL_NAME or sdist.name != EXPECTED_SDIST_NAME:
        raise ValueError(
            "expected VerCOR 3.0.0 artifacts named "
            f"{EXPECTED_WHEEL_NAME!r} and {EXPECTED_SDIST_NAME!r}"
        )
    if not wheel.is_file() or not sdist.is_file():
        raise ValueError(f"VerCOR 3.0.0 artifacts are missing: {wheel}, {sdist}")
    return BuiltDistributions(wheel=wheel, sdist=sdist, build_pythonpath="")


def build_distributions(
    project_root: Path,
    output_dir: Path,
    *,
    artifact_dir: Path | None = None,
    wheel_path: Path | None = None,
    sdist_path: Path | None = None,
) -> BuiltDistributions:
    """Reuse supplied artifacts or build them offline when none are configured."""

    configured_dir = artifact_dir
    if configured_dir is None and os.environ.get("VERCOR_ARTIFACT_DIR"):
        configured_dir = Path(os.environ["VERCOR_ARTIFACT_DIR"])
    configured_wheel = wheel_path
    if configured_wheel is None and os.environ.get("VERCOR_WHEEL_PATH"):
        configured_wheel = Path(os.environ["VERCOR_WHEEL_PATH"])
    configured_sdist = sdist_path
    if configured_sdist is None and os.environ.get("VERCOR_SDIST_PATH"):
        configured_sdist = Path(os.environ["VERCOR_SDIST_PATH"])

    if configured_dir is not None:
        if configured_wheel is not None or configured_sdist is not None:
            raise ValueError(
                "configure either VERCOR_ARTIFACT_DIR or explicit wheel/sdist paths"
            )
        return _existing_distributions(
            configured_dir / EXPECTED_WHEEL_NAME,
            configured_dir / EXPECTED_SDIST_NAME,
        )
    if configured_wheel is not None or configured_sdist is not None:
        if configured_wheel is None or configured_sdist is None:
            raise ValueError("both VerCOR 3.0.0 wheel and sdist paths are required")
        return _existing_distributions(configured_wheel, configured_sdist)

    output_dir.mkdir(parents=True, exist_ok=True)
    build_pythonpath = _cached_build_pythonpath()
    environment = os.environ.copy()
    if build_pythonpath:
        environment["PYTHONPATH"] = build_pythonpath
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
    wheels = tuple(output_dir.glob("*.whl"))
    sdists = tuple(output_dir.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise RuntimeError(
            f"expected one wheel and one sdist, found {wheels!r} and {sdists!r}"
        )
    return BuiltDistributions(
        wheel=wheels[0],
        sdist=sdists[0],
        build_pythonpath=build_pythonpath,
    )


def install_local_target(
    *,
    wheel: Path,
    plugin_root: Path,
    target: Path,
    build_pythonpath: str,
) -> None:
    """Install local artifacts into a target directory without dependencies."""

    environment = os.environ.copy()
    if build_pythonpath:
        environment["PYTHONPATH"] = build_pythonpath
    target.mkdir(parents=True, exist_ok=True)
    for source in (wheel, plugin_root):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-deps",
                "--no-build-isolation",
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
