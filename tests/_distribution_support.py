"""Offline distribution build/install helpers for artifact-boundary tests."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import subprocess
import sys


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


def build_distributions(project_root: Path, output_dir: Path) -> BuiltDistributions:
    """Build wheel and sdist without network access or isolated downloads."""

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
