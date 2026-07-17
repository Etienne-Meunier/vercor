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
EXPECTED_PLUGIN_VERSION = "0.1.0"
EXPECTED_PLUGIN_WHEEL_NAME = (
    f"vercor_public_plugin-{EXPECTED_PLUGIN_VERSION}-py3-none-any.whl"
)


@dataclass(frozen=True)
class BuiltDistributions:
    """Paths to the VerCOR and native public-plugin distributions."""

    wheel: Path
    sdist: Path
    plugin_wheel: Path
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


def _existing_distributions(
    wheel: Path,
    sdist: Path,
    plugin_wheel: Path,
) -> BuiltDistributions:
    """Validate and return externally supplied VerCOR and plugin artifacts."""

    if (
        wheel.name != EXPECTED_WHEEL_NAME
        or sdist.name != EXPECTED_SDIST_NAME
        or plugin_wheel.name != EXPECTED_PLUGIN_WHEEL_NAME
    ):
        raise ValueError(
            f"expected VerCOR {EXPECTED_VERSION} artifacts named "
            f"{EXPECTED_WHEEL_NAME!r}, {EXPECTED_SDIST_NAME!r}, and "
            f"{EXPECTED_PLUGIN_WHEEL_NAME!r}"
        )
    if not all(path.is_file() for path in (wheel, sdist, plugin_wheel)):
        raise ValueError(
            f"VerCOR {EXPECTED_VERSION} or plugin artifacts are missing: "
            f"{wheel}, {sdist}, {plugin_wheel}"
        )
    return BuiltDistributions(wheel, sdist, plugin_wheel, "")


def build_distributions(
    project_root: Path,
    output_dir: Path,
    *,
    artifact_dir: Path | None = None,
    wheel_path: Path | None = None,
    sdist_path: Path | None = None,
    plugin_wheel_path: Path | None = None,
) -> BuiltDistributions:
    """Reuse supplied artifacts or build them offline when none are configured."""

    direct_configuration = any(
        path is not None
        for path in (
            artifact_dir,
            wheel_path,
            sdist_path,
            plugin_wheel_path,
        )
    )
    if direct_configuration:
        configured_dir = artifact_dir
        configured_wheel = wheel_path
        configured_sdist = sdist_path
        configured_plugin_wheel = plugin_wheel_path
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
        configured_plugin_wheel = (
            Path(os.environ["VERCOR_PLUGIN_WHEEL_PATH"])
            if os.environ.get("VERCOR_PLUGIN_WHEEL_PATH")
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
            (
                configured_plugin_wheel
                if configured_plugin_wheel is not None
                else configured_dir / EXPECTED_PLUGIN_WHEEL_NAME
            ),
        )
    if any(
        path is not None
        for path in (
            configured_wheel,
            configured_sdist,
            configured_plugin_wheel,
        )
    ):
        if (
            configured_wheel is None
            or configured_sdist is None
            or configured_plugin_wheel is None
        ):
            raise ValueError(
                f"VerCOR {EXPECTED_VERSION} wheel/sdist and plugin wheel paths "
                "are all required"
            )
        return _existing_distributions(
            configured_wheel,
            configured_sdist,
            configured_plugin_wheel,
        )

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
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--no-isolation",
            "--wheel",
            "--outdir",
            str(output_dir),
            str(project_root / "tests" / "fixtures" / "public_plugin"),
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
        output_dir / EXPECTED_PLUGIN_WHEEL_NAME,
    )
    return BuiltDistributions(
        wheel=validated.wheel,
        sdist=validated.sdist,
        plugin_wheel=validated.plugin_wheel,
        build_pythonpath=build_pythonpath,
    )


def install_local_target(
    *,
    wheel: Path,
    plugin_wheel: Path,
    target: Path,
) -> None:
    """Install VerCOR and the native 0.4 plugin without dependencies."""

    environment = os.environ.copy()
    target.mkdir(parents=True, exist_ok=True)
    for source in (wheel, plugin_wheel):
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
