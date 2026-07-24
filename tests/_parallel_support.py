"""Private support for deterministic pytest worker isolation."""

from __future__ import annotations

from collections.abc import MutableMapping
from pathlib import Path

_DEFAULTED_ENV_MARKER = "VERCOR_TEST_DEFAULTED_ENV"
_MANAGED_ENV_KEYS = ("MPLBACKEND", "MPLCONFIGDIR", "XDG_CACHE_HOME")


def configure_test_cache_environment(
    environ: MutableMapping[str, str],
    *,
    cache_root: Path,
    worker_id: str | None,
) -> dict[str, bool]:
    """Set serial or worker-local defaults while preserving explicit values."""

    inherited_marker = environ.get(_DEFAULTED_ENV_MARKER)
    if inherited_marker is None:
        defaulted = {key: key not in environ for key in _MANAGED_ENV_KEYS}
        environ[_DEFAULTED_ENV_MARKER] = ",".join(
            key for key in _MANAGED_ENV_KEYS if defaulted[key]
        )
    else:
        inherited_defaults = set(filter(None, inherited_marker.split(",")))
        defaulted = {key: key in inherited_defaults for key in _MANAGED_ENV_KEYS}

    if defaulted["MPLBACKEND"]:
        environ["MPLBACKEND"] = "Agg"

    worker_suffix = f"-{worker_id}" if worker_id is not None else ""
    path_defaults = {
        "MPLCONFIGDIR": f"vercor-matplotlib-cache{worker_suffix}",
        "XDG_CACHE_HOME": f"vercor-xdg-cache{worker_suffix}",
    }
    for key, directory_name in path_defaults.items():
        if defaulted[key]:
            environ[key] = str(cache_root / directory_name)

    return defaulted
