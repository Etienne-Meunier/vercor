from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
from urllib.request import urlopen

from vercor.exceptions import AssetError

VERCOR_ASSETS_BASE_URL = (
    os.environ.get("VERCOR_ASSETS_BASE_URL")
    or "https://sid.erda.dk/share_redirect/bC5N6nQcbY/"
)

_ASSETS_CACHE_DIR = Path.home() / ".vercor" / "assets"

_FORCING_ASSETS: dict[str, dict[str, str]] = {
    "era5_model_levels": {
        "filename": "era5_198x_ml_4x4deg_monthly_mean.nc",
        "md5": "2ada464b2eb2bf3a7abec7f77a18634c",
    },
    "era5_surface": {
        "filename": "era5_198x_sfc_4x4deg_monthly_mean.nc",
        "md5": "304d547b72b3677f7bc44c71bcf7cb8f",
    },
    "era5_land": {
        "filename": "era5_lnd_skt_1980.nc",
        "md5": "b0877a7715c438b7a17593ad00bb8218",
    },
    "era5_land_masked": {
        "filename": "era5_lnd_skt_masked_1980.nc",
        "md5": "cea9349ee88f1ecb55572f87f065ff9b",
    },
    "erainterim_ocean_4deg": {
        "filename": "forcing_4deg_global_open_itf.nc",
        "md5": "cfcc6d8cde8da5a74ecec00309d92dd7",
    },
    "erainterim_ocean_1deg": {
        "filename": "forcing_1deg_global.nc",
        "md5": "1fc86f88acd820da078c8da5873cfa01",
    },
    "ecmwf_4deg_monthly": {
        "filename": "ecmwf_4deg_monthly_nc4.nc",
        "md5": "d1b4e0e199d7a5883cf7c88d3d6bcb27",
    },
}


def _md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_asset(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as response, target.open("wb") as out:
        shutil.copyfileobj(response, out)


def _asset_base_url() -> str | None:
    base_url = VERCOR_ASSETS_BASE_URL
    if base_url is None:
        return None
    stripped = base_url.strip().rstrip("/")
    return stripped if stripped else None


def _ensure_forcing_asset(asset_key: str) -> Path:
    asset = _FORCING_ASSETS[asset_key]
    filename = asset["filename"]
    expected_md5 = asset["md5"]

    cached_path = _ASSETS_CACHE_DIR / filename
    if cached_path.exists():
        if _md5sum(cached_path) == expected_md5:
            return cached_path
        cached_path.unlink()

    _ASSETS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    base_url = _asset_base_url()
    if base_url is None:
        raise AssetError(
            "Forcing asset not found in cache and no remote base URL configured. "
            "Set VERCOR_ASSETS_BASE_URL to a server hosting VerCOR forcing assets. "
            f"Missing asset: '{filename}'"
        )

    url = f"{base_url}/{filename}"
    try:
        _download_asset(url, cached_path)
    except Exception as e:
        raise AssetError(
            f"Failed to download forcing asset '{filename}' from '{url}': {e}"
        ) from e

    actual_md5 = _md5sum(cached_path)
    if actual_md5 != expected_md5:
        if cached_path.exists():
            cached_path.unlink()
        raise AssetError(
            f"MD5 mismatch for forcing asset '{filename}': expected {expected_md5}, got {actual_md5}"
        )

    return cached_path


def get_forcing_data(file_type: str) -> Path:
    """Resolve forcing data to cached assets in $HOME/.vercor/assets."""

    if file_type not in _FORCING_ASSETS:
        allowed = ", ".join(sorted(_FORCING_ASSETS.keys()))
        raise AssetError(
            f"Unknown file_type '{file_type}'. Allowed values are: {allowed}"
        )

    return _ensure_forcing_asset(file_type)
