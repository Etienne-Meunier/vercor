from __future__ import annotations

import importlib
from io import BytesIO
import hashlib
from pathlib import Path
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import vercor.tools as tools_module

from tests.assertions import assert_allclose_compact, assert_array_equal_compact
from vercor.exceptions import AssetError, RegridderError
from vercor.tools import (
    _asset_base_url,
    _download_asset,
    _ensure_forcing_asset,
    check_remap_conservation,
    check_total_lnd_ocn_mask_sum,
    compute_ocn_lnd_masks_on_atm_grid,
    create_lnd_mask_from_ocn,
)
from vercor.grid import RectilinearGrid

conservative_module = importlib.import_module("vercor.regridders.conservative")


def test_asset_base_url_normalizes_and_handles_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tools_module, "VERCOR_ASSETS_BASE_URL", " https://example.test/assets// "
    )
    assert _asset_base_url() == "https://example.test/assets"

    monkeypatch.setattr(tools_module, "VERCOR_ASSETS_BASE_URL", "   ")
    assert _asset_base_url() is None

    monkeypatch.setattr(tools_module, "VERCOR_ASSETS_BASE_URL", None)
    assert _asset_base_url() is None


def test_download_asset_writes_response_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"forcing-content"

    class DummyResponse:
        def __enter__(self) -> BytesIO:
            return BytesIO(payload)

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Literal[False]:
            return False

    monkeypatch.setattr(tools_module, "urlopen", lambda _url: DummyResponse())

    target = tmp_path / "nested" / "asset.nc"
    _download_asset("https://example.test/asset.nc", target)

    assert target.exists()
    assert target.read_bytes() == payload


def test_ensure_forcing_asset_uses_valid_cached_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    filename = "cached.nc"
    payload = b"valid-cached"
    (tmp_path / filename).write_bytes(payload)

    monkeypatch.setattr(tools_module, "_ASSETS_CACHE_DIR", tmp_path)
    monkeypatch.setattr(
        tools_module,
        "_FORCING_ASSETS",
        {"k": {"filename": filename, "md5": hashlib.md5(payload).hexdigest()}},
    )
    monkeypatch.setattr(
        tools_module,
        "_download_asset",
        lambda _url, _target: (_ for _ in ()).throw(
            AssertionError("unexpected download")
        ),
    )

    resolved = _ensure_forcing_asset("k")
    assert resolved == tmp_path / filename


def test_ensure_forcing_asset_downloads_when_cached_md5_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    filename = "downloaded.nc"
    cached = tmp_path / filename
    cached.write_bytes(b"stale")
    downloaded = b"fresh-content"

    monkeypatch.setattr(tools_module, "_ASSETS_CACHE_DIR", tmp_path)
    monkeypatch.setattr(
        tools_module,
        "_FORCING_ASSETS",
        {"k": {"filename": filename, "md5": hashlib.md5(downloaded).hexdigest()}},
    )
    monkeypatch.setattr(
        tools_module, "VERCOR_ASSETS_BASE_URL", "https://example.test/base"
    )

    def fake_download(url: str, target: Path) -> None:
        assert url == "https://example.test/base/downloaded.nc"
        target.write_bytes(downloaded)

    monkeypatch.setattr(tools_module, "_download_asset", fake_download)

    resolved = _ensure_forcing_asset("k")
    assert resolved == cached
    assert cached.read_bytes() == downloaded


def test_ensure_forcing_asset_errors_without_base_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tools_module, "_ASSETS_CACHE_DIR", tmp_path)
    monkeypatch.setattr(
        tools_module,
        "_FORCING_ASSETS",
        {"k": {"filename": "missing.nc", "md5": hashlib.md5(b"x").hexdigest()}},
    )
    monkeypatch.setattr(tools_module, "VERCOR_ASSETS_BASE_URL", None)

    with pytest.raises(AssetError, match="no remote base URL configured"):
        _ensure_forcing_asset("k")


def test_ensure_forcing_asset_raises_and_deletes_on_md5_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    filename = "bad.nc"
    target = tmp_path / filename

    monkeypatch.setattr(tools_module, "_ASSETS_CACHE_DIR", tmp_path)
    monkeypatch.setattr(
        tools_module,
        "_FORCING_ASSETS",
        {"k": {"filename": filename, "md5": hashlib.md5(b"expected").hexdigest()}},
    )
    monkeypatch.setattr(tools_module, "VERCOR_ASSETS_BASE_URL", "https://example.test")
    monkeypatch.setattr(
        tools_module, "_download_asset", lambda _url, tgt: tgt.write_bytes(b"wrong")
    )

    with pytest.raises(AssetError, match="MD5 mismatch"):
        _ensure_forcing_asset("k")

    assert not target.exists()


def test_compute_ocn_lnd_masks_on_atm_grid_clips_and_builds_binary_land_mask() -> None:
    class DummyRegridder:
        def __call__(self, _arr: np.ndarray) -> jax.Array:
            return jnp.asarray([[1.2, -0.2], [0.4, 0.0]])

    ocean_binary_mask = jnp.asarray([[1.0, 0.0], [1.0, 0.0]])
    ocn_fmask, lnd_fmask, lnd_bmask = compute_ocn_lnd_masks_on_atm_grid(
        ocean_binary_mask,
        cast(Any, DummyRegridder()),
    )

    assert_allclose_compact(ocn_fmask, np.array([[1.0, 0.0], [0.4, 0.0]]))
    assert_allclose_compact(lnd_fmask, np.array([[0.0, 1.0], [0.6, 1.0]]))
    assert_array_equal_compact(lnd_bmask, np.array([[0, 1], [1, 1]]))


def test_check_total_lnd_ocn_mask_sum_success_and_failure() -> None:
    lnd_good = jnp.asarray([[0.3, 1.0], [0.0, 0.8]])
    ocn_good = jnp.asarray([[0.7, 0.0], [1.0, 0.2]])
    check_total_lnd_ocn_mask_sum(lnd_good, ocn_good)

    lnd_bad = jnp.asarray([[0.3, 1.0], [0.0, 0.8]])
    ocn_bad = jnp.asarray([[0.7, 0.0], [1.0, 0.0]])
    with pytest.raises(RegridderError, match="must sum to approx. 1"):
        check_total_lnd_ocn_mask_sum(lnd_bad, ocn_bad)


@pytest.mark.fast_always
def test_check_remap_conservation_handles_skip_and_mismatch(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class DummyRemapper:
        def __init__(
            self,
            src_lat_b: np.ndarray,
            dst_lat_b: np.ndarray,
            src_mass: float,
            dst_mass: float,
        ) -> None:
            self.src_lat_b = src_lat_b
            self.dst_lat_b = dst_lat_b
            self._src_mass = src_mass
            self._dst_mass = dst_mass

        def get_src_total_mass(self, _arr: np.ndarray) -> float:
            return self._src_mass

        def get_dst_total_mass(self, _arr: np.ndarray) -> float:
            return self._dst_mass

    class DummyRegridder:
        def __init__(self, interpolator: Any) -> None:
            self.interpolator = interpolator

    monkeypatch.setattr(tools_module, "ConservativeRectilinearRemapper", DummyRemapper)

    skip_interp = DummyRemapper(
        src_lat_b=np.array([-90.0, 0.0, 90.0]),
        dst_lat_b=np.array([-80.0, 0.0, 80.0]),
        src_mass=10.0,
        dst_mass=1.0,
    )
    check_remap_conservation(
        cast(Any, DummyRegridder(skip_interp)),
        np.ones((2, 2)),
        np.ones((2, 2)),
    )
    assert "Skipping mass conservation check" in capsys.readouterr().out

    mismatch_interp = DummyRemapper(
        src_lat_b=np.array([-90.0, 0.0, 90.0]),
        dst_lat_b=np.array([-90.0, 0.0, 90.0]),
        src_mass=10.0,
        dst_mass=9.0,
    )
    with pytest.raises(RegridderError, match="does not conserve total mass"):
        check_remap_conservation(
            cast(Any, DummyRegridder(mismatch_interp)),
            np.ones((2, 2)),
            np.ones((2, 2)),
        )


def test_create_lnd_mask_from_ocn_accepts_jax_backed_masks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyRegridder:
        def __init__(
            self, source_grid: RectilinearGrid, destination_grid: RectilinearGrid
        ):
            self.source_grid = source_grid
            self.destination_grid = destination_grid
            self.interpolator = None

        def __call__(self, _field: jax.Array) -> jax.Array:
            return jnp.asarray([[0.8, 0.1], [0.0, 0.6]])

    monkeypatch.setattr(
        conservative_module,
        "ConservativeRectilinearRegridder",
        DummyRegridder,
    )

    ocn_grid = RectilinearGrid(
        name="OCN",
        longitude=jnp.asarray([0.0, 1.0]),
        latitude=jnp.asarray([0.0, 1.0]),
        binary_mask=jnp.asarray([[1.0, 0.0], [1.0, 0.0]]),
    )

    lnd_bmask, lnd_fmask = create_lnd_mask_from_ocn(
        atm_lat=jnp.asarray([0.0, 1.0]),
        atm_lon=jnp.asarray([0.0, 1.0]),
        ocn_grid=ocn_grid,
    )

    assert_array_equal_compact(lnd_bmask, np.asarray([[1, 1], [1, 1]]))
    assert_allclose_compact(lnd_fmask, np.asarray([[0.2, 0.9], [1.0, 0.4]]))
