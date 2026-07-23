"""Executable contracts for fail-closed release recovery state validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = PROJECT_ROOT / "tools" / "validate_release_state.py"
WHEEL = "vercor-0.4.0-py3-none-any.whl"
SDIST = "vercor-0.4.0.tar.gz"
UNEXPECTED = "unexpected-0.4.0-py3-none-any.whl"


def _run_validator(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Run the tracked release-state validator as a maintainer would."""

    return subprocess.run(
        [sys.executable, str(VALIDATOR_PATH), *arguments],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _write_manifest(path: Path) -> None:
    """Write deterministic synthetic release hashes."""

    path.write_text(f"{'a' * 64}  {WHEEL}\n{'b' * 64}  {SDIST}\n", encoding="utf-8")


@pytest.mark.fast_always
@pytest.mark.parametrize("present", [WHEEL, SDIST])
def test_pypi_recovery_state_requires_exact_expected_filename_set(
    tmp_path: Path,
    present: str,
) -> None:
    """Reject an unexpected third PyPI file even when the expected file is valid."""

    manifest = tmp_path / "SHA256SUMS"
    _write_manifest(manifest)
    expected_digest = "a" * 64 if present == WHEEL else "b" * 64
    payload = tmp_path / "pypi.json"
    payload.write_text(
        json.dumps(
            {
                "urls": [
                    {"filename": present, "digests": {"sha256": expected_digest}},
                    {
                        "filename": UNEXPECTED,
                        "digests": {"sha256": "c" * 64},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    completed = _run_validator(
        "pypi",
        "--json",
        str(payload),
        "--manifest",
        str(manifest),
        "--expect",
        present,
    )

    assert completed.returncode != 0
    assert "exact filename set" in completed.stderr


@pytest.mark.fast_always
@pytest.mark.parametrize("present", [WHEEL, SDIST])
def test_hosted_recovery_state_requires_exact_expected_asset_set(
    tmp_path: Path,
    present: str,
) -> None:
    """Reject an unexpected third hosted asset beside the expected asset."""

    payload = tmp_path / "release.json"
    payload.write_text(
        json.dumps({"assets": [{"name": present}, {"name": UNEXPECTED}]}),
        encoding="utf-8",
    )

    completed = _run_validator(
        "assets",
        "--json",
        str(payload),
        "--expect",
        present,
    )

    assert completed.returncode != 0
    assert "exact asset set" in completed.stderr


@pytest.mark.fast_always
def test_release_state_validator_accepts_exact_sets_and_verified_bytes(
    tmp_path: Path,
) -> None:
    """Accept exact package/asset sets and bytes matching the manifest."""

    downloaded = tmp_path / "downloaded"
    downloaded.mkdir()
    wheel = downloaded / WHEEL
    sdist = downloaded / SDIST
    wheel.write_bytes(b"wheel")
    sdist.write_bytes(b"sdist")
    manifest = tmp_path / "SHA256SUMS"
    completed_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (wheel, sdist)
    }
    manifest.write_text(
        "".join(f"{completed_hashes[name]}  {name}\n" for name in (WHEEL, SDIST)),
        encoding="utf-8",
    )
    pypi = tmp_path / "pypi.json"
    pypi.write_text(
        json.dumps(
            {
                "urls": [
                    {
                        "filename": name,
                        "digests": {"sha256": completed_hashes[name]},
                    }
                    for name in (WHEEL, SDIST)
                ]
            }
        ),
        encoding="utf-8",
    )
    release = tmp_path / "release.json"
    release.write_text(
        json.dumps({"assets": [{"name": WHEEL}, {"name": SDIST}]}),
        encoding="utf-8",
    )

    commands = (
        (
            "pypi",
            "--json",
            str(pypi),
            "--manifest",
            str(manifest),
            "--expect",
            WHEEL,
            SDIST,
        ),
        ("assets", "--json", str(release), "--expect", WHEEL, SDIST),
        (
            "files",
            "--directory",
            str(downloaded),
            "--manifest",
            str(manifest),
            "--expect",
            WHEEL,
            SDIST,
        ),
    )
    for command in commands:
        completed = _run_validator(*command)
        assert completed.returncode == 0, completed.stderr
