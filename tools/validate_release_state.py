"""Fail-closed validation for VerCOR release recovery state.

This script is intentionally standard-library-only so maintainers can validate
fresh PyPI JSON, GitHub release JSON, and downloaded hosted assets before a
recovery mutation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object or terminate with a concise error."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _read_manifest(path: Path) -> dict[str, str]:
    """Read a shasum-compatible manifest and reject duplicate names."""

    manifest: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        if name in manifest:
            raise ValueError(f"{path}: duplicate manifest name: {name}")
        manifest[name] = digest
    return manifest


def _require_exact_names(
    actual_names: list[str],
    expected_names: list[str],
    *,
    description: str,
) -> None:
    """Require an exact, duplicate-free filename set."""

    if len(actual_names) != len(set(actual_names)):
        raise ValueError(f"{description}: duplicate names: {actual_names}")
    if set(actual_names) != set(expected_names) or len(actual_names) != len(
        expected_names
    ):
        raise ValueError(
            f"{description}: expected exact {description} "
            f"{sorted(expected_names)}, got {sorted(actual_names)}"
        )


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_pypi(arguments: argparse.Namespace) -> None:
    """Validate an exact PyPI filename set and every expected digest."""

    payload = _read_json(arguments.json)
    urls = payload.get("urls")
    if not isinstance(urls, list) or not all(isinstance(item, dict) for item in urls):
        raise ValueError(f"{arguments.json}: expected a PyPI urls list")
    filenames = [str(item.get("filename")) for item in urls]
    _require_exact_names(
        filenames,
        arguments.expect,
        description="filename set",
    )
    manifest = _read_manifest(arguments.manifest)
    for item in urls:
        filename = str(item["filename"])
        digests = item.get("digests")
        if not isinstance(digests, dict):
            raise ValueError(f"{filename}: missing PyPI digests")
        expected_digest = manifest.get(filename)
        if expected_digest is None:
            raise ValueError(f"{filename}: missing from checksum manifest")
        if digests.get("sha256") != expected_digest:
            raise ValueError(f"{filename}: PyPI SHA-256 does not match manifest")


def _validate_assets(arguments: argparse.Namespace) -> None:
    """Validate an exact hosted-release asset-name set."""

    payload = _read_json(arguments.json)
    assets = payload.get("assets")
    if not isinstance(assets, list) or not all(
        isinstance(item, dict) for item in assets
    ):
        raise ValueError(f"{arguments.json}: expected a release assets list")
    names = [str(item.get("name")) for item in assets]
    _require_exact_names(names, arguments.expect, description="asset set")


def _validate_files(arguments: argparse.Namespace) -> None:
    """Validate exact downloaded filenames and their manifest digests."""

    actual_names = sorted(
        path.name for path in arguments.directory.iterdir() if path.is_file()
    )
    _require_exact_names(actual_names, arguments.expect, description="file set")
    manifest = _read_manifest(arguments.manifest)
    for name in arguments.expect:
        expected_digest = manifest.get(name)
        if expected_digest is None:
            raise ValueError(f"{name}: missing from checksum manifest")
        if _sha256(arguments.directory / name) != expected_digest:
            raise ValueError(f"{name}: downloaded SHA-256 does not match manifest")


def _validate_differs(arguments: argparse.Namespace) -> None:
    """Require a downloaded selected asset to differ from the manifest."""

    manifest = _read_manifest(arguments.manifest)
    expected_digest = manifest.get(arguments.name)
    if expected_digest is None:
        raise ValueError(f"{arguments.name}: missing from checksum manifest")
    if arguments.file.name != arguments.name:
        raise ValueError(
            f"selected filename mismatch: {arguments.file.name} != {arguments.name}"
        )
    if _sha256(arguments.file) == expected_digest:
        raise ValueError(
            f"{arguments.name}: selected asset unexpectedly matches manifest"
        )


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    pypi = subparsers.add_parser("pypi")
    pypi.add_argument("--json", type=Path, required=True)
    pypi.add_argument("--manifest", type=Path, required=True)
    pypi.add_argument("--expect", nargs="+", required=True)
    pypi.set_defaults(run=_validate_pypi)

    assets = subparsers.add_parser("assets")
    assets.add_argument("--json", type=Path, required=True)
    assets.add_argument("--expect", nargs="+", required=True)
    assets.set_defaults(run=_validate_assets)

    files = subparsers.add_parser("files")
    files.add_argument("--directory", type=Path, required=True)
    files.add_argument("--manifest", type=Path, required=True)
    files.add_argument("--expect", nargs="+", required=True)
    files.set_defaults(run=_validate_files)

    differs = subparsers.add_parser("differs")
    differs.add_argument("--file", type=Path, required=True)
    differs.add_argument("--manifest", type=Path, required=True)
    differs.add_argument("--name", required=True)
    differs.set_defaults(run=_validate_differs)
    return parser


def main() -> int:
    """Validate one release-state boundary."""

    arguments = _parser().parse_args()
    try:
        arguments.run(arguments)
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
