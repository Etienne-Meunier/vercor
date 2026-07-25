"""Wait for one exact GitHub Release mutation to become list-visible."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

VALIDATOR_PATH = Path(__file__).with_name("validate_release_state.py")
RELEASE_STATES = ("absent", "draft", "published")


def _parser() -> argparse.ArgumentParser:
    """Build the bounded release-visibility polling interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--notes-file", type=Path, required=True)
    parser.add_argument("--expect", nargs="+", required=True)
    parser.add_argument("--target-state", choices=RELEASE_STATES, required=True)
    parser.add_argument("--target-present", nargs="*", default=[])
    parser.add_argument("--transitional-state", choices=RELEASE_STATES, required=True)
    parser.add_argument("--transitional-present", nargs="*", default=[])
    parser.add_argument("--release-id", type=int)
    parser.add_argument("--attempts", type=int, default=12)
    parser.add_argument("--interval-seconds", type=float, default=2.0)
    parser.add_argument("--state-output", type=Path, required=True)
    return parser


def _validate_arguments(arguments: argparse.Namespace) -> None:
    """Reject ambiguous targets before querying GitHub."""

    repository_parts = arguments.repository.split("/")
    if (
        len(repository_parts) != 2
        or any(not part for part in repository_parts)
        or any(
            not all(character.isalnum() or character in "._-" for character in part)
            for part in repository_parts
        )
    ):
        raise ValueError("repository must be OWNER/REPOSITORY")
    expected = arguments.expect
    if len(expected) != len(set(expected)):
        raise ValueError("expected asset names must be unique")
    expected_names = set(expected)
    for label, state, present in (
        ("target", arguments.target_state, arguments.target_present),
        (
            "transitional",
            arguments.transitional_state,
            arguments.transitional_present,
        ),
    ):
        if len(present) != len(set(present)):
            raise ValueError(f"{label} present asset names must be unique")
        if not set(present).issubset(expected_names):
            raise ValueError(f"{label} present assets must be expected assets")
        if state == "absent" and present:
            raise ValueError(f"{label} absent state cannot contain assets")
    target = (arguments.target_state, arguments.target_present)
    transitional = (
        arguments.transitional_state,
        arguments.transitional_present,
    )
    if target == transitional:
        raise ValueError("target and transitional states must differ")
    if arguments.release_id is not None and arguments.release_id <= 0:
        raise ValueError("release id must be positive")
    if arguments.attempts <= 0:
        raise ValueError("attempts must be positive")
    if not math.isfinite(arguments.interval_seconds):
        raise ValueError("interval seconds must be finite")
    if arguments.interval_seconds < 0:
        raise ValueError("interval seconds must not be negative")


def _list_releases(repository: str, destination: Path) -> None:
    """Save one authenticated, paginated GitHub Release listing."""

    with destination.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            [
                "gh",
                "api",
                "--paginate",
                "--slurp",
                f"repos/{repository}/releases?per_page=100",
            ],
            text=True,
            stdout=stream,
            stderr=subprocess.PIPE,
            check=False,
        )
    if completed.returncode:
        if completed.stderr:
            sys.stderr.write(completed.stderr)
        raise RuntimeError(
            f"GitHub Release listing failed with exit code {completed.returncode}"
        )


def _validate_snapshot(
    arguments: argparse.Namespace,
    releases_path: Path,
    state_path: Path,
) -> dict[str, Any]:
    """Validate one listing with the canonical fail-closed validator."""

    allowed_states = list(
        dict.fromkeys(
            (arguments.target_state, arguments.transitional_state),
        )
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR_PATH),
            "github-releases",
            "--json",
            str(releases_path),
            "--manifest",
            str(arguments.manifest),
            "--tag",
            arguments.tag,
            "--title",
            arguments.title,
            "--notes-file",
            str(arguments.notes_file),
            "--expect",
            *arguments.expect,
            "--allow-state",
            *allowed_states,
            "--state-output",
            str(state_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        if completed.stdout:
            sys.stdout.write(completed.stdout)
        if completed.stderr:
            sys.stderr.write(completed.stderr)
        raise RuntimeError(
            f"GitHub Release validation failed with exit code {completed.returncode}"
        )
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("validator state output must be a JSON object")
    return payload


def _state_identity(payload: dict[str, Any]) -> tuple[str, list[str]]:
    """Return the validated state and ordered present-asset list."""

    state = payload.get("state")
    present = payload.get("present")
    if (
        not isinstance(state, str)
        or not isinstance(present, list)
        or not all(isinstance(name, str) for name in present)
    ):
        raise ValueError("validator returned malformed release state")
    return state, present


def _poll(arguments: argparse.Namespace) -> None:
    """Poll only while the exact expected pre-mutation state remains visible."""

    _validate_arguments(arguments)
    target = (arguments.target_state, arguments.target_present)
    transitional = (
        arguments.transitional_state,
        arguments.transitional_present,
    )
    with tempfile.TemporaryDirectory(prefix="vercor-github-release-") as directory:
        state_directory = Path(directory)
        releases_path = state_directory / "releases.json"
        state_path = state_directory / "release-state.json"
        for attempt in range(1, arguments.attempts + 1):
            _list_releases(arguments.repository, releases_path)
            payload = _validate_snapshot(arguments, releases_path, state_path)
            identity = _state_identity(payload)
            release_id = payload.get("release_id")
            if (
                arguments.release_id is not None
                and identity[0] != "absent"
                and release_id != arguments.release_id
            ):
                raise ValueError(
                    "unexpected GitHub Release id: "
                    f"expected {arguments.release_id}, got {release_id}"
                )
            if identity == target:
                arguments.state_output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(state_path, arguments.state_output)
                return
            if identity != transitional:
                raise ValueError(
                    "unexpected validated GitHub Release state: "
                    f"{identity[0]} with assets {identity[1]}"
                )
            if attempt < arguments.attempts:
                time.sleep(arguments.interval_seconds)
        raise TimeoutError(
            "GitHub Release visibility timed out after "
            f"{arguments.attempts} attempts while state remained "
            f"{transitional[0]} with assets {transitional[1]}"
        )


def main() -> int:
    """Run the command-line release-visibility gate."""

    arguments = _parser().parse_args()
    try:
        _poll(arguments)
    except (OSError, RuntimeError, TimeoutError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
