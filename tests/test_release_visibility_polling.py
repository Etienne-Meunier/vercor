"""Behavioral contracts for bounded GitHub Release visibility polling."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tests._distribution_support import (
    EXPECTED_SDIST_NAME,
    EXPECTED_VERSION,
    EXPECTED_WHEEL_NAME,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLLER_PATH = PROJECT_ROOT / "tools" / "wait_for_github_release_state.py"


def _release_fixture(
    present: tuple[str, ...] = (),
    *,
    draft: object = True,
    **overrides: object,
) -> list[list[dict[str, object]]]:
    """Return one paginated exact-tag release-list snapshot."""

    digests = {
        EXPECTED_WHEEL_NAME: "a" * 64,
        EXPECTED_SDIST_NAME: "b" * 64,
    }
    release: dict[str, object] = {
        "id": 42,
        "tag_name": f"v{EXPECTED_VERSION}",
        "name": f"VerCOR {EXPECTED_VERSION}",
        "body": "Release notes.\n",
        "draft": draft,
        "prerelease": False,
        "assets": [
            {
                "id": index,
                "name": name,
                "state": "uploaded",
                "digest": f"sha256:{digests[name]}",
            }
            for index, name in enumerate(present, start=10)
        ],
    }
    release.update(overrides)
    return [[release]]


def _run_poller(
    tmp_path: Path,
    snapshots: list[object],
    *,
    target_state: str,
    target_present: tuple[str, ...],
    transitional_state: str,
    transitional_present: tuple[str, ...],
    attempts: int = 3,
    release_id: int | None = None,
    repository: str = "example/vercor",
    interval_seconds: str = "0",
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    """Run the real poller against a deterministic fake GitHub CLI."""

    sequence_dir = tmp_path / "sequence"
    sequence_dir.mkdir()
    for index, snapshot in enumerate(snapshots):
        (sequence_dir / f"{index}.json").write_text(
            json.dumps(snapshot),
            encoding="utf-8",
        )
    counter_path = tmp_path / "gh-counter"
    counter_path.write_text("0\n", encoding="utf-8")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_gh = fake_bin / "gh"
    fake_gh.write_text(
        """#!/usr/bin/env python3
import os
from pathlib import Path
import sys

expected = [
    "api",
    "--paginate",
    "--slurp",
    f"repos/{os.environ['FAKE_GH_REPOSITORY']}/releases?per_page=100",
]
if sys.argv[1:] != expected:
    raise SystemExit(f"unexpected gh arguments: {sys.argv[1:]!r}")
counter = Path(os.environ["FAKE_GH_COUNTER"])
index = int(counter.read_text(encoding="utf-8"))
sequence = Path(os.environ["FAKE_GH_SEQUENCE"])
paths = sorted(sequence.glob("*.json"))
selected = paths[min(index, len(paths) - 1)]
counter.write_text(f"{index + 1}\\n", encoding="utf-8")
sys.stdout.write(selected.read_text(encoding="utf-8"))
""",
        encoding="utf-8",
    )
    fake_gh.chmod(0o755)

    manifest = tmp_path / "SHA256SUMS"
    manifest.write_text(
        f"{'a' * 64}  {EXPECTED_WHEEL_NAME}\n" f"{'b' * 64}  {EXPECTED_SDIST_NAME}\n",
        encoding="utf-8",
    )
    notes = tmp_path / "notes.md"
    notes.write_text("Release notes.\n", encoding="utf-8")
    state_output = tmp_path / "release-state.json"
    command = [
        sys.executable,
        str(POLLER_PATH),
        "--repository",
        repository,
        "--manifest",
        str(manifest),
        "--tag",
        f"v{EXPECTED_VERSION}",
        "--title",
        f"VerCOR {EXPECTED_VERSION}",
        "--notes-file",
        str(notes),
        "--expect",
        EXPECTED_WHEEL_NAME,
        EXPECTED_SDIST_NAME,
        "--target-state",
        target_state,
        "--target-present",
        *target_present,
        "--transitional-state",
        transitional_state,
        "--transitional-present",
        *transitional_present,
        "--attempts",
        str(attempts),
        "--interval-seconds",
        interval_seconds,
        "--state-output",
        str(state_output),
    ]
    if release_id is not None:
        command.extend(("--release-id", str(release_id)))
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env={
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "FAKE_GH_COUNTER": str(counter_path),
            "FAKE_GH_REPOSITORY": repository,
            "FAKE_GH_SEQUENCE": str(sequence_dir),
        },
        text=True,
        capture_output=True,
        check=False,
    )
    return completed, state_output, counter_path


@pytest.mark.fast_always
@pytest.mark.parametrize(
    (
        "snapshots",
        "target_state",
        "target_present",
        "transitional_state",
        "transitional_present",
        "release_id",
    ),
    [
        (
            [[[]], _release_fixture()],
            "draft",
            (),
            "absent",
            (),
            None,
        ),
        (
            [
                _release_fixture(),
                _release_fixture((EXPECTED_WHEEL_NAME,)),
            ],
            "draft",
            (EXPECTED_WHEEL_NAME,),
            "draft",
            (),
            42,
        ),
        (
            [
                _release_fixture((EXPECTED_WHEEL_NAME,)),
                _release_fixture((EXPECTED_WHEEL_NAME, EXPECTED_SDIST_NAME)),
            ],
            "draft",
            (EXPECTED_WHEEL_NAME, EXPECTED_SDIST_NAME),
            "draft",
            (EXPECTED_WHEEL_NAME,),
            42,
        ),
        (
            [
                _release_fixture((EXPECTED_SDIST_NAME,)),
                _release_fixture((EXPECTED_WHEEL_NAME, EXPECTED_SDIST_NAME)),
            ],
            "draft",
            (EXPECTED_WHEEL_NAME, EXPECTED_SDIST_NAME),
            "draft",
            (EXPECTED_SDIST_NAME,),
            42,
        ),
        (
            [
                _release_fixture((EXPECTED_WHEEL_NAME, EXPECTED_SDIST_NAME)),
                _release_fixture(
                    (EXPECTED_WHEEL_NAME, EXPECTED_SDIST_NAME),
                    draft=False,
                ),
            ],
            "published",
            (EXPECTED_WHEEL_NAME, EXPECTED_SDIST_NAME),
            "draft",
            (EXPECTED_WHEEL_NAME, EXPECTED_SDIST_NAME),
            42,
        ),
    ],
)
def test_poller_retries_only_expected_transition(
    tmp_path: Path,
    snapshots: list[object],
    target_state: str,
    target_present: tuple[str, ...],
    transitional_state: str,
    transitional_present: tuple[str, ...],
    release_id: int | None,
) -> None:
    """Reach each exact post-mutation state through one stale snapshot."""

    completed, state_output, counter_path = _run_poller(
        tmp_path,
        snapshots,
        target_state=target_state,
        target_present=target_present,
        transitional_state=transitional_state,
        transitional_present=transitional_present,
        release_id=release_id,
    )

    assert completed.returncode == 0, completed.stderr
    assert int(counter_path.read_text(encoding="utf-8")) == 2
    state = json.loads(state_output.read_text(encoding="utf-8"))
    assert state["state"] == target_state
    assert state["present"] == list(target_present)
    if release_id is not None:
        assert state["release_id"] == release_id


@pytest.mark.fast_always
@pytest.mark.parametrize(
    ("first_snapshot", "error"),
    [
        (
            [
                _release_fixture()[0],
                _release_fixture()[0],
            ],
            "duplicate exact-tag releases",
        ),
        (
            _release_fixture(draft="yes"),
            "release draft state is not boolean",
        ),
        (
            _release_fixture(name="Wrong title"),
            "unexpected release title",
        ),
        (
            _release_fixture((EXPECTED_WHEEL_NAME,), id=43),
            "unexpected GitHub Release id",
        ),
        (
            _release_fixture((EXPECTED_WHEEL_NAME, EXPECTED_SDIST_NAME)),
            "unexpected validated GitHub Release state",
        ),
    ],
)
def test_poller_fails_fast_on_unexpected_state(
    tmp_path: Path,
    first_snapshot: object,
    error: str,
) -> None:
    """Do not retry malformed, duplicate, mismatched, or unexpected state."""

    completed, state_output, counter_path = _run_poller(
        tmp_path,
        [
            first_snapshot,
            _release_fixture((EXPECTED_WHEEL_NAME,)),
        ],
        target_state="draft",
        target_present=(EXPECTED_WHEEL_NAME,),
        transitional_state="draft",
        transitional_present=(),
        release_id=42,
    )

    assert completed.returncode != 0
    assert error in completed.stderr
    assert int(counter_path.read_text(encoding="utf-8")) == 1
    assert not state_output.exists()


@pytest.mark.fast_always
def test_poller_times_out_bounded_transition(tmp_path: Path) -> None:
    """Stop nonzero after the configured number of stale valid snapshots."""

    completed, state_output, counter_path = _run_poller(
        tmp_path,
        [[[]]],
        target_state="draft",
        target_present=(),
        transitional_state="absent",
        transitional_present=(),
        attempts=3,
    )

    assert completed.returncode != 0
    assert "timed out after 3 attempts" in completed.stderr
    assert int(counter_path.read_text(encoding="utf-8")) == 3
    assert not state_output.exists()


@pytest.mark.fast_always
@pytest.mark.parametrize(
    ("repository", "interval_seconds", "error"),
    [
        ("example/vercor/extra", "0", "repository must be OWNER/REPOSITORY"),
        ("example/vercor", "inf", "interval seconds must be finite"),
    ],
)
def test_poller_rejects_invalid_arguments_before_listing(
    tmp_path: Path,
    repository: str,
    interval_seconds: str,
    error: str,
) -> None:
    """Reject malformed repository and non-finite interval before external I/O."""

    completed, state_output, counter_path = _run_poller(
        tmp_path,
        [_release_fixture()],
        target_state="draft",
        target_present=(),
        transitional_state="absent",
        transitional_present=(),
        repository=repository,
        interval_seconds=interval_seconds,
    )

    assert completed.returncode != 0
    assert error in completed.stderr
    assert int(counter_path.read_text(encoding="utf-8")) == 0
    assert not state_output.exists()
