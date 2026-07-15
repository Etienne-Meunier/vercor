"""Repository-wide contracts for VerCOR's supervised pre-1.0 versioning."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import tomllib

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURRENT_VERSION = "0.4.0a1"
TEXT_SUFFIXES = {".json", ".md", ".py", ".toml", ".yaml", ".yml"}
API_TOKEN_EXEMPT_PATHS = {
    Path("vercor/_interpolators/bilinear_rectilinear.py"),
}
FORBIDDEN_RELEASE_LABELS = (
    ".".join(("1", "0", "0")),
    ".".join(("2", "0", "0")),
    ".".join(("3", "0", "0")),
    ".".join(("3", "1", "0")),
    ".".join(("3", "1", "1")),
    ".".join(("4", "0", "0")) + "a1",
)
FORBIDDEN_API_TOKEN = re.compile(
    r"(?<![@A-Za-z0-9])[vV][" + "1234" + r"](?![A-Za-z0-9])"
)
FORBIDDEN_VERCOR_MAJOR = re.compile(r"\bVerCOR [" + "1234" + r"](?:\b|\.)")
_RELEASE_SHORTHAND = r"(?<![\d.])(?:3\." + r"[01]|4\.0|[1234]\.x)(?![\d.])"
_RELEASE_PREFIX_CONTEXT = (
    r"(?:VerCOR|version|release|API|compatibility|fixture|plugin|manifest|"
    r"artifact|current|frozen|later)"
)
_RELEASE_SUFFIX_CONTEXT = (
    r"(?:API|compatibility|fixture|plugin|manifest|artifact|line|release|migration)"
)
FORBIDDEN_RELEASE_SHORTHAND = re.compile(
    rf"(?:\b{_RELEASE_PREFIX_CONTEXT}\b[^\n]{{0,50}}{_RELEASE_SHORTHAND}|"
    rf"{_RELEASE_SHORTHAND}[^\n]{{0,50}}\b{_RELEASE_SUFFIX_CONTEXT}\b)",
    flags=re.IGNORECASE,
)
FORBIDDEN_PATH_FRAGMENTS = (
    "migration-" + "3-to-" + "4",
    "vercor-" + "4-api",
    "test_v" + "4_",
    "test_v" + "2_",
    "public_plugin_" + "3_0",
    "vercor-3." + "1.1",
    "vercor-4." + "0.0a1",
)


def _tracked_text_paths() -> tuple[Path, ...]:
    """Return existing tracked or intended repository text paths."""

    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        check=True,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    paths = (
        Path(name)
        for name in result.stdout.split("\0")
        if name and Path(name).suffix in TEXT_SUFFIXES
    )
    return tuple(path for path in paths if (PROJECT_ROOT / path).is_file())


@pytest.mark.fast_always
def test_current_vercor_release_is_the_approved_alpha() -> None:
    project = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    assert project["version"] == CURRENT_VERSION


@pytest.mark.fast_always
def test_tracked_repository_has_no_forbidden_vercor_release_labels() -> None:
    violations: list[str] = []
    for relative_path in _tracked_text_paths():
        rendered_path = relative_path.as_posix()
        for fragment in FORBIDDEN_PATH_FRAGMENTS:
            if fragment in rendered_path:
                violations.append(
                    f"{rendered_path}: forbidden path fragment {fragment!r}"
                )

        text = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            labels = tuple(label for label in FORBIDDEN_RELEASE_LABELS if label in line)
            api_tokens = (
                ()
                if relative_path in API_TOKEN_EXEMPT_PATHS
                else tuple(FORBIDDEN_API_TOKEN.findall(line))
            )
            major_names = tuple(FORBIDDEN_VERCOR_MAJOR.findall(line))
            shorthand_labels = tuple(FORBIDDEN_RELEASE_SHORTHAND.findall(line))
            if labels or api_tokens or major_names or shorthand_labels:
                violations.append(
                    f"{rendered_path}:{line_number}: "
                    f"labels={labels}, api_tokens={api_tokens}, "
                    f"major_names={major_names}, shorthand={shorthand_labels}"
                )

    assert not violations, "ERROR forbidden VerCOR release labels:\n" + "\n".join(
        violations
    )
