"""Mypy use site for the installed public plugin fixture."""

from pathlib import Path

from vercor_public_plugin import run_smoke


def exercise_plugin(output_dir: Path) -> dict[str, object]:
    """Return typed smoke evidence from the public plugin."""

    evidence: dict[str, object] = run_smoke(output_dir)
    return evidence
