"""Mypy use site for the frozen VerCOR 3.0 compatibility plugin."""

from vercor_compat_plugin_3_0 import run_smoke


def exercise_plugin() -> dict[str, object]:
    """Return typed smoke evidence from the frozen plugin."""

    evidence: dict[str, object] = run_smoke()
    return evidence
