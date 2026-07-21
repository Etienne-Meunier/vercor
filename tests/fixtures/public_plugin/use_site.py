"""Mypy use site for the installed public plugin fixture."""

from pathlib import Path

from vercor.regridding import RegridderFactory
from vercor_public_plugin import PluginRegridderFactory, run_smoke


def accepts_factory(factory: RegridderFactory) -> RegridderFactory:
    """Confirm a plugin factory satisfies the public factory protocol."""

    return factory


typed_factory = accepts_factory(PluginRegridderFactory("typed-route"))


def exercise_plugin(output_dir: Path) -> dict[str, object]:
    """Return typed smoke evidence from the public plugin."""

    evidence: dict[str, object] = run_smoke(output_dir)
    return evidence
