"""Mypy use site for the installed public plugin fixture."""

from pathlib import Path

from vercor.exchanges import Exchange
from vercor.regridding import RegridderFactory, bilinear, conservative
from vercor_public_plugin import PluginRegridderFactory, run_smoke


def accepts_factory(factory: RegridderFactory) -> RegridderFactory:
    """Confirm a plugin factory satisfies the public factory protocol."""

    return factory


typed_factory = accepts_factory(PluginRegridderFactory("typed-route"))
typed_bilinear: RegridderFactory = bilinear
typed_conservative: RegridderFactory = conservative
typed_exchange_factory: RegridderFactory = Exchange(
    "SOURCE", "TARGET", ("field",)
).regridder_factory


def exercise_plugin(output_dir: Path) -> dict[str, object]:
    """Return typed smoke evidence from the public plugin."""

    evidence: dict[str, object] = run_smoke(output_dir)
    return evidence
