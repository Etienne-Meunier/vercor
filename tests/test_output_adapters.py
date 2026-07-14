"""Regression guards for removal of the legacy mutable output adapter path."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def test_legacy_component_output_adapter_module_is_removed() -> None:
    assert importlib.util.find_spec("vercor.output._component_adapter") is None
    assert not Path("vercor/output/_component_adapter.py").exists()


def test_output_accumulation_has_one_immutable_owner() -> None:
    output_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(Path("vercor/output").glob("*.py"))
    )

    assert output_sources.count("class _OutputAccumulator") == 1
    assert "class PeriodAverageAccumulator" not in output_sources
    assert "_PeriodOutputAccumulator" not in output_sources
    assert "_period_output_handled_by_step" not in output_sources
    assert "_period_output_schema_factory" not in output_sources
