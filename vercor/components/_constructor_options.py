from __future__ import annotations

from vercor.components.contracts import (
    ComponentSpec,
    _AuthorFieldValues,
    _FieldNames,
)


def normalize_component_spec(
    *,
    inputs: _FieldNames = (),
    outputs: _FieldNames = (),
    defaults: _AuthorFieldValues = None,
) -> ComponentSpec:
    """Normalize public field declaration options to one ``ComponentSpec``."""

    return ComponentSpec(
        inputs=inputs,
        outputs=outputs,
        defaults=defaults or {},
    )


__all__ = ["normalize_component_spec"]
