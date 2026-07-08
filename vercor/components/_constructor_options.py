from __future__ import annotations

from vercor.components.contracts import (
    AuthorFieldValues,
    FieldNames,
    ComponentSpec,
)


def normalize_component_spec(
    *,
    inputs: FieldNames = (),
    outputs: FieldNames = (),
    defaults: AuthorFieldValues = None,
) -> ComponentSpec:
    """Normalize public field declaration options to one ``ComponentSpec``."""

    return ComponentSpec(
        inputs=inputs,
        outputs=outputs,
        defaults=defaults or {},
    )


__all__ = ["normalize_component_spec"]
