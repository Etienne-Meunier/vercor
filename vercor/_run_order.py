from __future__ import annotations

from collections.abc import Sequence


def normalize_run_order(run_order: Sequence[str]) -> tuple[str, ...]:
    """Return ``run_order`` as an immutable component-name tuple."""

    if not isinstance(run_order, Sequence) or isinstance(run_order, (str, bytes)):
        raise TypeError("run_order must be a sequence of component names")
    normalized = tuple(run_order)
    if not all(isinstance(name, str) and name for name in normalized):
        raise TypeError("run_order entries must be non-empty strings")
    return normalized
