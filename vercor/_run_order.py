from __future__ import annotations

from collections.abc import Sequence


def normalize_run_order(run_order: Sequence[str]) -> tuple[str, ...]:
    """Return ``run_order`` as an immutable component-name tuple."""

    if isinstance(run_order, str):
        raise TypeError("run_order must be a sequence of component names, not str")
    return tuple(run_order)
