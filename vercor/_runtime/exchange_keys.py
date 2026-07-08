"""Private runtime exchange key helpers."""

from __future__ import annotations

from vercor.exchanges import Exchange


def exchange_regrid_key(exchange: Exchange) -> str:
    """Return the stable regrid key for runtime mask bookkeeping."""

    return exchange._regrid_key


__all__ = ["exchange_regrid_key"]
