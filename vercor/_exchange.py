"""Private exchange helpers used by runtime mask bookkeeping."""

from vercor.exchanges import Exchange


def _exchange_regrid_key(exchange: Exchange) -> str:
    """Return the private stable regrid key for runtime mask bookkeeping."""

    return exchange._regrid_key


__all__ = ["Exchange", "_exchange_regrid_key"]
