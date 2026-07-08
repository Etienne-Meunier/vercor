class AssetError(Exception):
    """Base class for exceptions related to assets."""

    pass


class CouplerError(Exception):
    """Base class for exceptions inside Coupler."""

    pass


class ComponentError(CouplerError):
    """Base class for exceptions inside individual components."""

    pass


class GridError(CouplerError):
    """Base class for exceptions related to grid operations."""

    pass


class RegridderError(CouplerError):
    """Base class for exceptions during regridding operations."""

    pass


class ExchangeError(CouplerError):
    """Base class for exceptions during data exchange between components."""

    pass


ExchangerError = ExchangeError
"""Temporary compatibility alias for the 0.7 transition."""
