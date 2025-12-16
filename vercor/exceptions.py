class CouplerError(Exception):
    """Base class for exceptions inside Coupler."""

    pass


class ComponentError(Exception):
    """Base class for exceptions inside individual components."""

    pass


class RegridderError(Exception):
    """Base class for exceptions during regridding operations."""

    pass


class ExchangerError(Exception):
    """Base class for exceptions during data exchange between components."""

    pass
