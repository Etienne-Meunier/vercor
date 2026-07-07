"""Canonical VerCOR exchange-field vocabulary."""

from typing import Any
import warnings

from vercor.fields import VALID_FIELD_NAMES

__all__: list[str] = []
__all__.append("VALID_EXCHANGE_FIELD_NAMES")


def __getattr__(name: str) -> Any:
    """Return deprecated compatibility attributes."""

    if name == "VALID_EXCHANGE_FIELD_NAMES":
        warnings.warn(
            "vercor.field_names.VALID_EXCHANGE_FIELD_NAMES is deprecated; "
            "use vercor.fields.VALID_FIELD_NAMES instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return VALID_FIELD_NAMES
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
