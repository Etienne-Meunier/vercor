from __future__ import annotations

from collections.abc import Iterable
from typing import TypeAlias, TypeVar

FieldNames: TypeAlias = Iterable[str]
_NameItem = TypeVar("_NameItem")


def freeze_name_sequence(
    values: Iterable[_NameItem],
    *,
    label: str,
) -> tuple[_NameItem, ...]:
    """Freeze one public name sequence without treating text as a container."""

    if isinstance(values, (str, bytes)):
        raise TypeError(f"{label} must be a sequence, not {type(values).__name__}")
    return tuple(values)


def unique_field_names(field_names: FieldNames) -> tuple[str, ...]:
    """Validate and deduplicate field names while preserving order."""

    if isinstance(field_names, (str, bytes)) or not isinstance(field_names, Iterable):
        raise TypeError("field names must be an iterable of non-empty strings")

    unique: list[str] = []
    for field_name in field_names:
        if not isinstance(field_name, str) or not field_name.strip():
            raise TypeError("field names must be an iterable of non-empty strings")
        if field_name not in unique:
            unique.append(field_name)
    return tuple(unique)


__all__ = ["FieldNames", "freeze_name_sequence", "unique_field_names"]
