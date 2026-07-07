"""Runtime compatibility imports for public component views."""

from vercor.state import (
    ComponentView,
    RuntimeFieldSource,
    runtime_field,
    runtime_field_candidates,
)

RuntimeComponentView = ComponentView

__all__ = [
    "ComponentView",
    "RuntimeComponentView",
    "RuntimeFieldSource",
    "runtime_field",
    "runtime_field_candidates",
]
