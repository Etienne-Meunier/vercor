"""Shared output declarations for bundled setup factories."""

from vercor.output import OutputSpec


def resolve_output(output: OutputSpec | None = None) -> OutputSpec:
    """Return a validated explicit or disabled output declaration."""

    if output is None:
        return OutputSpec()
    if not isinstance(output, OutputSpec):
        raise TypeError("output must be OutputSpec or None")
    return output


__all__: list[str] = []
