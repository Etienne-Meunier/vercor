"""Shared output declarations for bundled setup factories."""

from vercor.output import OutputSpec, PeriodOutput


def bundled_output(output: OutputSpec | None = None) -> OutputSpec:
    """Return a validated output declaration with the bundled default."""

    if output is None:
        return OutputSpec(period=PeriodOutput(frequency="step"))
    if not isinstance(output, OutputSpec):
        raise TypeError("output must be OutputSpec or None")
    return output


def step_period_output() -> OutputSpec:
    """Return the generic step-cadence policy for bundled model output."""

    return bundled_output()


__all__: list[str] = []
