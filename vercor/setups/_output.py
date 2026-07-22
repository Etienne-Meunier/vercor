"""Shared output declarations for bundled setup factories."""

from vercor.output import OutputSpec, PeriodOutput


def step_period_output() -> OutputSpec:
    """Return the generic step-cadence policy for bundled model output."""

    return OutputSpec(period=PeriodOutput(frequency="step"))


__all__: list[str] = []
