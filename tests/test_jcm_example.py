"""Tests for the injectable JCM/ERA5 example entry point."""

from __future__ import annotations

import ast
from datetime import datetime
import importlib
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from vercor import Clock
from vercor.setups import JCMInputs

EXAMPLES_DIRECTORY = Path(__file__).resolve().parents[1] / "examples"


class _FakeCoupler:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.args = args
        self.kwargs = kwargs
        self.exchanges: tuple[object, ...] = ()

    def add_exchanges(self, exchanges: tuple[object, ...]) -> "_FakeCoupler":
        self.exchanges = exchanges
        return self


@pytest.mark.fast_always
def test_example_module_import_is_safe_without_jcm() -> None:
    script = """
import builtins

real_import = builtins.__import__

def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "jcm" or name.startswith("jcm."):
        raise ModuleNotFoundError("blocked jcm")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = blocked_import
import examples.run_jcm_with_era5data as example
assert callable(example.build_coupler)
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(EXAMPLES_DIRECTORY.parent)

    subprocess.run(
        [sys.executable, "-c", script],
        cwd=EXAMPLES_DIRECTORY.parent,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.fast_always
def test_build_coupler_uses_injected_ocean_inputs_and_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    example = importlib.import_module("examples.run_jcm_with_era5data")
    ocean = SimpleNamespace(name="OCN", grid=object())
    jcm_inputs = JCMInputs(coords=object(), terrain=object(), forcing=object())
    clock = Clock(
        start=datetime(2001, 2, 3),
        dt_seconds=3600.0,
        steps=4,
        calendar="noleap",
    )
    setup = SimpleNamespace(
        land=SimpleNamespace(name="LND"),
        atmosphere=SimpleNamespace(name="ATM"),
    )
    setup_calls: list[tuple[object, object]] = []

    def unexpected_ocean_loader() -> object:
        pytest.fail("an injected ocean must bypass the ERA5 loader")

    def fake_make_jcm_land_atmosphere(
        ocean_grid: object,
        *,
        inputs: JCMInputs | None,
        config: object,
    ) -> object:
        _ = config
        setup_calls.append((ocean_grid, inputs))
        return setup

    monkeypatch.setattr(example, "make_era5_ocean", unexpected_ocean_loader)
    monkeypatch.setattr(
        example,
        "make_jcm_land_atmosphere",
        fake_make_jcm_land_atmosphere,
    )
    monkeypatch.setattr(example, "Coupler", _FakeCoupler)

    coupler = example.build_coupler(
        ocean=ocean,
        jcm_inputs=jcm_inputs,
        clock=clock,
    )

    assert isinstance(coupler, _FakeCoupler)
    assert coupler.args[0] is clock
    assert coupler.kwargs["components"] == [ocean, setup.land, setup.atmosphere]
    assert setup_calls == [(ocean.grid, jcm_inputs)]
    assert len(coupler.exchanges) == 4


@pytest.mark.fast_always
def test_build_coupler_default_workflow_keeps_historic_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    example = importlib.import_module("examples.run_jcm_with_era5data")
    ocean = SimpleNamespace(name="OCN", grid=object())
    setup = SimpleNamespace(
        land=SimpleNamespace(name="LND"),
        atmosphere=SimpleNamespace(name="ATM"),
    )

    monkeypatch.setattr(example, "make_era5_ocean", lambda: ocean)
    monkeypatch.setattr(
        example, "make_jcm_land_atmosphere", lambda *args, **kwargs: setup
    )
    monkeypatch.setattr(example, "Coupler", _FakeCoupler)

    coupler = example.build_coupler()

    clock = coupler.args[0]
    assert isinstance(clock, Clock)
    assert clock.start == datetime(2000, 1, 3)
    assert clock.dt_seconds == 86400.0
    assert clock.steps == 365 * 100 - 2
    assert clock.calendar == "noleap"


class _RecordingRunCoupler:
    def __init__(self) -> None:
        self.events: list[tuple[str, object | None]] = []

    def initial_state(self) -> object:
        state = object()
        self.events.append(("initial_state", state))
        return state

    def run(self) -> object:
        state = object()
        self.events.append(("run", state))
        return state

    def write_outputs(self, state: object) -> None:
        self.events.append(("write_outputs", state))


@pytest.mark.fast_always
@pytest.mark.parametrize(
    ("arguments", "expected_steps", "expected_events"),
    (
        (
            ["--initial-state-only", "--steps", "3"],
            3,
            ("initial_state",),
        ),
        (["--steps", "2"], 2, ("run", "write_outputs")),
    ),
)
def test_cli_modes_use_requested_step_count_and_state_path(
    monkeypatch: pytest.MonkeyPatch,
    arguments: list[str],
    expected_steps: int,
    expected_events: tuple[str, ...],
) -> None:
    example = importlib.import_module("examples.run_jcm_with_era5data")
    coupler = _RecordingRunCoupler()
    clocks: list[Clock] = []

    def fake_build_coupler(*, clock: Clock | None = None, **kwargs: object) -> Any:
        _ = kwargs
        assert clock is not None
        clocks.append(clock)
        return coupler

    monkeypatch.setattr(example, "build_coupler", fake_build_coupler)

    example.main(arguments)

    assert clocks[0].steps == expected_steps
    assert tuple(event for event, _ in coupler.events) == expected_events
    if expected_events == ("run", "write_outputs"):
        assert coupler.events[1][1] is coupler.events[0][1]


@pytest.mark.fast_always
def test_examples_do_not_discard_initial_state_results() -> None:
    discarded: list[str] = []

    for path in sorted(EXAMPLES_DIRECTORY.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
                continue
            function = node.value.func
            if isinstance(function, ast.Attribute) and function.attr == "initial_state":
                discarded.append(f"{path.name}:{node.lineno}")

    assert discarded == []
