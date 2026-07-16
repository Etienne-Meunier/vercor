"""Contracts for serial and distributed test cache isolation."""

from __future__ import annotations

from pathlib import Path

from tests._parallel_support import configure_test_cache_environment


def test_serial_process_defaults_all_test_cache_values(tmp_path: Path) -> None:
    environ: dict[str, str] = {}

    defaulted = configure_test_cache_environment(
        environ,
        cache_root=tmp_path,
        worker_id=None,
    )

    assert defaulted == {
        "MPLBACKEND": True,
        "MPLCONFIGDIR": True,
        "XDG_CACHE_HOME": True,
    }
    assert environ == {
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(tmp_path / "vercor-matplotlib-cache"),
        "VERCOR_TEST_DEFAULTED_ENV": "MPLBACKEND,MPLCONFIGDIR,XDG_CACHE_HOME",
        "XDG_CACHE_HOME": str(tmp_path / "vercor-xdg-cache"),
    }


def test_worker_uses_distinct_defaults_without_a_controller(tmp_path: Path) -> None:
    environ: dict[str, str] = {}

    defaulted = configure_test_cache_environment(
        environ,
        cache_root=tmp_path,
        worker_id="gw2",
    )

    assert all(defaulted.values())
    assert environ["MPLBACKEND"] == "Agg"
    assert environ["MPLCONFIGDIR"] == str(tmp_path / "vercor-matplotlib-cache-gw2")
    assert environ["XDG_CACHE_HOME"] == str(tmp_path / "vercor-xdg-cache-gw2")


def test_worker_replaces_only_inherited_controller_defaults(tmp_path: Path) -> None:
    environ = {
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(tmp_path / "vercor-matplotlib-cache"),
        "VERCOR_TEST_DEFAULTED_ENV": "MPLBACKEND,MPLCONFIGDIR,XDG_CACHE_HOME",
        "XDG_CACHE_HOME": str(tmp_path / "vercor-xdg-cache"),
    }

    defaulted = configure_test_cache_environment(
        environ,
        cache_root=tmp_path,
        worker_id="gw1",
    )

    assert all(defaulted.values())
    assert environ["MPLCONFIGDIR"] == str(tmp_path / "vercor-matplotlib-cache-gw1")
    assert environ["XDG_CACHE_HOME"] == str(tmp_path / "vercor-xdg-cache-gw1")


def test_worker_configuration_is_idempotent(tmp_path: Path) -> None:
    environ: dict[str, str] = {}
    expected = configure_test_cache_environment(
        environ,
        cache_root=tmp_path,
        worker_id="gw0",
    )
    first_environment = environ.copy()

    actual = configure_test_cache_environment(
        environ,
        cache_root=tmp_path,
        worker_id="gw0",
    )

    assert actual == expected
    assert environ == first_environment


def test_explicit_user_values_are_preserved_in_workers(tmp_path: Path) -> None:
    environ = {
        "MPLBACKEND": "svg",
        "MPLCONFIGDIR": str(tmp_path / "user-matplotlib"),
        "XDG_CACHE_HOME": str(tmp_path / "user-xdg"),
    }
    original = environ.copy()

    defaulted = configure_test_cache_environment(
        environ,
        cache_root=tmp_path,
        worker_id="gw3",
    )

    assert defaulted == {
        "MPLBACKEND": False,
        "MPLCONFIGDIR": False,
        "XDG_CACHE_HOME": False,
    }
    assert environ == {
        **original,
        "VERCOR_TEST_DEFAULTED_ENV": "",
    }
    assert environ["MPLBACKEND"] == original["MPLBACKEND"]
    assert environ["MPLCONFIGDIR"] == original["MPLCONFIGDIR"]
    assert environ["XDG_CACHE_HOME"] == original["XDG_CACHE_HOME"]
