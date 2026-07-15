"""Freeze the historical public VerCOR 0.3.2 contract used for migration."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from typing import Any, cast

import pytest
import tests._distribution_support as distribution_support

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "tests/contracts/vercor-0.3.2-public-api.json"
REFERENCE_SHA = "9f0b9131c889bed5c1c2d8ded260add3cfef9524"
REFERENCE_VERSION = "0.3.2"
ARCHIVED_VERSION = ".".join(("3", "1", "1"))

OWNER_MODULES = (
    "vercor",
    "vercor.components",
    "vercor.coupling",
    "vercor.exchanges",
    "vercor.fields",
    "vercor.grids",
    "vercor.output",
    "vercor.regridding",
    "vercor.runtime",
    "vercor.setups",
    "vercor.settings",
    "vercor.state",
    "vercor.topology",
)

SIGNATURE_TARGETS = {
    "vercor.components.ComponentLike.initial_fields": (
        "vercor.components",
        "ComponentLike.initial_fields",
    ),
    "vercor.components.ComponentLike.initialize": (
        "vercor.components",
        "ComponentLike.initialize",
    ),
    "vercor.components.ComponentLike.step": (
        "vercor.components",
        "ComponentLike.step",
    ),
    "vercor.components.Component.from_step": (
        "vercor.components",
        "Component.from_step",
    ),
    "vercor.components.ComponentSpec": ("vercor.components", "ComponentSpec"),
    "vercor.components.DataComponent.from_fields": (
        "vercor.components",
        "DataComponent.from_fields",
    ),
    "vercor.components.HostComponent.from_step": (
        "vercor.components",
        "HostComponent.from_step",
    ),
    "vercor.components.LifecycleHooks": ("vercor.components", "LifecycleHooks"),
    "vercor.coupling.Coupler": ("vercor.coupling", "Coupler"),
    "vercor.coupling.Coupler.add_component": (
        "vercor.coupling",
        "Coupler.add_component",
    ),
    "vercor.coupling.Coupler.add_exchange": (
        "vercor.coupling",
        "Coupler.add_exchange",
    ),
    "vercor.coupling.Coupler.add_exchanges": (
        "vercor.coupling",
        "Coupler.add_exchanges",
    ),
    "vercor.coupling.Coupler.initial_state": (
        "vercor.coupling",
        "Coupler.initial_state",
    ),
    "vercor.coupling.Coupler.run": ("vercor.coupling", "Coupler.run"),
    "vercor.coupling.Coupler.set_run_order": (
        "vercor.coupling",
        "Coupler.set_run_order",
    ),
    "vercor.coupling.Coupler.write_outputs": (
        "vercor.coupling",
        "Coupler.write_outputs",
    ),
    "vercor.coupling.CouplerSpec": ("vercor.coupling", "CouplerSpec"),
    "vercor.coupling.CouplerSpec.build": ("vercor.coupling", "CouplerSpec.build"),
    "vercor.exchanges.Exchange": ("vercor.exchanges", "Exchange"),
    "vercor.fields.vector": ("vercor.fields", "vector"),
    "vercor.grids.RectilinearGrid": ("vercor.grids", "RectilinearGrid"),
    "vercor.grids.RectilinearGrid.from_coordinates": (
        "vercor.grids",
        "RectilinearGrid.from_coordinates",
    ),
    "vercor.grids.RectilinearGrid.uniform": (
        "vercor.grids",
        "RectilinearGrid.uniform",
    ),
    "vercor.output.OutputConfig": ("vercor.output", "OutputConfig"),
    "vercor.output.PeriodOutput": ("vercor.output", "PeriodOutput"),
    "vercor.regridding.bilinear": ("vercor.regridding", "bilinear"),
    "vercor.regridding.conservative": ("vercor.regridding", "conservative"),
    "vercor.runtime.ExecutionBackend.run": (
        "vercor.runtime",
        "ExecutionBackend.run",
    ),
    "vercor.runtime.RuntimeDriver.step_component": (
        "vercor.runtime",
        "RuntimeDriver.step_component",
    ),
    "vercor.runtime.RuntimeOptions": ("vercor.runtime", "RuntimeOptions"),
    "vercor.settings.Settings": ("vercor.settings", "Settings"),
    "vercor.state.RunState.component": ("vercor.state", "RunState.component"),
    "vercor.state.RunState.components": ("vercor.state", "RunState.components"),
    "vercor.state.RunState.replace_fields": (
        "vercor.state",
        "RunState.replace_fields",
    ),
    "vercor.topology.ExchangeTopologyPatch": (
        "vercor.topology",
        "ExchangeTopologyPatch",
    ),
    "vercor.topology.TopologyPolicy.applies": (
        "vercor.topology",
        "TopologyPolicy.applies",
    ),
    "vercor.topology.TopologyPolicy.build": (
        "vercor.topology",
        "TopologyPolicy.build",
    ),
}

REFERENCE_INSPECTION_SCRIPT = r"""
import importlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import re
import sys

wheel = Path(sys.argv[1]).resolve()
owner_modules = json.loads(sys.argv[2])
signature_targets = json.loads(sys.argv[3])
sys.path.insert(0, str(wheel))

def resolve(module_name, attribute_path):
    value = importlib.import_module(module_name)
    for attribute in attribute_path.split("."):
        value = getattr(value, attribute)
    return value

def public_signature(value):
    return re.sub(r" at 0x[0-9a-fA-F]+", "", str(inspect.signature(value)))

package = importlib.import_module("vercor")
origin = Path(package.__file__).resolve()
if str(wheel) not in str(origin):
    raise RuntimeError(f"VerCOR was not imported from {wheel}: {origin}")

print(json.dumps({
    "version": importlib.metadata.version("vercor"),
    "exports": {
        module_name: list(importlib.import_module(module_name).__all__)
        for module_name in owner_modules
    },
    "signatures": {
        public_name: public_signature(resolve(*target))
        for public_name, target in signature_targets.items()
    },
}))
"""


def _load_manifest() -> dict[str, Any]:
    assert MANIFEST_PATH.is_file(), f"missing compatibility manifest: {MANIFEST_PATH}"
    return cast(
        dict[str, Any],
        json.loads(MANIFEST_PATH.read_text(encoding="utf-8")),
    )


def _build_reference_wheel(work_root: Path) -> Path:
    archive_path = work_root / "vercor-0.3.2-reference.tar"
    source_root = work_root / "source"
    dist_root = work_root / "dist"
    source_root.mkdir()
    dist_root.mkdir()
    subprocess.run(
        [
            "git",
            "archive",
            "--format=tar",
            f"--output={archive_path}",
            REFERENCE_SHA,
        ],
        check=True,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    with tarfile.open(archive_path, mode="r") as archive:
        archive.extractall(source_root, filter="data")

    archived_pyproject = source_root / "pyproject.toml"
    archived_metadata = archived_pyproject.read_text(encoding="utf-8")
    incorrect_declaration = f'version = "{ARCHIVED_VERSION}"'
    corrected_declaration = f'version = "{REFERENCE_VERSION}"'
    assert archived_metadata.count(incorrect_declaration) == 1
    archived_pyproject.write_text(
        archived_metadata.replace(incorrect_declaration, corrected_declaration),
        encoding="utf-8",
    )

    environment = os.environ.copy()
    build_pythonpath = distribution_support._cached_build_pythonpath()
    if build_pythonpath:
        environment["PYTHONPATH"] = build_pythonpath
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--no-isolation",
            "--wheel",
            "--outdir",
            str(dist_root),
            str(source_root),
        ],
        check=True,
        cwd=source_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    wheel = dist_root / f"vercor-{REFERENCE_VERSION}-py3-none-any.whl"
    if not wheel.is_file():
        raise RuntimeError(f"reference build did not create {wheel}")
    return wheel


def _inspect_reference_wheel(wheel: Path) -> dict[str, Any]:
    expected_name = f"vercor-{REFERENCE_VERSION}-py3-none-any.whl"
    if wheel.name != expected_name or not wheel.is_file():
        raise ValueError(
            f"expected an existing {REFERENCE_VERSION} reference wheel named "
            f"{expected_name!r}, got {wheel}"
        )
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            REFERENCE_INSPECTION_SCRIPT,
            str(wheel.resolve()),
            json.dumps(OWNER_MODULES),
            json.dumps(SIGNATURE_TARGETS),
        ],
        check=True,
        cwd=wheel.parent,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return cast(dict[str, Any], json.loads(result.stdout))


@pytest.mark.fast_always
def test_manifest_declares_complete_historical_vercor_0_3_2_contract() -> None:
    manifest = _load_manifest()

    assert manifest["schema_version"] == 1
    assert manifest["reference_sha"] == REFERENCE_SHA
    assert manifest["version"] == REFERENCE_VERSION
    assert tuple(manifest["exports"]) == OWNER_MODULES
    assert set(manifest["signatures"]) == set(SIGNATURE_TARGETS)


@pytest.mark.fast_always
def test_clean_pinned_reference_wheel_matches_frozen_0_3_2_contract(
    tmp_path: Path,
) -> None:
    manifest = _load_manifest()

    reference_wheel = _build_reference_wheel(tmp_path)
    artifact_contract = _inspect_reference_wheel(reference_wheel)

    assert artifact_contract == {
        "version": manifest["version"],
        "exports": manifest["exports"],
        "signatures": manifest["signatures"],
    }


@pytest.mark.fast_always
def test_reference_wheel_inspection_rejects_unrelated_artifacts(
    tmp_path: Path,
) -> None:
    unrelated_wheel = tmp_path / "vercor-0.2.1-py3-none-any.whl"
    unrelated_wheel.touch()

    with pytest.raises(ValueError, match="0.3.2 reference wheel"):
        _inspect_reference_wheel(unrelated_wheel)
