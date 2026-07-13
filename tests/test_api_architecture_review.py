"""Release contracts for the VerCOR 3.1 API architecture documentation."""

from __future__ import annotations

import ast
from pathlib import Path
import re
import tomllib

import pytest
import vercor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REVIEW_PATH = PROJECT_ROOT / "docs" / "api-architecture-review.md"
README_PATH = PROJECT_ROOT / "README.md"
PROGRESS_PATH = PROJECT_ROOT / "PROGRESS.md"
DEPENDENCIES_PATH = PROJECT_ROOT / "DEPENDENCIES.md"

REQUIRED_REVIEW_HEADINGS = [
    "1. Executive summary",
    "2. Duplication map",
    "3. Bad design decisions",
    "4. Public API redesign",
    "5. Private API redesign",
    "6. Setup-agnostic plugin architecture",
    "7. Compatibility plan",
    "8. Final rewritten API",
]


def _normalized(markdown: str) -> str:
    return " ".join(markdown.split())


def _review_sections(review: str) -> dict[str, str]:
    return {
        heading: body
        for heading, body in re.findall(
            r"^## (\d+\. [^\n]+)\n(.*?)(?=^## |\Z)",
            review,
            flags=re.MULTILINE | re.DOTALL,
        )
    }


def _documented_root_exports(review: str) -> set[str]:
    inventory = re.search(
        r"^- `vercor` keeps its exact 48-symbol.*?:\n(.*?)(?=^- `vercor\.components`:)",
        review,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert inventory is not None
    return set(re.findall(r"`([A-Za-z][A-Za-z0-9_]*)`", inventory.group(1)))


@pytest.mark.fast_always
def test_architecture_review_has_exact_required_sections_in_order() -> None:
    review = REVIEW_PATH.read_text(encoding="utf-8")
    headings = re.findall(r"^## (.+)$", review, flags=re.MULTILINE)

    assert headings == REQUIRED_REVIEW_HEADINGS


@pytest.mark.fast_always
def test_architecture_review_guards_public_and_private_contracts() -> None:
    review = REVIEW_PATH.read_text(encoding="utf-8")

    required_contracts = (
        "ComponentLike",
        "ComponentStepReturn",
        "Component.from_step",
        "HostComponent.from_step",
        "DataComponent.from_fields",
        "DataComponent.from_step",
        "RuntimeDriver.step_component",
        "PreparedCoupling",
        "vercor.components._adapter",
        "vercor._runtime.backends",
        "vercor._runtime.runner",
        "vercor.output._session",
        "original user object",
        "custom backend",
        "period output",
        "2.x -> 3.x",
        "JCM/JAXGCM",
        "4.0",
        "external plugin compatibility matrix",
    )
    for contract in required_contracts:
        assert contract in review, contract

    for disposition in (
        "merged",
        "renamed",
        "removed",
        "kept separate",
        "moved public",
        "moved private",
        "deferred",
    ):
        assert disposition in review, disposition

    assert "**must change**" in review
    assert "**nice to improve**" in review
    assert "48-symbol" in review
    assert "49-symbol" not in review
    assert "same 49 symbols" not in review
    assert "Setup configs imported from root/config compatibility modules" not in review
    assert "Module/helper grid constructors" not in review
    assert "`Coupler.finalize(...)`" not in review
    assert "CAMulator writes native period files beneath its configured" in review
    assert "### Complete proposed public API" in review
    assert "### Complete proposed private API" in review

    for module in (
        "components",
        "runtime",
        "topology",
        "coupling",
        "exchanges",
        "regridding",
        "grids",
        "fields",
        "state",
        "output",
        "setups",
        "types",
        "dtypes",
        "jax_logging",
    ):
        assert f"`vercor.{module}`" in review, module


@pytest.mark.fast_always
def test_architecture_review_documents_vercor_3_1_1_hardening() -> None:
    review = REVIEW_PATH.read_text(encoding="utf-8")

    assert _documented_root_exports(review) == set(vercor.__all__)
    required_by_section = {
        "1. Executive summary": (
            "VerCOR 3.1.1",
            "`validate_component_contract()`",
            "`ComponentSpec.lifecycle`",
            "structural configuration snapshot",
            "does not re-execute `initial_fields()`",
            "does not materialize or hash seed array contents",
            "hidden closure, global, and default state is outside the supported configuration contract",
            "fan-in",
            "shape-preserving",
            "topology-mask",
            "frozen 3.0",
            "branch-coverage",
        ),
        "2. Duplication map": (
            "Component contract validation",
            "structural configuration snapshot",
            "prepared reuse",
            "Ambiguous exchange fan-in",
            "Custom-backend result schema",
            "Current-plugin proof versus frozen-3.0 proof",
        ),
        "3. Bad design decisions": (
            "duplicate producers",
            "replacement shapes",
            "nonnumeric or out-of-range topology masks",
        ),
        "4. Public API redesign": (
            "mutable setup-time metadata",
            "values may be JAX-traced",
            "complete one-off setup",
            "reusable recipes",
            "incremental assembly",
            "receive and send the same field",
        ),
        "5. Private API redesign": (
            "`validate_component_contract()`",
            "`PreparedCoupling.validate_configuration()`",
            "component, `Settings`, clock, runtime, and topology structure",
            "generic lifecycle and spec callables are identity-only",
            "hidden mutable closure, global, and default state is outside the supported configuration contract",
            (
                "mutable setup configuration belongs in `Settings`, component "
                "attributes, or an explicit author callable object"
            ),
            "explicit author callable objects, bound owners, and partials",
            "array identity, field name, shape, and dtype",
            "`validate_exchange_fan_in()`",
            "custom-backend result schema",
        ),
        "6. Setup-agnostic plugin architecture": (
            "`tests/fixtures/public_plugin`",
            "`tests/fixtures/public_plugin_3_0`",
            "`vercor>=3.1,<4`",
            "`vercor>=3.0,<4`",
            "90% branch-coverage gate",
        ),
        "7. Compatibility plan": (
            "corrections of invalid or ambiguous behavior",
            "VerCOR 3.1.1",
            "current and frozen-3.0 installed plugins",
        ),
        "8. Final rewritten API": (
            "48-symbol",
            "route IDs",
            "fan-in reducers",
            "unified output",
            "public payload access",
            "`ComponentBinding`",
            "`PreparedGraph`",
            "entry-point discovery",
            "Pydantic",
        ),
    }

    sections = _review_sections(review)
    for heading, concepts in required_by_section.items():
        section = _normalized(sections[heading]).casefold()
        for concept in concepts:
            assert _normalized(concept).casefold() in section, f"{heading}: {concept}"
    assert "fingerprint" not in review.casefold()


@pytest.mark.fast_always
def test_architecture_review_bounds_installed_plugin_proof() -> None:
    review = REVIEW_PATH.read_text(encoding="utf-8")
    sections = _review_sections(review)
    required_scenarios = (
        "DataComponent",
        "bilinear Exchange",
        "StepResult payload replacement",
        "RunState.replace_fields",
        "structural JAX/host components",
        "original-object lifecycle hooks",
        "custom sequential backend",
        "empty valid topology-policy patch",
        "snapshot",
    )

    for heading in (
        "1. Executive summary",
        "2. Duplication map",
        "6. Setup-agnostic plugin architecture",
    ):
        section = _normalized(sections[heading]).replace("`", "").casefold()
        for scenario in required_scenarios:
            assert scenario.casefold() in section, f"{heading}: {scenario}"

    assert "full 3.1 extension surface" not in review
    assert "all 3.1 extension points" not in review


@pytest.mark.fast_always
def test_docs_bound_custom_backend_schema_claim_to_implemented_checks() -> None:
    review = REVIEW_PATH.read_text(encoding="utf-8")
    sections = _review_sections(review)
    dependencies = DEPENDENCIES_PATH.read_text(encoding="utf-8")
    required_claims = (
        "exact component, store-field, and fractional-mask names",
        "per-field and mask shapes",
        "component grid shapes",
        "not grid coordinates or identity, mask values, or dtypes",
    )

    for owner, text in (
        ("2. Duplication map", sections["2. Duplication map"]),
        ("5. Private API redesign", sections["5. Private API redesign"]),
        ("DEPENDENCIES.md", dependencies),
    ):
        normalized = _normalized(text).casefold()
        for claim in required_claims:
            assert claim in normalized, f"{owner}: {claim}"

    for overclaim in (
        "prepared components, stores, grids, masks, and shapes",
        "exact input stores/grids/masks",
    ):
        assert overclaim not in review + dependencies


@pytest.mark.fast_always
def test_readme_distinguishes_configuration_and_assembly_ownership() -> None:
    readme = README_PATH.read_text(encoding="utf-8")

    for statement in (
        "`Settings` is mutable setup-time metadata",
        "values may be JAX-traced",
        "`RuntimeOptions` owns static policy",
        "complete one-off setups",
        "`CouplerSpec` for reusable recipes",
        "mutators for incremental assembly",
    ):
        assert statement in readme, statement


@pytest.mark.fast_always
def test_repository_memory_records_vercor_3_1_1_ownership() -> None:
    progress = PROGRESS_PATH.read_text(encoding="utf-8")
    dependencies = DEPENDENCIES_PATH.read_text(encoding="utf-8")

    stale_claim = (
        "component `.data`/`.setup_metadata` remain mutable setup attributes "
        "for compatibility"
    )
    assert stale_claim not in _normalized(progress)
    assert "VerCOR 3.1.1 API hardening documented on 2026-07-13" in progress
    for evidence in (
        "focused 148/148",
        "fast suite passes 461/461 with 397 deselected",
        "full suite passes 858/858",
    ):
        assert evidence in _normalized(progress)

    for ownership in (
        "`ComponentSpec.lifecycle`",
        "`validate_component_contract()`",
        "fan-in rejection",
        "legal receive/send overlap",
        "shape-preserving replacement",
        "custom-backend result-schema validation",
        "concrete finite numeric/bool topology masks",
        "tests/fixtures/public_plugin_3_0/",
        "90% branch-coverage gate",
        "structural configuration snapshot",
        "does not re-execute `initial_fields()`",
        "array identity, field name, shape, and dtype",
        "lifecycle/spec callables are identity-only",
        "hidden closure/global/default mutable state is outside the supported configuration contract",
    ):
        assert ownership in dependencies, ownership
    assert "fingerprint" not in dependencies.casefold()
    overclaim = "defaults, keyword defaults, closure cells, partial arguments, and bound-owner state"
    assert overclaim not in REVIEW_PATH.read_text(encoding="utf-8") + dependencies


def _python_fences(markdown: str) -> tuple[str, ...]:
    return tuple(re.findall(r"```python\n(.*?)```", markdown, flags=re.DOTALL))


@pytest.mark.fast_always
def test_readme_documents_every_public_customization_path() -> None:
    readme = README_PATH.read_text(encoding="utf-8")
    for heading in (
        "Default built-in setup",
        "Structural custom JAX component",
        "Host component",
        "Custom execution backend",
        "Custom topology policy",
        "Lifecycle hooks and output",
    ):
        assert f"### {heading}" in readme, heading

    assert "docs/api-architecture-review.md" in readme
    assert "tests/fixtures/public_plugin" in readme

    snippets = _python_fences(readme)
    assert snippets
    for index, snippet in enumerate(snippets):
        tree = ast.parse(snippet, filename=f"README snippet {index}")
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                modules.append(node.module)
            for module in modules:
                if module == "vercor" or module.startswith("vercor."):
                    assert not any(
                        part.startswith("_") for part in module.split(".")[1:]
                    ), f"README snippet imports private module {module}"


@pytest.mark.fast_always
def test_readme_python_snippets_run_as_one_quick_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readme = README_PATH.read_text(encoding="utf-8")
    snippets = _python_fences(readme)
    source = "\n\n".join(snippets)
    monkeypatch.chdir(tmp_path)

    exec(compile(source, str(README_PATH), "exec"), {})

    assert (tmp_path / "output" / "output.snapshot.nc").is_file()
    assert tuple(tmp_path.glob("OUTPUT.averages.*.nc"))


@pytest.mark.fast_always
def test_release_metadata_and_plugin_requirement_are_vercor_3_1() -> None:
    project = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    plugin = tomllib.loads(
        (PROJECT_ROOT / "tests/fixtures/public_plugin/pyproject.toml").read_text(
            encoding="utf-8"
        )
    )["project"]
    frozen_plugin = tomllib.loads(
        (PROJECT_ROOT / "tests/fixtures/public_plugin_3_0/pyproject.toml").read_text(
            encoding="utf-8"
        )
    )["project"]

    assert project["version"] == "3.1.1"
    assert plugin["dependencies"] == ["vercor>=3.1,<4"]
    assert frozen_plugin["dependencies"] == ["vercor>=3.0,<4"]
