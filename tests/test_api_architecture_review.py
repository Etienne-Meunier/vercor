"""Release contracts for the VerCOR 3.1 API architecture documentation."""

from __future__ import annotations

import ast
from pathlib import Path
import re
import tomllib

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REVIEW_PATH = PROJECT_ROOT / "docs" / "api-architecture-review.md"
README_PATH = PROJECT_ROOT / "README.md"


@pytest.mark.fast_always
def test_architecture_review_has_exact_required_sections_in_order() -> None:
    review = REVIEW_PATH.read_text(encoding="utf-8")
    headings = re.findall(r"^## (\d+\. .+)$", review, flags=re.MULTILINE)

    assert headings == [
        "1. Executive summary",
        "2. Duplication map",
        "3. Bad design decisions",
        "4. Public API redesign",
        "5. Private API redesign",
        "6. Setup-agnostic plugin architecture",
        "7. Compatibility plan",
        "8. Final rewritten API",
    ]


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

    assert project["version"] == "3.1.0"
    assert plugin["dependencies"] == ["vercor>=3.1.0"]
