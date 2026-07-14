"""Executable documentation and release contracts for VerCOR 4.0.0a1."""

from __future__ import annotations

import ast
from collections.abc import Callable
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import re
import tomllib
from typing import Any, cast, get_type_hints

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REVIEW_PATH = PROJECT_ROOT / "docs" / "api-architecture-review.md"
README_PATH = PROJECT_ROOT / "README.md"
DESIGN_PATH = PROJECT_ROOT / "DESIGN.md"
MIGRATION_PATH = PROJECT_ROOT / "docs" / "migration-3-to-4.md"
RELEASING_PATH = PROJECT_ROOT / "docs" / "releasing.md"
CHANGELOG_PATH = PROJECT_ROOT / "CHANGELOG.md"
PROGRESS_PATH = PROJECT_ROOT / "PROGRESS.md"
SIGNATURE_CONTRACT_PATH = (
    PROJECT_ROOT / "tests" / "contracts" / "vercor-4.0.0a1-public-signatures.json"
)
DEPENDENCIES_PATH = PROJECT_ROOT / "DEPENDENCIES.md"
PROGRESS_ARCHIVE_PATH = (
    PROJECT_ROOT / "docs" / "progress-archive-2026-05-16-to-2026-07-14.md"
)
PROGRESS_ARCHIVE_SHA256 = (
    "ed016d7d8c1fe8b2158baddf3a52dbb61d149d8d12818245ba3e2697c85fb9b3"
)

REQUIRED_REVIEW_HEADINGS = (
    "1. Executive summary",
    "2. Duplication map",
    "3. Bad design decisions",
    "4. Public API redesign",
    "5. Private API redesign",
    "6. Setup-agnostic plugin architecture",
    "7. Compatibility plan",
    "8. Final rewritten API",
)


def _python_fences(markdown: str) -> tuple[str, ...]:
    """Return Python snippets from Markdown in source order."""

    return tuple(re.findall(r"```python\n(.*?)```", markdown, flags=re.DOTALL))


def _assert_public_imports_only(source: str, *, owner: str) -> None:
    """Reject imports from underscored VerCOR modules in a documentation snippet."""

    tree = ast.parse(source, filename=owner)
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
                ), f"{owner} imports private VerCOR module {module}"


def _documented_public_manifest(review: str) -> dict[str, tuple[str, ...]]:
    """Parse the JSON public manifest embedded in the architecture review."""

    match = re.search(
        r"<!-- public-api-manifest:start -->\n"
        r"```json\n(.*?)\n```\n"
        r"<!-- public-api-manifest:end -->",
        review,
        flags=re.DOTALL,
    )
    assert match is not None, "architecture review lacks its public API manifest"
    manifest = json.loads(match.group(1))
    assert isinstance(manifest, dict)
    return {name: tuple(exports) for name, exports in manifest.items()}


def _public_signature_contract() -> dict[str, dict[str, str]]:
    """Load the static callable-export and behavioral-method signature contract."""

    contract = json.loads(SIGNATURE_CONTRACT_PATH.read_text(encoding="utf-8"))
    assert contract["schema_version"] == 1
    sections = {
        name: values for name, values in contract.items() if name != "schema_version"
    }
    assert set(sections) == {"exports", "methods"}
    assert all(
        isinstance(values, dict)
        and all(isinstance(value, str) for value in values.values())
        for values in sections.values()
    )
    return cast(dict[str, dict[str, str]], sections)


def _canonical_public_callable_names(
    manifest: dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    """Return every concrete callable from canonical non-root owner manifests."""

    qualified_names: list[str] = []
    for module_name, exports in manifest.items():
        if module_name == "vercor":
            continue
        module = importlib.import_module(module_name)
        qualified_names.extend(
            f"{module_name}.{name}"
            for name in exports
            if inspect.isclass(getattr(module, name))
            or inspect.isroutine(getattr(module, name))
        )
    return tuple(qualified_names)


def _canonical_public_method_names(
    manifest: dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    """Return public class/protocol behavior, excluding inherited exceptions."""

    qualified_names: list[str] = []
    for owner_name in _canonical_public_callable_names(manifest):
        owner = _resolve_qualified_name(owner_name)
        if not inspect.isclass(owner):
            continue
        if issubclass(owner, BaseException):
            continue
        qualified_names.extend(
            f"{owner_name}.{method_name}"
            for method_name, method in inspect.getmembers(owner)
            if not method_name.startswith("_") and inspect.isroutine(method)
        )
    qualified_names.append("vercor.regridding.RegridderFactory.__call__")
    return tuple(qualified_names)


def _resolve_qualified_name(qualified_name: str) -> object:
    """Resolve a documented object while keeping its canonical module explicit."""

    parts = qualified_name.split(".")
    for stop in range(len(parts), 0, -1):
        try:
            value: object = importlib.import_module(".".join(parts[:stop]))
        except ModuleNotFoundError:
            continue
        for attribute in parts[stop:]:
            value = getattr(value, attribute)
        return value
    raise AssertionError(f"cannot resolve documented signature {qualified_name}")


def _normalized_signature(value: object) -> str:
    """Return a stable, resolved signature including defaults and annotations."""

    callable_value = cast(Callable[..., object], value)
    hint_target = value.__init__ if inspect.isclass(value) else callable_value
    hints = get_type_hints(hint_target)
    try:
        signature = inspect.signature(callable_value)
    except ValueError:
        assert inspect.isclass(value) and issubclass(value, BaseException)
        init_signature = inspect.signature(value.__init__)
        signature = init_signature.replace(
            parameters=tuple(init_signature.parameters.values())[1:]
        )
    signature = signature.replace(
        parameters=[
            parameter.replace(
                annotation=hints.get(parameter.name, parameter.annotation)
            )
            for parameter in signature.parameters.values()
        ],
        return_annotation=hints.get("return", signature.return_annotation),
    )
    rendered = str(signature)
    rendered = re.sub(
        r"<function ([^ >]+) at 0x[0-9a-fA-F]+>",
        r"<function \1>",
        rendered,
    )
    rendered = re.sub(r"<object object at 0x[0-9a-fA-F]+>", "<object>", rendered)
    return (
        rendered.replace("vercor.components.contracts.", "vercor.components.")
        .replace("vercor.components.contexts.", "vercor.components.")
        .replace("vercor.components.data.", "vercor.components.")
        .replace("vercor.setups.config.", "vercor.setups.")
        .replace("vercor.setups._jcm.", "vercor.setups.")
        .replace("pathlib._local.Path", "pathlib.Path")
        .replace(" -> NoneType", " -> None")
    )


@pytest.mark.fast_always
def test_architecture_review_has_exact_v4_title_and_eight_sections() -> None:
    """Keep the approved review shape exact without asserting explanatory prose."""

    review = REVIEW_PATH.read_text(encoding="utf-8")
    assert review.startswith("# VerCOR 4.0.0a1 API architecture review\n")
    assert tuple(re.findall(r"^## (.+)$", review, flags=re.MULTILINE)) == (
        REQUIRED_REVIEW_HEADINGS
    )


@pytest.mark.fast_always
def test_documented_public_manifest_matches_live_canonical_owners() -> None:
    """Execute the review's public inventory against live ``__all__`` values."""

    manifest = _documented_public_manifest(REVIEW_PATH.read_text(encoding="utf-8"))
    assert tuple(manifest) == ("vercor",) + tuple(sorted(manifest)[1:])
    for module_name, documented_exports in manifest.items():
        module = importlib.import_module(module_name)
        assert tuple(module.__all__) == documented_exports, module_name
        assert all(hasattr(module, name) for name in documented_exports), module_name


@pytest.mark.fast_always
def test_static_public_signature_contract_is_complete_and_matches_source() -> None:
    """Freeze every canonical callable export and relevant behavioral method."""

    manifest = _documented_public_manifest(REVIEW_PATH.read_text(encoding="utf-8"))
    contract = _public_signature_contract()
    assert tuple(contract["exports"]) == _canonical_public_callable_names(manifest)
    assert tuple(contract["methods"]) == _canonical_public_method_names(manifest)
    for qualified_name, documented_signature in {
        **contract["exports"],
        **contract["methods"],
    }.items():
        value = _resolve_qualified_name(qualified_name)
        assert _normalized_signature(value) == documented_signature


@pytest.mark.fast_always
def test_documented_private_inventory_matches_all_nonpublic_modules() -> None:
    """Keep the descriptive private inventory complete without freezing behavior."""

    review = REVIEW_PATH.read_text(encoding="utf-8")
    private_section = review.split("## 5. Private API redesign", 1)[1].split(
        "### Foundations and numerical implementations", 1
    )[0]
    match = re.search(r"```text\n(.*?)\n```", private_section, flags=re.DOTALL)
    assert match is not None
    documented = set(match.group(1).splitlines())

    public_modules = {"vercor", *_documented_public_manifest(review)}
    discovered: set[str] = set()
    for path in (PROJECT_ROOT / "vercor").rglob("*.py"):
        relative = path.relative_to(PROJECT_ROOT / "vercor")
        parts = (
            relative.parent.parts
            if relative.name == "__init__.py"
            else relative.with_suffix("").parts
        )
        discovered.add("vercor" + (f".{'.'.join(parts)}" if parts else ""))

    assert documented == discovered - public_modules


@pytest.mark.fast_always
def test_readme_python_snippets_run_as_one_public_quick_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run every README Python block together, outside the repository directory."""

    snippets = _python_fences(README_PATH.read_text(encoding="utf-8"))
    assert snippets
    source = "\n\n".join(snippets)
    _assert_public_imports_only(source, owner="README.md")
    monkeypatch.chdir(tmp_path)

    namespace: dict[str, object] = {}
    exec(compile(source, str(README_PATH), "exec"), namespace)

    assert (tmp_path / "output" / "output.snapshot.nc").is_file()
    assert tuple((tmp_path / "output").glob("output.averages.*.nc"))


@pytest.mark.fast_always
def test_migration_v4_snippet_runs_without_private_or_compat_imports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execute the supported v4 migration result and verify its observable state."""

    snippets = _python_fences(MIGRATION_PATH.read_text(encoding="utf-8"))
    assert len(snippets) == 1
    source = snippets[0]
    _assert_public_imports_only(source, owner="docs/migration-3-to-4.md")
    assert "vercor.compat" not in source
    monkeypatch.chdir(tmp_path)

    namespace: dict[str, object] = {}
    exec(compile(source, str(MIGRATION_PATH), "exec"), namespace)

    migrated_temperature = cast(Any, namespace["migrated_temperature"])
    assert float(migrated_temperature[0, 0]) == pytest.approx(282.0)
    assert not tuple(tmp_path.iterdir())


@pytest.mark.fast_always
def test_release_files_and_metadata_describe_the_built_alpha() -> None:
    """Bind release documentation to installed project metadata and artifact names."""

    project = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    assert project["version"] == "4.0.0a1"
    assert "Development Status :: 3 - Alpha" in project["classifiers"]
    assert "Development Status :: 4 - Beta" not in project["classifiers"]

    changelog = CHANGELOG_PATH.read_text(encoding="utf-8")
    assert re.search(r"^## \[4\.0\.0a1\] - 2026-07-14$", changelog, re.MULTILINE)
    releasing = RELEASING_PATH.read_text(encoding="utf-8")
    commands = "\n".join(re.findall(r"```bash\n(.*?)```", releasing, re.DOTALL))
    for command in (
        "python -m build",
        "python -m pytest tests/ -q --tb=short",
        "python -m pytest tests/ -q --cov=vercor --cov-branch",
        "VERCOR_ARTIFACT_DIR",
        "tests/test_distribution_boundaries.py",
        "test_output_free_workflow_preserves_jvp_and_reverse_mode_gradients",
        "python -m vercor_public_plugin.smoke",
        "shasum -a 256",
        "git diff --check",
    ):
        assert command in commands
    assert "git push" not in commands
    assert "twine upload" not in commands

    progress = PROGRESS_PATH.read_text(encoding="utf-8")
    assert "JCM 1.1.1" in progress
    assert "Veros 1.6.2" in progress
    for artifact in (
        "vercor-4.0.0a1-py3-none-any.whl",
        "vercor-4.0.0a1.tar.gz",
        "vercor_public_plugin-0.1.0-py3-none-any.whl",
        "vercor_compat_plugin_3_0-0.1.0-py3-none-any.whl",
    ):
        assert re.search(rf"{re.escape(artifact)}.*`[0-9a-f]{{64}}`", progress)


@pytest.mark.fast_always
def test_active_memory_is_current_and_historical_detail_is_archived() -> None:
    """Keep active orientation bounded while preserving the detailed history."""

    progress = PROGRESS_PATH.read_text(encoding="utf-8")
    assert len(progress.splitlines()) <= 180
    archive_paths = tuple(
        PROJECT_ROOT / path
        for path in re.findall(r"`(docs/progress-archive-[^`]+\.md)`", progress)
    )
    assert archive_paths
    assert all(path.is_file() for path in archive_paths)
    assert PROGRESS_ARCHIVE_PATH in archive_paths
    assert (
        hashlib.sha256(PROGRESS_ARCHIVE_PATH.read_bytes()).hexdigest()
        == PROGRESS_ARCHIVE_SHA256
    )
    assert "4.0.0a1" in progress

    design = DESIGN_PATH.read_text(encoding="utf-8")
    dependencies = DEPENDENCIES_PATH.read_text(encoding="utf-8")
    assert "vercor.compat.v3" not in design
    assert "vercor.compat.v3" not in dependencies


@pytest.mark.fast_always
def test_task9_namespace_is_absent_and_frozen_fixture_is_historical_only() -> None:
    """Enforce the explicit decision to ship v4 alpha without Task 9 adapters."""

    with pytest.raises(ModuleNotFoundError, match="vercor.compat"):
        importlib.import_module("vercor.compat.v3")

    migration = MIGRATION_PATH.read_text(encoding="utf-8")
    assert "tests/fixtures/public_plugin_3_0" in migration
    assert "historical artifact" in migration.casefold()
    assert "vercor.compat.v3" not in migration
