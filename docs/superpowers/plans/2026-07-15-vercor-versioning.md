# VerCOR Pre-1.0 Versioning Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct every tracked VerCOR release label and artifact boundary to the approved pre-1.0 sequence whose current candidate is `0.4.0a1`.

**Architecture:** Add one executable repository policy before changing release files, then perform the path/content migration atomically so imports, links, and artifact contracts never settle in a half-renamed state. Historical source snapshots retain their Git SHA but have their package metadata normalized in the temporary extracted tree before evidence wheels are built. Fresh artifacts and the full release gates provide the final evidence.

**Tech Stack:** Python 3.13, pytest, TOML, YAML/GitHub Actions, Flit/build, Black, flake8, mypy, JAX, Markdown.

## Global Constraints

- Apply exactly: `1.0.0` → `0.2.0`, `2.0.0` → `0.2.1`, `3.0.0` → `0.3.0`, `3.1.0` → `0.3.1`, `3.1.1` → `0.3.2`, and `4.0.0a1` → `0.4.0a1`.
- VerCOR 4/v4 becomes VerCOR 0.4/v0.4; VerCOR 3/v3 release evidence becomes VerCOR 0.3/v0.3.
- Neutral internal fixture names that merely used v1/v2/v3/v4 as generation markers become descriptive, version-free names when a release mapping would misstate their history.
- Do not change Python, JAX, NumPy, SciPy, JCM, Veros, GitHub Action, Apache License, schema, or independently versioned plugin release numbers.
- Do not relabel old candidate hashes; record hashes only for freshly built `0.4.0a1` artifacts.
- Do not rewrite Git history, create or move tags, push, publish, upload artifacts, or create a GitHub release.
- Use the direct `scipy` environment executable if the Conda launcher panics.

---

## File structure and ownership

- `tests/test_versioning_policy.py`: sole exhaustive tracked-file guard for VerCOR release labels and version-bearing paths.
- `pyproject.toml`, `.github/workflows/python-package.yml`, `tests/_distribution_support.py`, and `tests/test_distribution_boundaries.py`: current package and built-artifact identity.
- `tests/test_api_architecture_review.py` and `tests/contracts/vercor-0.4.0a1-public-signatures.json`: executable current-alpha documentation and signature contract.
- `tests/test_v0_4_compatibility_baseline.py` and `tests/contracts/vercor-0.3.2-public-api.json`: historical public API evidence normalized from the pinned source SHA.
- `tests/fixtures/public_plugin_0_3/`: frozen public plugin evidence for the corrected 0.3 line.
- `README.md`, `CHANGELOG.md`, `DESIGN.md`, `DEPENDENCIES.md`, `PROGRESS.md`, and `docs/`: user-facing, release, design, migration, and historical records.
- `vercor/__init__.py`: the only production source file whose package-generation docstring changes.
- Renamed `tests/test_v0_4_*.py` modules: current-alpha behavioral contracts; test behavior is unchanged.

### Task 1: Add failing version-policy and current-metadata contracts

**Files:**
- Create: `tests/test_versioning_policy.py`
- Modify: `tests/test_api_architecture_review.py`

**Interfaces:**
- Consumes: tracked paths returned by `git ls-files`, `pyproject.toml`, and the approved current version `0.4.0a1`.
- Produces: `test_current_vercor_release_is_the_approved_alpha()` and `test_tracked_repository_has_no_forbidden_vercor_release_labels()`; Task 2 makes both pass.

- [ ] **Step 1: Create the exhaustive tracked-file test**

```python
"""Repository-wide contracts for VerCOR's supervised pre-1.0 versioning."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import tomllib

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURRENT_VERSION = "0.4.0a1"
TEXT_SUFFIXES = {".json", ".md", ".py", ".toml", ".yaml", ".yml"}
API_TOKEN_EXEMPT_PATHS = {
    Path("vercor/_interpolators/bilinear_rectilinear.py"),
}
FORBIDDEN_RELEASE_LABELS = (
    ".".join(("1", "0", "0")),
    ".".join(("2", "0", "0")),
    ".".join(("3", "0", "0")),
    ".".join(("3", "1", "0")),
    ".".join(("3", "1", "1")),
    ".".join(("4", "0", "0")) + "a1",
)
FORBIDDEN_API_TOKEN = re.compile(
    r"(?<![@A-Za-z0-9])[vV][" + "1234" + r"](?![A-Za-z0-9])"
)
FORBIDDEN_VERCOR_MAJOR = re.compile(
    r"\bVerCOR [" + "1234" + r"](?:\b|\.)"
)
FORBIDDEN_PATH_FRAGMENTS = (
    "migration-" + "3-to-" + "4",
    "vercor-" + "4-api",
    "test_" + "v4_",
    "test_" + "v2_",
    "public_plugin_" + "3_0",
    "vercor-" + "3.1.1",
    "vercor-" + "4.0.0a1",
)


def _tracked_text_paths() -> tuple[Path, ...]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        check=True,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    return tuple(
        Path(name)
        for name in result.stdout.split("\0")
        if name and Path(name).suffix in TEXT_SUFFIXES
    )


@pytest.mark.fast_always
def test_current_vercor_release_is_the_approved_alpha() -> None:
    project = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    assert project["version"] == CURRENT_VERSION


@pytest.mark.fast_always
def test_tracked_repository_has_no_forbidden_vercor_release_labels() -> None:
    violations: list[str] = []
    for relative_path in _tracked_text_paths():
        rendered_path = relative_path.as_posix()
        for fragment in FORBIDDEN_PATH_FRAGMENTS:
            if fragment in rendered_path:
                violations.append(f"{rendered_path}: forbidden path fragment {fragment!r}")

        text = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            labels = tuple(
                label for label in FORBIDDEN_RELEASE_LABELS if label in line
            )
            api_tokens = (
                ()
                if relative_path in API_TOKEN_EXEMPT_PATHS
                else tuple(FORBIDDEN_API_TOKEN.findall(line))
            )
            major_names = tuple(FORBIDDEN_VERCOR_MAJOR.findall(line))
            if labels or api_tokens or major_names:
                violations.append(
                    f"{rendered_path}:{line_number}: "
                    f"labels={labels}, api_tokens={api_tokens}, "
                    f"major_names={major_names}"
                )

    assert not violations, "ERROR forbidden VerCOR release labels:\n" + "\n".join(
        violations
    )
```

- [ ] **Step 2: Change the existing executable release expectations before metadata**

In `tests/test_api_architecture_review.py`, set these exact paths and values while leaving behavior assertions intact:

```python
MIGRATION_PATH = PROJECT_ROOT / "docs" / "migration-0.3-to-0.4.md"
SIGNATURE_CONTRACT_PATH = (
    PROJECT_ROOT / "tests" / "contracts" / "vercor-0.4.0a1-public-signatures.json"
)


def test_architecture_review_has_exact_v0_4_title_and_eight_sections() -> None:
    review = REVIEW_PATH.read_text(encoding="utf-8")
    assert review.startswith("# VerCOR 0.4.0a1 API architecture review\n")
    assert tuple(re.findall(r"^## (.+)$", review, flags=re.MULTILINE)) == (
        REQUIRED_REVIEW_HEADINGS
    )
```

Also change the metadata assertion to `project["version"] == "0.4.0a1"`, the progress assertion to `"0.4.0a1" in progress`, migration fixture references to `tests/fixtures/public_plugin_0_3`, and current/frozen artifact names to the corrected names specified in Task 2.

Replace the old hash requirement in
`test_release_files_and_metadata_describe_the_built_alpha()` with an artifact
command check that is valid both before and after Task 3 records fresh hashes:

```python
for artifact in (
    "vercor-0.4.0a1-py3-none-any.whl",
    "vercor-0.4.0a1.tar.gz",
    "vercor_public_plugin-0.1.0-py3-none-any.whl",
    "vercor_compat_plugin_0_3-0.1.0-py3-none-any.whl",
):
    assert artifact in releasing
```

- [ ] **Step 3: Run the focused tests and verify RED for the intended reasons**

Run:

```bash
conda run -n scipy pytest tests/test_versioning_policy.py tests/test_api_architecture_review.py -q --tb=short
```

Expected: failures report current metadata `4.0.0a1`, missing renamed paths, and forbidden tracked labels. Existing behavior tests unrelated to version identity remain passing.

### Task 2: Perform the atomic repository and fixture migration

**Files:**
- Rename: `docs/migration-3-to-4.md` → `docs/migration-0.3-to-0.4.md`
- Rename: `docs/superpowers/specs/2026-07-13-vercor-4-api-design.md` → `docs/superpowers/specs/2026-07-13-vercor-0.4-api-design.md`
- Rename: `docs/superpowers/plans/2026-07-13-vercor-4-api.md` → `docs/superpowers/plans/2026-07-13-vercor-0.4-api.md`
- Rename: `tests/contracts/vercor-3.1.1-public-api.json` → `tests/contracts/vercor-0.3.2-public-api.json`
- Rename: `tests/contracts/vercor-4.0.0a1-public-signatures.json` → `tests/contracts/vercor-0.4.0a1-public-signatures.json`
- Rename: `tests/fixtures/public_plugin_3_0/` → `tests/fixtures/public_plugin_0_3/`
- Rename: `tests/fixtures/public_plugin_0_3/src/vercor_compat_plugin_3_0/` → `tests/fixtures/public_plugin_0_3/src/vercor_compat_plugin_0_3/`
- Rename: `tests/test_v2_api_boundary_redesign.py` → `tests/test_v0_2_1_api_boundary_redesign.py`
- Rename: `tests/test_v4_compatibility_baseline.py` → `tests/test_v0_4_compatibility_baseline.py`
- Rename: `tests/test_v4_component_contracts.py` → `tests/test_v0_4_component_contracts.py`
- Rename: `tests/test_v4_output_providers.py` → `tests/test_v0_4_output_providers.py`
- Rename: `tests/test_v4_physics.py` → `tests/test_v0_4_physics.py`
- Rename: `tests/test_v4_public_api.py` → `tests/test_v0_4_public_api.py`
- Rename: `tests/test_v4_routes_and_state.py` → `tests/test_v0_4_routes_and_state.py`
- Rename: `tests/test_v4_workflow_execution.py` → `tests/test_v0_4_workflow_execution.py`
- Rename: `tests/test_v4_workflows.py` → `tests/test_v0_4_workflows.py`
- Modify: `.github/workflows/python-package.yml`, `pyproject.toml`, `vercor/__init__.py`
- Modify: `tests/_component_test_support.py`, `tests/_distribution_support.py`, `tests/_output_test_support.py`, `tests/_workflow_test_support.py`
- Modify: `tests/test_api_architecture_review.py`, `tests/test_api_boundaries.py`, `tests/test_camulator_component_kernels.py`, `tests/test_component_base_coverage.py`, `tests/test_distribution_boundaries.py`, `tests/test_final_review_boundaries.py`, `tests/test_native_period_output_compatibility.py`, `tests/test_public_api_contracts.py`, `tests/test_runtime_facade_boundaries.py`
- Modify: all renamed test modules, contract JSON files, and plugin fixture files above
- Modify: `CHANGELOG.md`, `DEPENDENCIES.md`, `DESIGN.md`, `PROGRESS.md`, `README.md`, `docs/api-architecture-review.md`, `docs/progress-archive-2026-05-16-to-2026-07-14.md`, `docs/releasing.md`
- Modify: `docs/superpowers/specs/2026-07-15-vercor-versioning-design.md`, `docs/superpowers/plans/2026-07-15-vercor-versioning.md`, and the renamed 2026-07-13 design/plan

**Interfaces:**
- Consumes: Task 1's failing policy and current-release assertions, the canonical mapping, pinned historical SHA `9f0b9131c889bed5c1c2d8ded260add3cfef9524`, and unchanged public signatures.
- Produces: importable renamed tests/fixtures, package metadata `0.4.0a1`, normalized historical evidence `0.3.2`, plugin dependency interval `vercor>=0.3,<0.4`, and zero version-policy violations.

- [ ] **Step 1: Rename every version-bearing path as one mechanical operation**

Perform exactly the renames listed in this task. After renaming, run:

```bash
rg --files | rg '(migration-3-to-4|vercor-4-api|vercor-3\.1\.1|vercor-4\.0\.0a1|public_plugin_3_0|test_v[24]_)'
```

Expected: no output.

- [ ] **Step 2: Correct current package, CI, artifact, and import identity**

Set `pyproject.toml` to:

```toml
[project]
name = "vercor"
version = "0.4.0a1"
```

In `.github/workflows/python-package.yml`, use these exact VerCOR-specific values while retaining GitHub Action revisions unchanged:

```yaml
python -m build --wheel --outdir dist tests/fixtures/public_plugin_0_3
WHEEL_PATH="${GITHUB_WORKSPACE}/dist/vercor-0.4.0a1-py3-none-any.whl"
SDIST_PATH="${GITHUB_WORKSPACE}/dist/vercor-0.4.0a1.tar.gz"
plugin-lane: [native-v0.4, historical-v0.3-artifact]
python -m pip install "${GITHUB_WORKSPACE}/dist/vercor-0.4.0a1-py3-none-any.whl"
python -c "import zipfile; p='dist/vercor_compat_plugin_0_3-0.1.0-py3-none-any.whl'; z=zipfile.ZipFile(p); m=z.read(next(n for n in z.namelist() if n.endswith('.dist-info/METADATA'))).decode(); assert 'Requires-Dist: vercor>=0.3,<0.4' in m"
python -m pip install "dist/vercor-0.4.0a1-py3-none-any.whl"
```

Rename the workflow step labels and conditions consistently to `native-v0.4` and `historical-v0.3-artifact`. Change imports such as:

```python
from tests.test_v0_4_public_api import PUBLIC_MODULE_EXPORTS
```

In `tests/_distribution_support.py` and `tests/test_distribution_boundaries.py`, set the frozen wheel name and fixture root to:

```python
EXPECTED_FROZEN_PLUGIN_WHEEL_NAME = (
    f"vercor_compat_plugin_0_3-{EXPECTED_FROZEN_PLUGIN_VERSION}-py3-none-any.whl"
)
FROZEN_PLUGIN_ROOT = PROJECT_ROOT / "tests" / "fixtures" / "public_plugin_0_3"
```

Keep `actions/checkout@v4`, `actions/upload-artifact@v4`, and `actions/download-artifact@v4` unchanged.

- [ ] **Step 3: Normalize the pinned historical API build without rewriting Git history**

In `tests/test_v0_4_compatibility_baseline.py`, use:

```python
MANIFEST_PATH = PROJECT_ROOT / "tests/contracts/vercor-0.3.2-public-api.json"
REFERENCE_SHA = "9f0b9131c889bed5c1c2d8ded260add3cfef9524"
REFERENCE_VERSION = "0.3.2"
ARCHIVED_VERSION = ".".join(("3", "1", "1"))
```

Immediately after extracting the pinned archive and before building its wheel, normalize only its project version:

```python
archived_pyproject = source_root / "pyproject.toml"
archived_metadata = archived_pyproject.read_text(encoding="utf-8")
incorrect_declaration = f'version = "{ARCHIVED_VERSION}"'
corrected_declaration = f'version = "{REFERENCE_VERSION}"'
assert archived_metadata.count(incorrect_declaration) == 1
archived_pyproject.write_text(
    archived_metadata.replace(incorrect_declaration, corrected_declaration),
    encoding="utf-8",
)
```

Use `vercor-0.3.2-reference.tar`, expect `vercor-0.3.2-py3-none-any.whl`, rename test functions/docstrings to 0.3.2, and use an unrelated `vercor-0.2.1-py3-none-any.whl` in the rejection test. Change the JSON manifest's `version` to `0.3.2` while preserving the SHA, exports, and signatures.

- [ ] **Step 4: Rename and correct the frozen plugin fixture**

Set `tests/fixtures/public_plugin_0_3/pyproject.toml` to:

```toml
[project]
name = "vercor-compat-plugin-0-3"
version = "0.1.0"
description = "Frozen installed-artifact fixture for VerCOR 0.3 public contracts"
requires-python = ">=3.12"
dependencies = ["vercor>=0.3,<0.4"]

[tool.flit.module]
name = "vercor_compat_plugin_0_3"
```

Change fixture imports to:

```python
from vercor_compat_plugin_0_3.plugin import run_smoke
```

and:

```python
from vercor_compat_plugin_0_3 import run_smoke
```

Keep the independently versioned plugin release at `0.1.0`.

- [ ] **Step 5: Correct source, tests, identifiers, and Markdown semantically**

Apply the canonical release mapping to explicit VerCOR releases and artifacts. Use `0.4` for the current architecture (`v0_4` inside Python identifiers), `0.3` for the frozen historical release fixture, and descriptive version-free names for arbitrary grids or helpers that used `v1`, `v2`, or `v3` only as internal generation markers.

Required examples include:

```python
"""Primary VerCOR 0.4 assembly conveniences."""
```

Rename `test_runtime_module_owns_only_the_v4_workflow_contracts` to
`test_runtime_module_owns_only_the_v0_4_workflow_contracts` without changing
its assertions.

```python
grid = make_test_grid(name="public-api-state")
```

Update all cross-references to the renamed migration, contract, plan, spec,
test, and fixture paths. In `CHANGELOG.md`, use the `0.4.0a1` heading and a
comparison link from `v0.3.2` to `v0.4.0a1`. In historical Markdown, replace
explicit releases using the canonical mapping; replace non-release generation
labels with neutral descriptions when applying a release number would create a
false chronology.

Remove the old candidate hash blocks from `PROGRESS.md`; do not retain their
hashes beside corrected filenames. Rewrite this plan and its design spec after
the migration so they describe the completed correction without containing the
forbidden former labels themselves.

- [ ] **Step 6: Refresh the progress-archive checksum contract**

Run:

```bash
shasum -a 256 docs/progress-archive-2026-05-16-to-2026-07-14.md
```

Copy the exact resulting digest into `PROGRESS_ARCHIVE_SHA256` in
`tests/test_api_architecture_review.py`. Do not change the archive after this
step without recalculating the digest.

- [ ] **Step 7: Verify the intended GREEN migration**

Run:

```bash
conda run -n scipy pytest tests/test_versioning_policy.py tests/test_api_architecture_review.py tests/test_v0_4_compatibility_baseline.py tests/test_distribution_boundaries.py -q --tb=short
conda run -n scipy pytest tests/ -q --fast --tb=short
```

Expected: all focused tests pass; the fast suite passes with no import errors,
missing paths, or forbidden VerCOR labels. GitHub Action `@v4` references and
external dependency versions remain unchanged.

- [ ] **Step 8: Commit the complete green migration**

```bash
git add .github pyproject.toml vercor tests README.md CHANGELOG.md DESIGN.md DEPENDENCIES.md PROGRESS.md docs
git commit -m "fix: correct VerCOR pre-1.0 version history"
```

### Task 3: Build corrected artifacts and complete release verification

**Files:**
- Modify: `PROGRESS.md`
- Verify: all production, example, test, workflow, fixture, JSON, and Markdown files changed by Task 2
- Build outside checkout: `/private/tmp/vercor-0.4.0a1-dist/`

**Interfaces:**
- Consumes: Task 2's green source tree and `0.4.0a1` build metadata.
- Produces: fresh wheel, source distribution, native plugin wheel, historical plugin wheel, exact hashes, final test/lint/type/build evidence, and a bounded progress handoff.

- [ ] **Step 1: Run formatting, lint, typing, and compilation gates**

Run:

```bash
conda run -n scipy black --check vercor examples tests
conda run -n scipy flake8 . --count --max-line-length=120 --statistics
conda run -n scipy mypy vercor examples tests
conda run -n scipy python -m compileall -q vercor examples tests
git diff --check
```

Expected: Black reports no changes needed; flake8 reports zero errors; mypy and
compileall exit zero; `git diff --check` emits no output.

- [ ] **Step 2: Run fast, full, coverage, optional-model, and gradient gates**

Run:

```bash
conda run -n scipy pytest tests/ -q --fast --tb=short
conda run -n scipy pytest tests/ -q --tb=short
conda run -n scipy pytest tests/ -q --tb=short --cov=vercor --cov-branch --cov-report=term-missing --cov-fail-under=90
conda run -n scipy pytest tests/test_setup_lifecycle_helpers.py::test_make_jcm_land_atmosphere_replaces_only_missing_forcing tests/test_external_components_coverage.py::test_jax_gcm_initialize_uses_provided_forcing_and_can_spin_up tests/test_external_components_coverage.py::test_jax_gcm_initialize_builds_default_forcing_when_missing tests/test_setup_boundaries.py::test_veros_implementation_import_does_not_configure_runtime tests/test_setup_boundaries.py::test_veros_factory_configures_once_before_implementation_import tests/test_external_components_coverage.py::test_veros_initialize_spinup_follows_enabled_only -q --tb=short
conda run -n scipy pytest tests/test_v0_4_workflow_execution.py::test_output_free_workflow_preserves_jvp_and_reverse_mode_gradients tests/test_v0_4_workflow_execution.py::test_payload_dependent_multi_step_scan_preserves_treedef_jvp_and_grad tests/test_v0_4_output_providers.py::test_all_disabled_target_remains_jit_and_gradient_compatible -q --tb=short
```

Expected: every suite passes, branch coverage is at least 90%, and only already
documented third-party warnings may remain.

- [ ] **Step 3: Build all four corrected artifacts from the verified source**

Run:

```bash
conda run -n scipy python -m build --outdir /private/tmp/vercor-0.4.0a1-dist
conda run -n scipy python -m build --wheel --outdir /private/tmp/vercor-0.4.0a1-dist tests/fixtures/public_plugin
conda run -n scipy python -m build --wheel --outdir /private/tmp/vercor-0.4.0a1-dist tests/fixtures/public_plugin_0_3
```

Expected files:

```text
vercor-0.4.0a1-py3-none-any.whl
vercor-0.4.0a1.tar.gz
vercor_public_plugin-0.1.0-py3-none-any.whl
vercor_compat_plugin_0_3-0.1.0-py3-none-any.whl
```

- [ ] **Step 4: Run the supplied-artifact boundary against the exact bundle**

Run:

```bash
VERCOR_ARTIFACT_DIR=/private/tmp/vercor-0.4.0a1-dist conda run -n scipy pytest tests/test_distribution_boundaries.py -q --tb=short
```

Expected: all distribution-boundary tests pass using the supplied wheel, sdist,
native plugin, and historical plugin artifacts.

- [ ] **Step 5: Record exact fresh hashes and verification evidence**

Run:

```bash
shasum -a 256 /private/tmp/vercor-0.4.0a1-dist/vercor-0.4.0a1-py3-none-any.whl /private/tmp/vercor-0.4.0a1-dist/vercor-0.4.0a1.tar.gz /private/tmp/vercor-0.4.0a1-dist/vercor_public_plugin-0.1.0-py3-none-any.whl /private/tmp/vercor-0.4.0a1-dist/vercor_compat_plugin_0_3-0.1.0-py3-none-any.whl
```

Update `PROGRESS.md` with the exact observed counts, coverage, warnings, artifact
directory, and four SHA-256 values. State explicitly that tagging, pushing, and
publication remain unperformed.

- [ ] **Step 6: Re-run policy and whitespace checks after recording evidence**

Run:

```bash
conda run -n scipy pytest tests/test_versioning_policy.py tests/test_api_architecture_review.py -q --tb=short
git diff --check
git status --short
```

Expected: policy/documentation tests pass, whitespace is clean, and status lists
only the intended final `PROGRESS.md` evidence update.

- [ ] **Step 7: Commit the verified release evidence**

```bash
git add PROGRESS.md
git commit -m "docs: record VerCOR 0.4.0a1 verification"
```

- [ ] **Step 8: Confirm the final repository state without publishing**

Run:

```bash
git status --short
git log -3 --oneline --decorate
```

Expected: the worktree is clean; the version-correction and verification commits
are present; no tag, push, upload, or release action has occurred.
