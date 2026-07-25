# Pyproject Version Source Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `pyproject.toml` the sole executable owner of VerCOR's package version and set the requested Apache license classifier.

**Architecture:** Tests read `[project].version` through the existing distribution-support boundary and derive every version-sensitive expectation. The GitHub Actions build job parses project metadata once, exposes exact version and artifact names as job outputs, and every consumer uses those outputs while retaining the exact two-file release boundary.

**Tech Stack:** Python 3.12+, `tomllib`, pytest, PyYAML, Bash, GitHub Actions, Flit

## Global Constraints

- The license classifier is exactly `License :: OSI Approved :: Apache Software License`.
- `pyproject.toml` is the only executable owner of the VerCOR package version.
- Python tests and GitHub workflow YAML contain no literal copy of the current VerCOR package version.
- Historical changelog, release-note, archived-plan, and progress identities remain literal.
- Python, dependency, action, schema, and external-extension fixture versions remain unchanged.
- Release artifact validation remains fail-closed and requires exactly the derived wheel and sdist.
- No tag, push, publication, release creation, or remote mutation is authorized.
- Use the `scipy` Conda environment and concise pytest output.
- Run fast and full tests before each implementation commit.

---

### Task 1: Correct the License Classifier

**Files:**
- Modify: `tests/test_distribution_boundaries.py:164`
- Modify: `pyproject.toml:22-32`

**Interfaces:**
- Consumes: `[project].classifiers` from `pyproject.toml`
- Produces: exactly one Apache license classifier with the requested PyPI classifier spelling

- [ ] **Step 1: Add the failing license metadata assertion**

Extend `test_runtime_metadata_separates_test_and_development_dependencies` with:

```python
    license_classifiers = {
        classifier
        for classifier in project["classifiers"]
        if classifier.startswith("License ::")
    }
    assert license_classifiers == {
        "License :: OSI Approved :: Apache Software License"
    }
```

This test catches a missing, duplicated, or incorrectly spelled project license
classifier.

- [ ] **Step 2: Run the focused test and record RED**

Run:

```bash
CONDA_NO_PLUGINS=true conda run -n scipy pytest \
  tests/test_distribution_boundaries.py::test_runtime_metadata_separates_test_and_development_dependencies \
  -q -n0 --tb=short
```

Expected: FAIL because the current classifier set contains
`License :: OSI Approved :: Apache-2.0`.

- [ ] **Step 3: Apply the minimal metadata change**

In `pyproject.toml`, replace only the license classifier:

```toml
    "License :: OSI Approved :: Apache Software License",
```

- [ ] **Step 4: Verify focused GREEN**

Run the Step 2 command again.

Expected: PASS.

- [ ] **Step 5: Run pre-commit regression gates**

Run:

```bash
CONDA_NO_PLUGINS=true conda run -n scipy pytest tests/ -q --fast --tb=short
CONDA_NO_PLUGINS=true conda run -n scipy pytest tests/ -q --tb=short
git diff --check
```

Expected: both pytest suites pass and `git diff --check` prints nothing.

- [ ] **Step 6: Commit the license correction**

```bash
git add pyproject.toml tests/test_distribution_boundaries.py
git commit -m "Update Apache license classifier"
```

---

### Task 2: Prove and Implement Single-Source Versioning

**Files:**
- Modify: `tests/test_versioning_policy.py:1-225`
- Modify: `tests/test_distribution_boundaries.py:1-1000`
- Modify: `tests/test_api_architecture_review.py:1-1020`
- Modify: `.github/workflows/python-package.yml:1-520`

**Interfaces:**
- Consumes: `EXPECTED_VERSION`, `EXPECTED_WHEEL_NAME`, and
  `EXPECTED_SDIST_NAME` derived from `pyproject.toml` by
  `tests._distribution_support`
- Produces: GitHub job outputs named `version`, `wheel_name`, and `sdist_name`
- Produces: downstream workflow paths derived only from
  `needs.build-artifacts.outputs`

- [ ] **Step 1: Remove tautological test-owned version state**

In `tests/test_versioning_policy.py`, remove:

```python
import tomllib
CURRENT_VERSION = "0.4.2"
```

Delete `test_current_vercor_release_is_the_approved_stable_release`. It only
compares a value parsed from `pyproject.toml` with a second copy of that value
and therefore cannot validate behavior independently.

In `test_runtime_metadata_separates_test_and_development_dependencies`, remove
the literal `project["version"]` equality assertion. Preserve all dependency,
coverage, and license assertions.

- [ ] **Step 2: Add a boundary test that rejects duplicated current versions**

Add this test to `tests/test_distribution_boundaries.py`:

```python
@pytest.mark.fast_always
def test_active_tests_and_workflows_do_not_duplicate_project_version() -> None:
    """Keep pyproject.toml as the only executable package-version owner."""

    workflow_paths = tuple(
        sorted((PROJECT_ROOT / ".github" / "workflows").glob("*.yml"))
    ) + tuple(sorted((PROJECT_ROOT / ".github" / "workflows").glob("*.yaml")))
    active_paths = (
        tuple(sorted((PROJECT_ROOT / "tests").rglob("*.py"))) + workflow_paths
    )
    violations = {
        str(path.relative_to(PROJECT_ROOT)): tuple(
            line_number
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(),
                start=1,
            )
            if EXPECTED_VERSION in line
        )
        for path in active_paths
    }

    assert {path: lines for path, lines in violations.items() if lines} == {}
```

The production mutation caught by this test is reintroducing the current
package version as a test or workflow literal.

- [ ] **Step 3: Add an executable CI metadata contract**

Add this test to `tests/test_distribution_boundaries.py`:

```python
@pytest.mark.fast_always
def test_ci_project_metadata_step_derives_outputs_from_pyproject(
    tmp_path: Path,
) -> None:
    """Execute metadata extraction against a non-repository project version."""

    workflow = yaml.safe_load(
        (PROJECT_ROOT / ".github/workflows/python-package.yml").read_text(
            encoding="utf-8"
        )
    )
    build_job = workflow["jobs"]["build-artifacts"]
    assert build_job["outputs"] == {
        "version": "${{ steps.project-metadata.outputs.version }}",
        "wheel_name": "${{ steps.project-metadata.outputs.wheel_name }}",
        "sdist_name": "${{ steps.project-metadata.outputs.sdist_name }}",
    }
    metadata_step = next(
        step
        for step in build_job["steps"]
        if step.get("id") == "project-metadata"
    )

    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "vercor"\nversion = "9.8.7"\n',
        encoding="utf-8",
    )
    github_output = tmp_path / "github-output"
    completed = subprocess.run(
        ["bash"],
        input=metadata_step["run"],
        cwd=tmp_path,
        env={**os.environ, "GITHUB_OUTPUT": str(github_output)},
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert github_output.read_text(encoding="utf-8").splitlines() == [
        "version=9.8.7",
        "wheel_name=vercor-9.8.7-py3-none-any.whl",
        "sdist_name=vercor-9.8.7.tar.gz",
    ]
```

This test executes the real workflow shell block. It fails if the metadata step
uses a literal version or derives the wrong artifact names.

- [ ] **Step 4: Convert existing test expectations to derived metadata**

In `tests/test_api_architecture_review.py`, import:

```python
from tests._distribution_support import (
    EXPECTED_SDIST_NAME,
    EXPECTED_VERSION,
    EXPECTED_WHEEL_NAME,
)
```

Derive all current-release expectations with:

```python
expected_tag = f"v{EXPECTED_VERSION}"
expected_title = f"VerCOR {EXPECTED_VERSION}"
expected_branch = f"release/vercor-{EXPECTED_VERSION}"
expected_release_notes = f"docs/release-notes-{EXPECTED_VERSION}.md"
```

Use these values for release-note paths, artifact names, PyPI URLs, GitHub tag
URLs, PR branch/title/body checks, tag preflights, release recovery checks, and
GitHub release commands. Preserve literal stale-tag history such as `v0.4.1`;
that is historical evidence, not current version ownership.

In `tests/test_distribution_boundaries.py`, replace current-version strings
with `EXPECTED_VERSION`, `EXPECTED_WHEEL_NAME`, `EXPECTED_SDIST_NAME`, and local
derived tag/title variables. Change the release-note entry in
`test_active_sources_do_not_use_retired_public_plugin_fixture_name` to enumerate
`docs/release-notes-*.md`, so historical release-note filenames remain
historical rather than becoming alternate current-version owners.

Change `test_progress_records_refreshed_corrected_release_artifacts` to locate
the existing entry containing
`evidence was refreshed after the README and release-runbook corrections`
instead of assuming it is always the first current-status entry.

Update `test_tag_release_rejects_version_mismatch_before_build` so it asserts:

```python
    assert guard["env"]["VERSION"] == (
        "${{ steps.project-metadata.outputs.version }}"
    )
```

Run its Bash block with `VERSION=EXPECTED_VERSION` in the subprocess
environment. Keep one matching tag and one deliberately mismatched synthetic
tag case.

Update workflow-structure expectations so they require:

```python
expected_outputs = {
    "version": "${{ steps.project-metadata.outputs.version }}",
    "wheel_name": "${{ steps.project-metadata.outputs.wheel_name }}",
    "sdist_name": "${{ steps.project-metadata.outputs.sdist_name }}",
}
```

and require downstream step environments to reference:

```python
"${{ needs.build-artifacts.outputs.version }}"
"${{ needs.build-artifacts.outputs.wheel_name }}"
"${{ needs.build-artifacts.outputs.sdist_name }}"
```

Do not weaken assertions for exact inventory size, checksums, immutable action
pins, installed-artifact origins, release-tag matching, or publication state.

- [ ] **Step 5: Run the versioning contracts and record RED**

Run:

```bash
CONDA_NO_PLUGINS=true conda run -n scipy pytest \
  tests/test_distribution_boundaries.py \
  tests/test_api_architecture_review.py \
  tests/test_versioning_policy.py \
  -q -n0 --tb=short
```

Expected: FAIL because `build-artifacts` has no metadata outputs and the
workflow still contains literal artifact names. The license test remains green.

- [ ] **Step 6: Add the build-job metadata outputs**

Add this mapping to `build-artifacts`:

```yaml
    outputs:
      version: ${{ steps.project-metadata.outputs.version }}
      wheel_name: ${{ steps.project-metadata.outputs.wheel_name }}
      sdist_name: ${{ steps.project-metadata.outputs.sdist_name }}
```

Add this step immediately after Python setup:

```yaml
      - name: Read project metadata
        id: project-metadata
        run: |
          set -euo pipefail
          VERSION="$(python -c 'import pathlib, tomllib; print(tomllib.loads(pathlib.Path("pyproject.toml").read_text(encoding="utf-8"))["project"]["version"])')"
          WHEEL_NAME="vercor-${VERSION}-py3-none-any.whl"
          SDIST_NAME="vercor-${VERSION}.tar.gz"
          {
            echo "version=${VERSION}"
            echo "wheel_name=${WHEEL_NAME}"
            echo "sdist_name=${SDIST_NAME}"
          } >> "$GITHUB_OUTPUT"
```

This step exits nonzero for missing or malformed project metadata because the
shell uses `set -euo pipefail` and Python exceptions are not suppressed.

- [ ] **Step 7: Make build and tag validation consume the metadata step**

Give `Reject mismatched release tag`:

```yaml
        env:
          VERSION: ${{ steps.project-metadata.outputs.version }}
```

Remove its inline `tomllib` command and retain the exact
`GITHUB_REF_NAME == v${VERSION}` check.

Give `Build VerCOR distributions once`:

```yaml
        env:
          WHEEL_NAME: ${{ steps.project-metadata.outputs.wheel_name }}
          SDIST_NAME: ${{ steps.project-metadata.outputs.sdist_name }}
```

Use quoted derived names while preserving the two-file boundary:

```bash
test -f "dist/${WHEEL_NAME}"
test -f "dist/${SDIST_NAME}"
mkdir -p release-manifest
(
  cd dist
  sha256sum "$WHEEL_NAME" "$SDIST_NAME" > ../release-manifest/SHA256SUMS
  sha256sum -c ../release-manifest/SHA256SUMS
)
```

Set the distribution upload paths to:

```yaml
          path: |
            dist/${{ steps.project-metadata.outputs.wheel_name }}
            dist/${{ steps.project-metadata.outputs.sdist_name }}
```

- [ ] **Step 8: Propagate exact artifact outputs to downstream jobs**

On the installed-artifact install step, add:

```yaml
        env:
          WHEEL_NAME: ${{ needs.build-artifacts.outputs.wheel_name }}
          SDIST_NAME: ${{ needs.build-artifacts.outputs.sdist_name }}
```

Set:

```bash
WHEEL_PATH="${GITHUB_WORKSPACE}/dist/${WHEEL_NAME}"
SDIST_PATH="${GITHUB_WORKSPACE}/dist/${SDIST_NAME}"
```

On external-extension and macOS smoke steps, add:

```yaml
        env:
          WHEEL_NAME: ${{ needs.build-artifacts.outputs.wheel_name }}
```

Install respectively:

```bash
python -m pip install "${GITHUB_WORKSPACE}/dist/${WHEEL_NAME}"
python -m pip install "dist/${WHEEL_NAME}"
```

Do not change the independent external extension fixture artifact name or
version.

- [ ] **Step 9: Reverify publish metadata against pyproject**

Give `Verify CI-produced release inventory`:

```yaml
        env:
          PROJECT_VERSION: ${{ needs.build-artifacts.outputs.version }}
          WHEEL_NAME: ${{ needs.build-artifacts.outputs.wheel_name }}
          SDIST_NAME: ${{ needs.build-artifacts.outputs.sdist_name }}
```

At the start of the shell block, parse `pyproject.toml` into `VERSION`, then
require all propagated values to match:

```bash
VERSION="$(python -c 'import pathlib, tomllib; print(tomllib.loads(pathlib.Path("pyproject.toml").read_text(encoding="utf-8"))["project"]["version"])')"
test "$PROJECT_VERSION" = "$VERSION"
test "$WHEEL_NAME" = "vercor-${VERSION}-py3-none-any.whl"
test "$SDIST_NAME" = "vercor-${VERSION}.tar.gz"
test "$GITHUB_REF_NAME" = "v${VERSION}"
WHEEL="dist/${WHEEL_NAME}"
SDIST="dist/${SDIST_NAME}"
```

Remove duplicate assignments that synthesize `WHEEL_NAME` or `SDIST_NAME`
inside this step. Preserve manifest validation, exact inventory, release-note
existence, tag/SHA checks, and all exported release environment variables.

- [ ] **Step 10: Run focused GREEN**

Run the Step 5 command again.

Expected: all selected tests pass with no VerCOR package-version literal in
tests or workflow YAML.

- [ ] **Step 11: Validate YAML and every Bash block**

Run:

```bash
CONDA_NO_PLUGINS=true conda run -n scipy python -c \
  'import pathlib,yaml; yaml.safe_load(pathlib.Path(".github/workflows/python-package.yml").read_text(encoding="utf-8")); print("workflow YAML OK")'
CONDA_NO_PLUGINS=true conda run -n scipy pytest \
  tests/test_api_architecture_review.py::test_workflow_run_blocks_are_bash_syntax_valid \
  tests/test_distribution_boundaries.py::test_ci_project_metadata_step_derives_outputs_from_pyproject \
  tests/test_distribution_boundaries.py::test_tag_release_rejects_version_mismatch_before_build \
  -q -n0 --tb=short
```

Expected: YAML parses and all three executable workflow contracts pass.

- [ ] **Step 12: Run static, fast, and full gates**

Run:

```bash
CONDA_NO_PLUGINS=true conda run -n scipy black --check vercor examples tests
CONDA_NO_PLUGINS=true conda run -n scipy flake8 . --count --max-line-length=120 --statistics
CONDA_NO_PLUGINS=true conda run -n scipy mypy vercor examples tests
CONDA_NO_PLUGINS=true conda run -n scipy python -m compileall -q vercor examples tests
CONDA_NO_PLUGINS=true conda run -n scipy pytest tests/ -q --fast --tb=short
CONDA_NO_PLUGINS=true conda run -n scipy pytest tests/ -q --tb=short
git diff --check
```

Expected: every command exits zero. Only already-known third-party warnings may
remain.

- [ ] **Step 13: Commit the single-source refactor**

```bash
git add .github/workflows/python-package.yml \
  tests/test_api_architecture_review.py \
  tests/test_distribution_boundaries.py \
  tests/test_versioning_policy.py
git commit -m "Derive package version from pyproject"
```

---

### Task 3: Record and Reverify Release-Ready Evidence

**Files:**
- Modify: `PROGRESS.md:5`

**Interfaces:**
- Consumes: final command output from Tasks 1 and 2
- Produces: a concise current-status entry describing the license, dynamic
  version flow, RED/GREEN evidence, and final validation

- [ ] **Step 1: Run fresh final verification**

Use superpowers:verification-before-completion, then run:

```bash
CONDA_NO_PLUGINS=true conda run -n scipy pytest \
  tests/test_distribution_boundaries.py \
  tests/test_api_architecture_review.py \
  tests/test_versioning_policy.py \
  -q -n0 --tb=short
CONDA_NO_PLUGINS=true conda run -n scipy pytest tests/ -q --fast --tb=short
CONDA_NO_PLUGINS=true conda run -n scipy pytest tests/ -q --tb=short
git diff --check
```

Expected: all commands exit zero with fresh output from the final source state.

- [ ] **Step 2: Update the active progress log**

Prepend one dated bullet under `## Current Status` recording:

- the exact license classifier correction;
- `pyproject.toml` as the only executable VerCOR version owner;
- build-job outputs and downstream workflow consumption;
- the intended RED failures and focused GREEN result;
- static, fast, and full suite results with actual pass counts; and
- that no tag, push, publication, release, or remote mutation occurred.

Use the actual command results; do not predict counts or hashes.

- [ ] **Step 3: Recheck progress-sensitive and fast tests**

Run:

```bash
CONDA_NO_PLUGINS=true conda run -n scipy pytest \
  tests/test_distribution_boundaries.py::test_progress_records_refreshed_corrected_release_artifacts \
  tests/test_versioning_policy.py \
  -q -n0 --tb=short
CONDA_NO_PLUGINS=true conda run -n scipy pytest tests/ -q --fast --tb=short
git diff --check
```

Expected: all tests pass and whitespace is clean.

- [ ] **Step 4: Commit the verification record**

```bash
git add PROGRESS.md
git commit -m "docs: record version source verification"
```

- [ ] **Step 5: Request final code review**

Use superpowers:requesting-code-review against the complete change from the
design-spec parent through `HEAD`. Resolve any actionable findings, rerun the
affected focused tests plus the final fast suite, and report the reviewed commit
range without pushing or opening a pull request.
