# External Extension Test Fixture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the installed extension compatibility fixture and keep its wheel entirely outside VerCOR release and shared CI artifacts.

**Architecture:** Separate VerCOR distribution construction from temporary extension-fixture construction in the test helper. CI and release checks build the renamed fixture only in temporary directories, while `dist/`, `SHA256SUMS`, PyPI uploads, GitHub release assets, and the shared CI artifact contain only the VerCOR wheel and source distribution.

**Tech Stack:** Python 3.12/3.13, Flit, pytest, mypy, GitHub Actions YAML, Bash release transcripts, JAX.

## Global Constraints

- The fixture source directory is `tests/fixtures/external_extension_test_fixture`.
- The distribution name is `external-extension-test-fixture`.
- The import package is `external_extension_test_fixture`.
- The wheel name is `external_extension_test_fixture-0.1.0-py3-none-any.whl`.
- VerCOR release artifacts are exactly `vercor-0.4.0-py3-none-any.whl` and `vercor-0.4.0.tar.gz`.
- The fixture wheel must never enter `dist/`, `dist/SHA256SUMS`, the shared `vercor-distributions` CI artifact, PyPI uploads, or GitHub release assets.
- Generic documentation may continue to use “plugin” for third-party VerCOR extensions.
- Dated plans, specifications, and archived progress remain historical and are not rewritten.
- Do not change VerCOR runtime APIs, numerical behavior, or dependency versions.
- Use the direct `scipy` interpreter at `/Users/romannuterman/miniforge3/envs/scipy/bin/python`.

---

### Task 1: Rename and separate the installed extension fixture

**Files:**
- Move: `tests/fixtures/public_plugin/pyproject.toml` to `tests/fixtures/external_extension_test_fixture/pyproject.toml`
- Move: `tests/fixtures/public_plugin/use_site.py` to `tests/fixtures/external_extension_test_fixture/use_site.py`
- Move: `tests/fixtures/public_plugin/src/vercor_public_plugin/__init__.py` to `tests/fixtures/external_extension_test_fixture/src/external_extension_test_fixture/__init__.py`
- Move: `tests/fixtures/public_plugin/src/vercor_public_plugin/plugin.py` to `tests/fixtures/external_extension_test_fixture/src/external_extension_test_fixture/plugin.py`
- Move: `tests/fixtures/public_plugin/src/vercor_public_plugin/smoke.py` to `tests/fixtures/external_extension_test_fixture/src/external_extension_test_fixture/smoke.py`
- Move: `tests/fixtures/public_plugin/src/vercor_public_plugin/py.typed` to `tests/fixtures/external_extension_test_fixture/src/external_extension_test_fixture/py.typed`
- Modify: `tests/_distribution_support.py`
- Modify: `tests/test_distribution_boundaries.py`
- Modify: `tests/test_v0_4_public_api.py`
- Modify: `tests/test_versioning_policy.py`

**Interfaces:**
- Consumes: the stable `vercor>=0.4.0,<0.5` public extension tier.
- Produces: `build_distributions(project_root, output_dir, *, artifact_dir=None, wheel_path=None, sdist_path=None) -> BuiltDistributions`.
- Produces: `build_external_extension_fixture(project_root, output_dir) -> Path`.
- Produces: `install_local_target(*, wheel, extension_fixture_wheel, target) -> None`.
- Produces: `EXPECTED_EXTENSION_FIXTURE_WHEEL_NAME = "external_extension_test_fixture-0.1.0-py3-none-any.whl"`.

- [ ] **Step 1: Change the contract tests before moving or refactoring implementation**

In `tests/test_distribution_boundaries.py`, replace the plugin constants and
fixtures with the desired public test contract:

```python
from tests._distribution_support import (
    BuiltDistributions,
    EXPECTED_EXTENSION_FIXTURE_WHEEL_NAME,
    EXPECTED_SDIST_NAME,
    EXPECTED_VERSION,
    EXPECTED_WHEEL_NAME,
    build_distributions,
    build_external_extension_fixture,
    install_local_target,
)

EXTERNAL_EXTENSION_FIXTURE_ROOT = (
    PROJECT_ROOT / "tests" / "fixtures" / "external_extension_test_fixture"
)


@pytest.fixture(scope="module")
def external_extension_fixture_wheel(
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    return build_external_extension_fixture(
        PROJECT_ROOT,
        tmp_path_factory.mktemp("external-extension-fixture-build"),
    )
```

Rename fixture-boundary tests and require these files:

```python
def test_external_extension_test_fixture_is_present() -> None:
    required_files = (
        "pyproject.toml",
        "src/external_extension_test_fixture/__init__.py",
        "src/external_extension_test_fixture/plugin.py",
        "src/external_extension_test_fixture/smoke.py",
        "src/external_extension_test_fixture/py.typed",
        "use_site.py",
    )
    for relative_path in required_files:
        assert (EXTERNAL_EXTENSION_FIXTURE_ROOT / relative_path).is_file()
```

Change `test_distribution_helper_reuses_explicit_artifact_directory_without_building`
so the supplied directory contains only `EXPECTED_WHEEL_NAME` and
`EXPECTED_SDIST_NAME`, then assert:

```python
assert distributions.wheel == wheel
assert distributions.sdist == sdist
assert distributions.build_pythonpath == ""
assert tuple(artifact_dir.iterdir()) == (wheel, sdist)
```

Change composed-install tests to receive `external_extension_fixture_wheel:
Path` separately and use the new import module and smoke command:

```python
install_local_target(
    wheel=built_distributions.wheel,
    extension_fixture_wheel=external_extension_fixture_wheel,
    target=target,
)
```

```python
[
    sys.executable,
    "-m",
    "external_extension_test_fixture.smoke",
    "--output-dir",
    str(smoke_output),
]
```

Update `tests/test_v0_4_public_api.py` to prepend
`tests/fixtures/external_extension_test_fixture/src`, import
`PluginRegridderFactory` from `external_extension_test_fixture`, and inspect
`src/external_extension_test_fixture/plugin.py`.

Update the external-version example in `tests/test_versioning_policy.py`:

```python
"external_extension_test_fixture-0.1.0-py3-none-any.whl",
```

- [ ] **Step 2: Run the new contracts and confirm RED**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest \
  tests/test_distribution_boundaries.py \
  tests/test_v0_4_public_api.py::test_regridder_factory_is_one_runtime_protocol_with_public_hints \
  tests/test_v0_4_public_api.py::test_examples_and_current_plugin_use_direct_constructor_assembly \
  tests/test_versioning_policy.py::test_release_shorthand_matcher_allows_external_and_numeric_labels \
  -q --tb=short
```

Expected: collection or assertion failures because
`build_external_extension_fixture`,
`EXPECTED_EXTENSION_FIXTURE_WHEEL_NAME`, and the renamed fixture paths do not
exist.

- [ ] **Step 3: Refactor the distribution helper into release and fixture responsibilities**

In `tests/_distribution_support.py`, define:

```python
EXPECTED_EXTENSION_FIXTURE_VERSION = "0.1.0"
EXPECTED_EXTENSION_FIXTURE_WHEEL_NAME = (
    "external_extension_test_fixture-"
    f"{EXPECTED_EXTENSION_FIXTURE_VERSION}-py3-none-any.whl"
)


@dataclass(frozen=True)
class BuiltDistributions:
    """Paths to the two publishable VerCOR distributions."""

    wheel: Path
    sdist: Path
    build_pythonpath: str
```

Make `_existing_distributions(wheel: Path, sdist: Path) -> BuiltDistributions`
validate only the two VerCOR names and files. Remove `plugin_wheel_path` and
`VERCOR_PLUGIN_WHEEL_PATH` handling from `build_distributions`.

Extract the shared offline build environment:

```python
def _build_environment() -> tuple[dict[str, str], str]:
    """Return a subprocess environment and any offline backend path."""

    build_pythonpath = _cached_build_pythonpath()
    environment = os.environ.copy()
    if build_pythonpath:
        environment["PYTHONPATH"] = build_pythonpath
    return environment, build_pythonpath
```

Add the temporary fixture builder:

```python
def build_external_extension_fixture(
    project_root: Path,
    output_dir: Path,
) -> Path:
    """Build the external extension test fixture outside release artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    environment, _ = _build_environment()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--no-isolation",
            "--wheel",
            "--outdir",
            str(output_dir),
            str(
                project_root
                / "tests"
                / "fixtures"
                / "external_extension_test_fixture"
            ),
        ],
        check=True,
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
    )
    fixture_wheel = output_dir / EXPECTED_EXTENSION_FIXTURE_WHEEL_NAME
    if not fixture_wheel.is_file():
        raise RuntimeError(
            "external extension test fixture build did not produce "
            f"{fixture_wheel}"
        )
    return fixture_wheel
```

Change `install_local_target` to accept
`extension_fixture_wheel: Path` and install `(wheel,
extension_fixture_wheel)`.

- [ ] **Step 4: Move and rename the fixture without changing its behavior**

Use `apply_patch` moves for the six fixture files. In the moved
`pyproject.toml`, use:

```toml
[project]
name = "external-extension-test-fixture"
version = "0.1.0"
description = "Installed test fixture for VerCOR public extension contracts"
requires-python = ">=3.12"
dependencies = ["vercor>=0.4.0,<0.5"]

[tool.flit.module]
name = "external_extension_test_fixture"
```

Change all internal imports to:

```python
from external_extension_test_fixture.plugin import run_smoke
```

Change `use_site.py` to import from `external_extension_test_fixture` and call
the artifact an “external extension test fixture.” Keep the existing smoke
values, component configuration, and public VerCOR imports unchanged.

- [ ] **Step 5: Run the focused contracts and confirm GREEN**

Run the exact command from Step 2.

Expected: all selected tests pass.

- [ ] **Step 6: Commit the isolated fixture/helper unit**

```bash
git add tests/_distribution_support.py tests/test_distribution_boundaries.py \
  tests/test_v0_4_public_api.py tests/test_versioning_policy.py \
  tests/fixtures/external_extension_test_fixture tests/fixtures/public_plugin
git diff --cached --check
git commit -m "test: isolate external extension fixture artifacts"
```

---

### Task 2: Remove the fixture from CI and release artifacts

**Files:**
- Modify: `.github/workflows/python-package.yml`
- Modify: `tests/test_distribution_boundaries.py`
- Modify: `tests/test_api_architecture_review.py`
- Modify: `docs/releasing.md`

**Interfaces:**
- Consumes: `external_extension_test_fixture-0.1.0-py3-none-any.whl` built in a temporary directory.
- Produces: shared CI artifact `vercor-distributions` containing exactly the VerCOR wheel and source distribution.
- Produces: `dist/SHA256SUMS` containing exactly two checksums.

- [ ] **Step 1: Write failing CI and release-boundary assertions**

In `test_ci_validates_installed_artifacts_across_supported_environments`, require
the build job to omit the fixture and require the extension job to build it
under `RUNNER_TEMP`:

```python
assert "tests/fixtures/external_extension_test_fixture" not in build_commands
assert "external_extension_test_fixture" not in build_commands
assert upload_step["with"]["path"] == "dist/"

extension_job = jobs["external-extension-contract-tests"]
extension_commands = "\n".join(
    step.get("run", "")
    for step in extension_job["steps"]
    if isinstance(step, dict)
)
assert (
    'python -m build --wheel --outdir "$RUNNER_TEMP/'
    'external-extension-fixture-dist" '
    "tests/fixtures/external_extension_test_fixture"
) in extension_commands
assert EXPECTED_EXTENSION_FIXTURE_WHEEL_NAME in extension_commands
assert "external_extension_test_fixture.smoke" in extension_commands
```

Add a release-guide assertion:

```python
def test_release_bundle_contains_only_vercor_distributions() -> None:
    releasing = RELEASING_PATH.read_text(encoding="utf-8")
    checksum_line = next(
        line.strip()
        for line in releasing.splitlines()
        if line.strip().startswith("shasum -a 256 vercor-")
    )
    assert checksum_line == (
        "shasum -a 256 vercor-0.4.0-py3-none-any.whl "
        "vercor-0.4.0.tar.gz > SHA256SUMS"
    )
    assert "dist/external_extension_test_fixture" not in releasing
```

Update `tests/test_api_architecture_review.py` so current release commands must
contain `python -m external_extension_test_fixture.smoke`, but the published
artifact loop contains only:

```python
for artifact in (
    "vercor-0.4.0-py3-none-any.whl",
    "vercor-0.4.0.tar.gz",
):
    assert artifact in commands
```

- [ ] **Step 2: Run the CI/release contracts and confirm RED**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest \
  tests/test_distribution_boundaries.py::test_ci_validates_installed_artifacts_across_supported_environments \
  tests/test_distribution_boundaries.py::test_release_bundle_contains_only_vercor_distributions \
  tests/test_api_architecture_review.py::test_release_files_and_metadata_describe_the_stable_release \
  -q --tb=short
```

Expected: failures because the build job still puts the old fixture wheel in
`dist/`, the old job identifier remains, and the checksum manifest has three
entries.

- [ ] **Step 3: Make the CI fixture build temporary**

In `.github/workflows/python-package.yml`:

- rename `plugin-contract-tests` to `external-extension-contract-tests`;
- change the build job label to “Build VerCOR distributions once”;
- remove the fixture build from `build-artifacts`;
- keep the uploaded path as `dist/`;
- in the extension job, install `build`, build the fixture to
  `$RUNNER_TEMP/external-extension-fixture-dist`, install its renamed wheel,
  run `external_extension_test_fixture.smoke`, and type-check the installed
  `external_extension_test_fixture` plus `external_extension_use_site.py`;
- add `actions/checkout@v4` with the triggering SHA to the macOS job; and
- build the fixture into `$RUNNER_TEMP/external-extension-fixture-dist` before
  installing and running it on macOS.

The core extension-job commands are:

```bash
python -m pip install --upgrade pip build mypy
python -m build --wheel \
  --outdir "$RUNNER_TEMP/external-extension-fixture-dist" \
  tests/fixtures/external_extension_test_fixture
python -m pip install "${GITHUB_WORKSPACE}/dist/vercor-0.4.0-py3-none-any.whl"
python -m pip install \
  "$RUNNER_TEMP/external-extension-fixture-dist/external_extension_test_fixture-0.1.0-py3-none-any.whl"
cd "$RUNNER_TEMP"
python -m external_extension_test_fixture.smoke \
  --output-dir "$RUNNER_TEMP/extension-output"
```

- [ ] **Step 4: Restrict the release guide to two publishable artifacts**

In `docs/releasing.md`:

- build only VerCOR into `dist/`;
- create `SHA256SUMS` from exactly the wheel and source distribution;
- build the fixture in `external_extension_fixture_dir="$(mktemp -d)"`;
- install its wheel from that temporary directory;
- run `external_extension_test_fixture.smoke`; and
- state explicitly that the temporary fixture wheel is neither checksummed nor
  published.

Use this acceptance fragment:

```bash
external_extension_fixture_dir="$(mktemp -d)"
python -m build --wheel \
  --outdir "$external_extension_fixture_dir" \
  tests/fixtures/external_extension_test_fixture
python -m pip install --target "$smoke_dir/site" \
  "$external_extension_fixture_dir/external_extension_test_fixture-0.1.0-py3-none-any.whl"
PYTHONPATH="$smoke_dir/site" \
  python -m external_extension_test_fixture.smoke \
  --output-dir "$smoke_dir/extension-output"
```

- [ ] **Step 5: Run the focused CI/release contracts and confirm GREEN**

Run the exact command from Step 2.

Expected: 3 passed.

- [ ] **Step 6: Commit the CI and release boundary**

```bash
git add .github/workflows/python-package.yml docs/releasing.md \
  tests/test_distribution_boundaries.py tests/test_api_architecture_review.py
git diff --cached --check
git commit -m "ci: keep extension fixture outside release artifacts"
```

---

### Task 3: Align active architecture and release documentation

**Files:**
- Modify: `DESIGN.md`
- Modify: `DEPENDENCIES.md`
- Modify: `CHANGELOG.md`
- Modify: `docs/release-notes-0.4.0.md`
- Modify: `docs/plugin-authoring.md`
- Modify: `docs/api-architecture-review.md`
- Modify: `tests/test_distribution_boundaries.py`

**Interfaces:**
- Consumes: the implemented temporary fixture boundary from Tasks 1 and 2.
- Produces: active documentation consistently naming the external extension test fixture.

- [ ] **Step 1: Add an active-source legacy-name test**

Add to `tests/test_distribution_boundaries.py`:

```python
@pytest.mark.fast_always
def test_active_sources_do_not_use_retired_public_plugin_fixture_name() -> None:
    active_paths = (
        PROJECT_ROOT / ".github" / "workflows" / "python-package.yml",
        PROJECT_ROOT / "DESIGN.md",
        PROJECT_ROOT / "DEPENDENCIES.md",
        PROJECT_ROOT / "CHANGELOG.md",
        PROJECT_ROOT / "docs" / "release-notes-0.4.0.md",
        PROJECT_ROOT / "docs" / "plugin-authoring.md",
        PROJECT_ROOT / "docs" / "api-architecture-review.md",
        PROJECT_ROOT / "docs" / "releasing.md",
        PROJECT_ROOT / "tests" / "_distribution_support.py",
        PROJECT_ROOT / "tests" / "test_distribution_boundaries.py",
        PROJECT_ROOT / "tests" / "test_api_architecture_review.py",
        PROJECT_ROOT / "tests" / "test_v0_4_public_api.py",
    )
    retired_markers = (
        "tests/fixtures/" + "public_plugin",
        "vercor_" + "public_plugin",
        "public-" + "plugin fixture",
        "public " + "plugin fixture",
    )
    violations = {
        str(path.relative_to(PROJECT_ROOT)): marker
        for path in active_paths
        for marker in retired_markers
        if marker in path.read_text(encoding="utf-8")
    }
    assert violations == {}
```

- [ ] **Step 2: Run the legacy-name test and confirm RED**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest \
  tests/test_distribution_boundaries.py::test_active_sources_do_not_use_retired_public_plugin_fixture_name \
  -q --tb=short
```

Expected: failure listing active documents that still use the old fixture
identity.

- [ ] **Step 3: Update active documentation without rewriting generic plugin terminology**

Make these exact semantic changes:

- `DESIGN.md`: describe a temporarily built external extension test fixture;
  state that CI uploads only the two VerCOR distributions and builds the
  fixture per contract job.
- `DEPENDENCIES.md`: point entry 24 at
  `tests/fixtures/external_extension_test_fixture/` and describe temporary
  installed-extension verification.
- `CHANGELOG.md` and `docs/release-notes-0.4.0.md`: replace public-plugin
  artifact claims with temporary installed external-extension fixture
  verification.
- `docs/plugin-authoring.md`: keep the generic plugin authoring title and
  guidance, but point the installed example at
  `tests/fixtures/external_extension_test_fixture`.
- `docs/api-architecture-review.md`: use the renamed path and explain that the
  wheel is built in a temporary directory rather than installed next to or
  uploaded with VerCOR release artifacts.

- [ ] **Step 4: Run the active documentation and architecture tests**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest \
  tests/test_distribution_boundaries.py \
  tests/test_api_architecture_review.py \
  tests/test_plugin_architecture.py \
  tests/test_v0_4_public_api.py \
  -q --fast --tb=short
```

Expected: all selected fast tests pass.

- [ ] **Step 5: Commit the active documentation**

```bash
git add DESIGN.md DEPENDENCIES.md CHANGELOG.md \
  docs/release-notes-0.4.0.md docs/plugin-authoring.md \
  docs/api-architecture-review.md tests/test_distribution_boundaries.py
git diff --cached --check
git commit -m "docs: clarify external extension test fixture"
```

---

### Task 4: Verify, record progress, and finalize

**Files:**
- Modify: `PROGRESS.md`

**Interfaces:**
- Consumes: the complete renamed fixture and release boundary.
- Produces: concise verification evidence and a clean committed worktree.

- [ ] **Step 1: Format and run static gates**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m black \
  vercor examples tests
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m flake8 . \
  --count --max-line-length=120 --statistics
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m mypy \
  vercor examples tests
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m mypy --strict \
  tests/fixtures/external_extension_test_fixture/src/external_extension_test_fixture \
  tests/fixtures/external_extension_test_fixture/use_site.py
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m compileall -q \
  vercor examples tests
```

Expected: Black completes without changes after formatting, flake8 reports zero
violations, both mypy commands pass, and compileall exits zero.

- [ ] **Step 2: Run fast, full, and branch-coverage suites**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest \
  tests/ -q --fast --tb=short
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest \
  tests/ -q --tb=short
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest \
  tests/ -q --cov=vercor --cov-branch --cov-report=term-missing \
  --cov-fail-under=90 --tb=short
```

Expected: all selected tests pass and branch coverage remains at least 90%.

- [ ] **Step 3: Build and inspect release and fixture artifacts separately**

Run:

```bash
release_dist_dir="$(mktemp -d)"
extension_fixture_dist_dir="$(mktemp -d)"
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m build \
  --outdir "$release_dist_dir"
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m build --wheel \
  --outdir "$extension_fixture_dist_dir" \
  tests/fixtures/external_extension_test_fixture
test "$(ls -1 "$release_dist_dir" | wc -l | tr -d ' ')" = "2"
test -f "$release_dist_dir/vercor-0.4.0-py3-none-any.whl"
test -f "$release_dist_dir/vercor-0.4.0.tar.gz"
test -f "$extension_fixture_dist_dir/external_extension_test_fixture-0.1.0-py3-none-any.whl"
```

Expected: the release directory contains exactly two VerCOR artifacts and the
fixture wheel exists only in its temporary fixture directory.

- [ ] **Step 4: Record durable verification in `PROGRESS.md`**

Add a dated first bullet under `## Current Status` containing:

- the complete rename;
- the two-file release/CI/checksum boundary;
- the focused RED failure reason and GREEN counts;
- Black, flake8, mypy, strict fixture mypy, and compileall outcomes;
- fast/full/coverage pass counts and coverage percentage;
- separate release and fixture build inspection results; and
- confirmation that no push, publication, upload, tag, or hosted release was
  performed.

- [ ] **Step 5: Re-run documentation focus and final repository checks**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest \
  tests/test_distribution_boundaries.py \
  tests/test_api_architecture_review.py \
  tests/test_plugin_architecture.py \
  tests/test_v0_4_public_api.py \
  -q --fast --tb=short
git diff --check
git status --short
```

Expected: all selected tests pass, `git diff --check` is silent, and only the
intended `PROGRESS.md` plus any Black formatting changes remain uncommitted.

- [ ] **Step 6: Review the complete diff and commit final evidence**

```bash
git diff --stat 67885d1
git diff --check
git add PROGRESS.md
git diff --cached --check
git commit -m "docs: record external extension fixture verification"
git status --short --branch
```

Expected: the final commit succeeds and the worktree is clean.
