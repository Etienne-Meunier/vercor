# Test-suite Artifact Reuse Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the duplicate VerCOR wheel build from the complete pytest gate while preserving every build, installation, public-boundary, isolation, and coverage guarantee.

**Architecture:** A session-scoped root fixture builds or resolves one frozen `BuiltDistributions` bundle. Distribution tests and the installed public-API boundary share only those immutable artifact paths; every installation directory and interpreter probe remains test-local.

**Tech Stack:** Python 3.13, pytest 9.1, subprocess-based offline builds and installs, pytest-cov/coverage.py, Black, flake8, and mypy.

## Global Constraints

- Optimize the complete `pytest tests/` gate, including artifact, optional-dependency, JAX transformation, and numerical contracts.
- Do not delete, skip, disable, quarantine, xfail, merge, or weaken tests or assertions.
- Preserve at least 93.05% statement/line coverage, 78.36% branch coverage, 90.52% combined branch-aware coverage, and 92.04% named-function entry coverage.
- Preserve fresh installation directories and interpreter isolation for installed-distribution probes.
- Do not introduce shared mutable state, test-order dependence, pytest-xdist, or production-code changes in this batch.
- Use the direct `/Users/romannuterman/miniforge3/envs/scipy/bin/python` interpreter because the Conda launcher can panic before pytest.
- Keep `PROGRESS.md` at or below its executable 180-line limit.
- Do not tag, push, publish, upload, or create a release.

---

## File structure

- `tests/conftest.py`: owns the session-scoped immutable distribution fixture available to every test module.
- `tests/test_distribution_boundaries.py`: retains artifact behavior tests and adds the static contract preventing a second build path in the installed public-boundary test.
- `tests/test_v0_4_public_api.py`: installs and inspects the shared VerCOR wheel in a unique temporary target; it no longer builds a duplicate wheel.
- `PROGRESS.md`: records durable timing, coverage, quality-gate, and remaining-bottleneck evidence.

### Task 1: Share one immutable artifact bundle across serial test modules

**Files:**
- Modify: `tests/conftest.py:3-33,135-162`
- Modify: `tests/test_distribution_boundaries.py:5-35,80-87`
- Modify: `tests/test_v0_4_public_api.py:37,979-1027`
- Test: `tests/test_distribution_boundaries.py`
- Test: `tests/test_v0_4_public_api.py`

**Interfaces:**
- Consumes: `tests._distribution_support.build_distributions(project_root: Path, output_dir: Path) -> BuiltDistributions` using the helper's default optional artifact arguments.
- Produces: pytest fixture `built_distributions(tmp_path_factory: pytest.TempPathFactory) -> BuiltDistributions`, scoped to one test session.
- Preserves: `test_installed_wheel_preserves_the_complete_public_boundary(tmp_path: Path, built_distributions: BuiltDistributions) -> None` with the existing installation and probe assertions.

- [ ] **Step 1: Measure the focused baseline twice before editing tests**

Run each command separately:

```bash
/usr/bin/time -p -o /private/tmp/vercor-artifact-focused-before-1.time \
  /Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest \
  tests/test_distribution_boundaries.py tests/test_v0_4_public_api.py \
  --durations=15

/usr/bin/time -p -o /private/tmp/vercor-artifact-focused-before-2.time \
  /Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest \
  tests/test_distribution_boundaries.py tests/test_v0_4_public_api.py \
  --durations=15
```

Expected: both runs pass 103 tests. Record both wall times and the duration of `test_installed_wheel_preserves_the_complete_public_boundary`.

- [ ] **Step 2: Write the failing static performance contract**

Add `inspect` to the standard-library imports in `tests/test_distribution_boundaries.py`. Extend the existing public-API import without exposing a second collectable test name:

```python
from tests.test_v0_4_public_api import (
    PUBLIC_MODULE_EXPORTS,
    test_installed_wheel_preserves_the_complete_public_boundary
    as _installed_public_boundary_test,
)
```

Add this test immediately before the existing runtime-metadata test:

```python
@pytest.mark.fast_always
def test_installed_public_boundary_reuses_session_artifacts() -> None:
    signature = inspect.signature(_installed_public_boundary_test)
    assert tuple(signature.parameters) == ("tmp_path", "built_distributions")

    source = inspect.getsource(_installed_public_boundary_test)
    assert "built_distributions.wheel" in source
    assert "_cached_build_pythonpath" not in source
```

- [ ] **Step 3: Run the new contract and verify RED**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest \
  tests/test_distribution_boundaries.py::test_installed_public_boundary_reuses_session_artifacts \
  -v
```

Expected: one failure because the baseline signature contains only `tmp_path` and the source still calls `_cached_build_pythonpath`.

- [ ] **Step 4: Move the artifact fixture to root test configuration**

Add this import in `tests/conftest.py` after the pytest import:

```python
from tests._distribution_support import BuiltDistributions, build_distributions
```

Add the project root beside `_TEST_CACHE_ROOT`:

```python
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
```

Add the session fixture immediately before `fast_mode`:

```python
@pytest.fixture(scope="session")
def built_distributions(
    tmp_path_factory: pytest.TempPathFactory,
) -> BuiltDistributions:
    """Build once or reuse the explicitly supplied immutable artifact bundle."""

    return build_distributions(
        _PROJECT_ROOT,
        tmp_path_factory.mktemp("distribution-build") / "dist",
    )
```

Delete the module-scoped `built_distributions` fixture from `tests/test_distribution_boundaries.py`. Keep its `BuiltDistributions` and `build_distributions` imports because test annotations and direct helper tests still consume them.

- [ ] **Step 5: Make the installed public-boundary test consume the shared wheel**

Replace the private helper import in `tests/test_v0_4_public_api.py`:

```python
from tests._distribution_support import BuiltDistributions
```

Change the beginning of the installed-wheel test to:

```python
def test_installed_wheel_preserves_the_complete_public_boundary(
    tmp_path: Path,
    built_distributions: BuiltDistributions,
) -> None:
    installed_root = tmp_path / "site-packages"
    installed_root.mkdir()
    wheel = built_distributions.wheel
```

Delete only the local `distribution_dir`, build-environment, `python -m build`, and `next(distribution_dir.glob("vercor-*.whl"))` block. Keep the existing pip installation subprocess and every probe assertion unchanged.

- [ ] **Step 6: Run the new contract and verify GREEN**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest \
  tests/test_distribution_boundaries.py::test_installed_public_boundary_reuses_session_artifacts \
  -v
```

Expected: `1 passed`.

- [ ] **Step 7: Run both affected modules in both orders**

Run each command separately:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest \
  tests/test_distribution_boundaries.py tests/test_v0_4_public_api.py

/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest \
  tests/test_v0_4_public_api.py tests/test_distribution_boundaries.py
```

Expected: both runs pass 104 tests. The reverse-order run demonstrates that the session fixture can be initialized by either consumer without order dependence.

- [ ] **Step 8: Measure the focused result twice**

Run the Step 1 commands with output files renamed to `vercor-artifact-focused-after-1.time` and `vercor-artifact-focused-after-2.time`.

Expected: both runs pass 104 tests; the installed public-boundary test no longer contains a wheel-build duration, and the two-run mean is lower despite the new static contract.

- [ ] **Step 9: Format and run focused static checks**

Run each command separately:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m black \
  tests/conftest.py tests/test_distribution_boundaries.py \
  tests/test_v0_4_public_api.py
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m flake8 \
  tests/conftest.py tests/test_distribution_boundaries.py \
  tests/test_v0_4_public_api.py --count --exit-zero \
  --max-line-length=120 --statistics
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m mypy \
  tests/conftest.py tests/test_distribution_boundaries.py \
  tests/test_v0_4_public_api.py
```

Expected: Black exits 0, strict flake8 reports 0, and mypy reports success.

- [ ] **Step 10: Commit the green optimization batch**

```bash
git add tests/conftest.py tests/test_distribution_boundaries.py \
  tests/test_v0_4_public_api.py
git commit -m "test: reuse built distributions across modules"
```

Expected: one commit contains the RED contract and GREEN fixture reuse, with no production-code change.

### Task 2: Verify the complete gate and record measured evidence

**Files:**
- Modify: `PROGRESS.md:5-125`
- Verify: `tests/`, `vercor/`, and `examples/`

**Interfaces:**
- Consumes: the session-scoped `built_distributions` fixture from Task 1.
- Produces: exact final timing and coverage evidence in `PROGRESS.md` and the user handoff.
- Preserves: 1,257 collected tests after the new static performance contract and every baseline coverage floor.

- [ ] **Step 1: Run the deterministic fast gate**

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ --fast
```

Expected: all selected tests pass with no failure, skip, xfail, or error. Record the selected and deselected counts.

- [ ] **Step 2: Run complete serial timing twice**

Run each command separately:

```bash
/usr/bin/time -p -o /private/tmp/vercor-final-serial-1.time \
  /Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ \
  --durations=25

/usr/bin/time -p -o /private/tmp/vercor-final-serial-2.time \
  /Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ \
  --durations=25
```

Expected: both runs pass 1,257 tests. Compute the final mean, absolute saving, and `((125.91 - final_mean) / 125.91) * 100`. Compare the new slowest files and tests with the baseline.

- [ ] **Step 3: Run complete branch coverage**

```bash
/usr/bin/time -p -o /private/tmp/vercor-final-coverage.time \
  /Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ \
  --cov=vercor --cov-branch --cov-report=term:skip-covered \
  --cov-report=json:/private/tmp/vercor-final-coverage.json
```

Expected: 1,257 tests pass, the configured 90% floor passes, and statement/line, branch, and combined coverage are no lower than 93.05%, 78.36%, and 90.52%.

- [ ] **Step 4: Recompute named-function entry coverage**

Run the exact baseline method:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -c "import ast,json; from pathlib import Path; from coverage import CoverageData; report=json.load(open('/private/tmp/vercor-final-coverage.json')); data=CoverageData(); data.read(); total=covered=0; files=report['files']; exec('for path in Path(\"vercor\").rglob(\"*.py\"):\n tree=ast.parse(path.read_text())\n arcs=data.arcs(str(path.resolve())) or []\n entries={-start for start,end in arcs if start < 0}\n item=files.get(str(path))\n executable=set(item[\"executed_lines\"]+item[\"missing_lines\"]) if item else set()\n for node in ast.walk(tree):\n  if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and any(node.lineno <= line <= (node.end_lineno or node.lineno) for line in executable):\n   total+=1\n   first=min([node.lineno]+[decorator.lineno for decorator in node.decorator_list])\n   covered+=first in entries'); print({'covered':covered,'total':total,'percent':100*covered/total})"
```

Expected: at least 671 of 729 named functions/methods entered and at least 92.04%.

- [ ] **Step 5: Verify randomized and parallel execution applicability**

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -c \
  "import importlib.util; print('pytest_randomly', bool(importlib.util.find_spec('pytest_randomly'))); print('xdist', bool(importlib.util.find_spec('xdist')))"
```

Expected: both are `False`. Record randomized-order and parallel comparison as not applicable for this batch; do not install dependencies or claim results that were not run. Order independence is covered by Task 1 Step 7.

- [ ] **Step 6: Run repository-wide formatting, lint, typing, bytecode, and whitespace gates**

Run each command separately:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m black vercor examples tests
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m flake8 . --count \
  --exit-zero --max-line-length=120 --statistics
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m mypy vercor examples tests
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m compileall -q \
  vercor examples tests
git diff --check
```

Expected: Black exits 0, strict flake8 reports 0, mypy succeeds, compileall exits 0, and the whitespace check is clean.

- [ ] **Step 7: Update the bounded progress log with exact results**

Add one concise dated bullet at the top of `## Current Status` recording the baseline and final means, absolute and percentage savings, focused means, fast/full/coverage counts, all four coverage metrics, quality gates, unchanged isolation, lack of production changes, and the remaining bottleneck ranking. Condense existing wrapped prose so `wc -l PROGRESS.md` remains at or below 180. Do not edit either frozen progress archive.

- [ ] **Step 8: Run progress-policy and final focused regressions**

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest \
  tests/test_api_architecture_review.py \
  tests/test_versioning_policy.py \
  tests/test_distribution_boundaries.py \
  tests/test_v0_4_public_api.py
wc -l PROGRESS.md
git diff --check
```

Expected: all focused tests pass, `PROGRESS.md` has at most 180 lines, and the whitespace check is clean.

- [ ] **Step 9: Revert instead of claiming success if the gates fail**

If complete behavior or coverage regresses, restore Task 1's three test files with a new forward patch and rerun Steps 1-8. If focused timing does not remove the duplicate-build cost or the final mean is not lower than 125.91 seconds, restore the original module fixture and inline public-API wheel build with a forward patch, document the failed approach in `PROGRESS.md`, and do not claim an improvement.

- [ ] **Step 10: Commit measured evidence**

```bash
git add PROGRESS.md
git commit -m "docs: record test-suite performance results"
```

Expected: implementation and evidence commits are present, the worktree is clean, and no required validation remains.
