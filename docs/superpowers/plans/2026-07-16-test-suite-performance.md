# Test-suite Artifact Reuse Implementation Plan

> **Execution record:** All steps now have a checked disposition. A checked
> "NOT RUN" item records that the failure path made the command inapplicable;
> it does not claim that command ran.

**Status:** Closed after the planned failure path; experiment forward-reverted.

**Historical goal:** Remove the duplicate VerCOR wheel build from the complete pytest gate while preserving every build, installation, public-boundary, isolation, and coverage guarantee.

**Rejected experimental architecture:** A session-scoped root fixture built or resolved one frozen `BuiltDistributions` bundle. Distribution tests and the installed public-API boundary shared only those immutable artifact paths; every installation directory and interpreter probe remained test-local. Commit `242bbe7` restored independent builds.

**Tech Stack:** Python 3.13, pytest 9.1, subprocess-based offline builds and installs, pytest-cov/coverage.py, Black, flake8, and mypy.

## Execution outcome

Task 1 was performed in full and committed as `20ac416`. Its RED contract
failed 1/1 and then passed 1/1; both module orders passed 104/104. The original
focused pair passed 103/103 at a 29.215s wall-time mean, while the attempted
pair passed 104/104 at 29.975s. The required aggregate timing result therefore
failed even though the targeted installed-boundary call decreased.

Task 2 followed Step 9 rather than the success path. Commit `242bbe7`
forward-reverted the three test files to the exact `0d86341` state; the
post-revert artifact/public gate passed 103/103. The full-suite timing,
coverage, function-entry coverage, randomized/parallel applicability, and
repository-wide quality steps were not run for a retained optimization.
Failure evidence and policy verification were recorded in `30086f4` and
`16fd230`: the four-file focused gate passed 264/264, and the follow-up
policy/architecture gate passed 161/161. No speedup, production change, or test
optimization is retained.

## Original constraints

- Optimize the complete `pytest tests/` gate, including artifact, optional-dependency, JAX transformation, and numerical contracts.
- Do not delete, skip, disable, quarantine, xfail, merge, or weaken tests or assertions.
- Preserve at least 93.05% statement/line coverage, 78.36% branch coverage, 90.52% combined branch-aware coverage, and 92.04% named-function entry coverage.
- Preserve fresh installation directories and interpreter isolation for installed-distribution probes.
- Do not introduce shared mutable state, test-order dependence, pytest-xdist, or production-code changes in this batch.
- Use the direct `/Users/romannuterman/miniforge3/envs/scipy/bin/python` interpreter because the Conda launcher can panic before pytest.
- Keep `PROGRESS.md` at or below its executable 180-line limit.
- Do not tag, push, publish, upload, or create a release.

---

## Historical file scope and restored structure

- `tests/conftest.py`: temporarily owned the session-scoped fixture; the revert
  removed it.
- `tests/test_distribution_boundaries.py`: temporarily consumed the root
  fixture and held the static reuse contract; the module-scoped fixture was
  restored and that experimental contract was removed.
- `tests/test_v0_4_public_api.py`: temporarily consumed the shared wheel; its
  independent inline wheel build was restored.
- `PROGRESS.md`: records the failed approach and focused rejection evidence.

### Task 1: Share one immutable artifact bundle across serial test modules

**Files:**
- Modify: `tests/conftest.py:3-33,135-162`
- Modify: `tests/test_distribution_boundaries.py:5-35,80-87`
- Modify: `tests/test_v0_4_public_api.py:37,979-1027`
- Test: `tests/test_distribution_boundaries.py`
- Test: `tests/test_v0_4_public_api.py`

**Historical experimental interfaces:**
- Consumes: `tests._distribution_support.build_distributions(project_root: Path, output_dir: Path) -> BuiltDistributions` using the helper's default optional artifact arguments.
- Produces: pytest fixture `built_distributions(tmp_path_factory: pytest.TempPathFactory) -> BuiltDistributions`, scoped to one test session.
- Preserves: `test_installed_wheel_preserves_the_complete_public_boundary(tmp_path: Path, built_distributions: BuiltDistributions) -> None` with the existing installation and probe assertions.

- [x] **Step 1: Measured the focused baseline twice before editing tests**

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

Result: both runs passed 103/103. Wall times were 29.83s and 28.60s
(29.215s mean); the installed-boundary call was 1.28s in both samples.

- [x] **Step 2: Wrote the failing static performance contract**

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

- [x] **Step 3: Ran the new contract and verified RED**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest \
  tests/test_distribution_boundaries.py::test_installed_public_boundary_reuses_session_artifacts \
  -v
```

Result: 1/1 failed at the intended signature assertion because the baseline
contained only `tmp_path`.

- [x] **Step 4: Moved the artifact fixture to root test configuration**

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

- [x] **Step 5: Made the installed public-boundary test consume the shared wheel**

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

- [x] **Step 6: Ran the new contract and verified GREEN**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest \
  tests/test_distribution_boundaries.py::test_installed_public_boundary_reuses_session_artifacts \
  -v
```

Result: 1/1 passed.

- [x] **Step 7: Ran both affected modules in both orders**

Run each command separately:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest \
  tests/test_distribution_boundaries.py tests/test_v0_4_public_api.py

/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest \
  tests/test_v0_4_public_api.py tests/test_distribution_boundaries.py
```

Result: both orders passed 104/104 (31.39s and 29.65s).

- [x] **Step 8: Measured the focused result twice — acceptance failed**

Run the Step 1 commands with output files renamed to `vercor-artifact-focused-after-1.time` and `vercor-artifact-focused-after-2.time`.

Result: both runs passed 104/104, but wall times were 29.73s and 30.22s
(29.975s mean), 0.760s above the 29.215s baseline mean. The target-call mean
fell from 1.28s to 1.04s, which did not satisfy the aggregate timing gate.

- [x] **Step 9: Formatted and ran focused static checks**

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

Result: Black exited 0, the final flake8 pass reported 0, mypy found no issues
in 3 source files, and `git diff --check` was clean.

- [x] **Step 10: Committed the experimental optimization batch**

```bash
git add tests/conftest.py tests/test_distribution_boundaries.py \
  tests/test_v0_4_public_api.py
git commit -m "test: reuse built distributions across modules"
```

Result: `20ac416` contained the RED contract and experimental fixture reuse,
with no production-code change. It was later forward-reverted by `242bbe7`.

### Task 2: Verify the complete gate and record measured evidence

**Files:**
- Modify: `PROGRESS.md:5-125`
- Verify: `tests/`, `vercor/`, and `examples/`

**Execution-path interfaces:**
- The attempted success path would have consumed Task 1's session-scoped
  fixture and produced full timing and coverage evidence.
- The actual failure path restored independent builds, removed the additional
  static contract, and produced only focused rejection and policy evidence.

- [x] **Step 1 disposition: NOT RUN because Step 9's forward-revert path triggered**

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ --fast
```

Result: not run for a retained optimization; no fast-gate count is claimed.

- [x] **Step 2 disposition: NOT RUN because Step 9's forward-revert path triggered**

Run each command separately:

```bash
/usr/bin/time -p -o /private/tmp/vercor-final-serial-1.time \
  /Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ \
  --durations=25

/usr/bin/time -p -o /private/tmp/vercor-final-serial-2.time \
  /Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ \
  --durations=25
```

Result: not run; no full-suite timing or percentage improvement is claimed.

- [x] **Step 3 disposition: NOT RUN because Step 9's forward-revert path triggered**

```bash
/usr/bin/time -p -o /private/tmp/vercor-final-coverage.time \
  /Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ \
  --cov=vercor --cov-branch --cov-report=term:skip-covered \
  --cov-report=json:/private/tmp/vercor-final-coverage.json
```

Result: not run; no post-experiment coverage result is claimed.

- [x] **Step 4 disposition: NOT RUN because Step 9's forward-revert path triggered**

Run the exact baseline method:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -c "import ast,json; from pathlib import Path; from coverage import CoverageData; report=json.load(open('/private/tmp/vercor-final-coverage.json')); data=CoverageData(); data.read(); total=covered=0; files=report['files']; exec('for path in Path(\"vercor\").rglob(\"*.py\"):\n tree=ast.parse(path.read_text())\n arcs=data.arcs(str(path.resolve())) or []\n entries={-start for start,end in arcs if start < 0}\n item=files.get(str(path))\n executable=set(item[\"executed_lines\"]+item[\"missing_lines\"]) if item else set()\n for node in ast.walk(tree):\n  if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and any(node.lineno <= line <= (node.end_lineno or node.lineno) for line in executable):\n   total+=1\n   first=min([node.lineno]+[decorator.lineno for decorator in node.decorator_list])\n   covered+=first in entries'); print({'covered':covered,'total':total,'percent':100*covered/total})"
```

Result: not run; no post-experiment function-entry result is claimed.

- [x] **Step 5 disposition: NOT RUN because Step 9's forward-revert path triggered**

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -c \
  "import importlib.util; print('pytest_randomly', bool(importlib.util.find_spec('pytest_randomly'))); print('xdist', bool(importlib.util.find_spec('xdist')))"
```

Result: not run. No randomized-order or parallel-applicability result is
claimed; Task 1 Step 7 only established the two explicit module orders.

- [x] **Step 6 disposition: NOT RUN because Step 9's forward-revert path triggered**

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

Result: the repository-wide success-path gates were not run. Task 1's focused
three-file Black, flake8, mypy, and whitespace checks are recorded separately.

- [x] **Step 7: Updated the bounded progress log with failure evidence**

Result: `PROGRESS.md` records the focused and alternating rejection evidence,
the absence of retained changes, and the forward revert without claiming
unrun success-path gates. Commits: `30086f4` and clarification `16fd230`.

- [x] **Step 8: Ran progress-policy and final focused regressions**

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest \
  tests/test_api_architecture_review.py \
  tests/test_versioning_policy.py \
  tests/test_distribution_boundaries.py \
  tests/test_v0_4_public_api.py
wc -l PROGRESS.md
git diff --check
```

Result: the four-file gate passed 264/264. The follow-up policy/architecture
gate passed 161/161. `PROGRESS.md` remained within 180 lines, and the then-current
working diff passed `git diff --check`.

- [x] **Step 9: Forward-reverted after the aggregate timing gate failed**

Result: commit `242bbe7` restored the three test files exactly to `0d86341`.
The post-revert distribution/public gate passed 103/103. No improvement is
claimed.

- [x] **Step 10: Committed failure evidence and policy clarification**

```bash
git add PROGRESS.md
git commit -m "docs: record test-suite performance results"
```

Result: evidence commits `30086f4` and `16fd230` are present. The experimental
commit `20ac416` and forward-revert commit `242bbe7` preserve the complete
historical execution record.
