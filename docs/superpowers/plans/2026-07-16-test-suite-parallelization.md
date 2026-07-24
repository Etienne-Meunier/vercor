# Controlled Pytest Parallelization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce complete-suite wall time with a measured, deterministic pytest-xdist default while preserving serial execution, all behavioral assertions, and every baseline coverage metric.

**Architecture:** A private test-only helper records which plotting/cache environment values VerCOR defaulted, then gives each xdist worker distinct writable cache paths while preserving explicit user values. Full-suite benchmarks select two, four, automatic, or no workers; only a demonstrably faster stable candidate becomes the pytest default, and `-n0` remains the serial escape hatch.

**Tech Stack:** Python 3.13, pytest 9.1, pytest-xdist 3.7 or newer, pytest-cov 7.1, coverage.py 7.15, JAX 0.10.2, TOML, GitHub Actions, Black, flake8, and mypy.

## Global Constraints

- Work directly on the current `refactor` branch; do not create a worktree.
- Use `/Users/romannuterman/miniforge3/envs/scipy/bin/python` for every Python command.
- Add only `pytest-xdist>=3.7`, to both `test` and `dev` extras; add no retry, randomization, or unrelated parallelism dependency.
- Preserve all 1,256 baseline tests plus new configuration contracts; add no skip, xfail, retry, quarantine, or weakened assertion.
- Preserve or improve 6,844/7,355 statements (93.05%), 1,202/1,534 branches (78.36%), 90.52% combined coverage, and 671/729 named-function entries (92.04%).
- Use `--dist=loadscope` and `--max-worker-restart=0`; keep `-n0` supported.
- Preserve explicit `MPLBACKEND`, `MPLCONFIGDIR`, and `XDG_CACHE_HOME`; isolate only VerCOR-owned defaults per `gwN` worker.
- Do not change production code, coverage thresholds, public behavior, release tags, remote branches, or published artifacts.
- Keep `PROGRESS.md` at or below 180 lines and record exact measured evidence, including a rejected result if no candidate qualifies.

---

## File structure

- Create `tests/_parallel_support.py`: pure test-environment cache ownership and worker-isolation helper.
- Create `tests/test_parallel_support.py`: focused serial, worker, inheritance, idempotence, and explicit-value contracts.
- Modify `tests/conftest.py`: invoke the helper before test-module imports and retain `_PLOTTING_CACHE_ENV_DEFAULTED` compatibility.
- Modify `tests/test_tools_components_and_plotting.py`: assert exact serial/worker default paths when VerCOR owns them.
- Modify `tests/test_distribution_boundaries.py`: bind dependency metadata, selected pytest defaults, and CI installation to executable contracts.
- Modify `pyproject.toml`: declare pytest-xdist and, only after measurement, configure the accepted default.
- Modify `.github/workflows/python-package.yml`: install pytest-xdist in the installed-artifact lane because that lane reads repository pytest defaults without installing the `dev` extra.
- Modify `PROGRESS.md`: retain compact timing, coverage, stability, and remaining-bottleneck evidence.
- Verify `docs/releasing.md`: existing source commands intentionally inherit the measured default; change it only if an executable release contract fails.

---

### Task 1: Isolate VerCOR-owned test caches per worker

**Files:**
- Create: `tests/_parallel_support.py`
- Create: `tests/test_parallel_support.py`
- Modify: `tests/conftest.py:3-31`
- Modify: `tests/test_tools_components_and_plotting.py:41-53`
- Modify: `tests/test_distribution_boundaries.py:90-117`
- Modify: `pyproject.toml:42-56`

**Interfaces:**
- Consumes: mutable process environment, `Path(tempfile.gettempdir())`, and `os.environ.get("PYTEST_XDIST_WORKER")`.
- Produces: `configure_test_cache_environment(environ: MutableMapping[str, str], *, cache_root: Path, worker_id: str | None) -> dict[str, bool]`.
- Preserves: `_PLOTTING_CACHE_ENV_DEFAULTED: dict[str, bool]` for the existing plotting contract.
- Records ownership in the private inherited environment key `VERCOR_TEST_DEFAULTED_ENV`; values are a comma-separated ordered subset of `MPLBACKEND,MPLCONFIGDIR,XDG_CACHE_HOME`.

- [ ] **Step 1: Write the focused helper contracts before the helper exists**

Create `tests/test_parallel_support.py` with this complete content:

```python
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
    assert environ["MPLCONFIGDIR"] == str(
        tmp_path / "vercor-matplotlib-cache-gw2"
    )
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
    assert environ["MPLCONFIGDIR"] == str(
        tmp_path / "vercor-matplotlib-cache-gw1"
    )
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
```

- [ ] **Step 2: Run the helper contracts and verify RED**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/test_parallel_support.py -v
```

Expected: collection fails with `ModuleNotFoundError: No module named 'tests._parallel_support'`.

- [ ] **Step 3: Extend the dependency metadata contract and verify RED**

Add these exact assertions after the existing pytest-cov assertions in `test_runtime_metadata_separates_test_and_development_dependencies`:

```python
    assert "pytest-xdist>=3.7" in extras["test"]
    assert "pytest-xdist>=3.7" in extras["dev"]
```

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/test_distribution_boundaries.py::test_runtime_metadata_separates_test_and_development_dependencies -v
```

Expected: FAIL because `pytest-xdist>=3.7` is absent from both extras.

- [ ] **Step 4: Implement the pure cache helper**

Create `tests/_parallel_support.py`:

```python
"""Private support for deterministic pytest worker isolation."""

from __future__ import annotations

from collections.abc import MutableMapping
from pathlib import Path

_DEFAULTED_ENV_MARKER = "VERCOR_TEST_DEFAULTED_ENV"
_MANAGED_ENV_KEYS = ("MPLBACKEND", "MPLCONFIGDIR", "XDG_CACHE_HOME")


def configure_test_cache_environment(
    environ: MutableMapping[str, str],
    *,
    cache_root: Path,
    worker_id: str | None,
) -> dict[str, bool]:
    """Set serial or worker-local defaults while preserving explicit values."""

    inherited_marker = environ.get(_DEFAULTED_ENV_MARKER)
    if inherited_marker is None:
        defaulted = {key: key not in environ for key in _MANAGED_ENV_KEYS}
        environ[_DEFAULTED_ENV_MARKER] = ",".join(
            key for key in _MANAGED_ENV_KEYS if defaulted[key]
        )
    else:
        inherited_defaults = set(filter(None, inherited_marker.split(",")))
        defaulted = {key: key in inherited_defaults for key in _MANAGED_ENV_KEYS}

    if defaulted["MPLBACKEND"]:
        environ["MPLBACKEND"] = "Agg"

    worker_suffix = f"-{worker_id}" if worker_id is not None else ""
    path_defaults = {
        "MPLCONFIGDIR": f"vercor-matplotlib-cache{worker_suffix}",
        "XDG_CACHE_HOME": f"vercor-xdg-cache{worker_suffix}",
    }
    for key, directory_name in path_defaults.items():
        if defaulted[key]:
            environ[key] = str(cache_root / directory_name)

    return defaulted
```

- [ ] **Step 5: Route root test configuration through the helper**

Add this import after the third-party imports in `tests/conftest.py`:

```python
from tests._parallel_support import configure_test_cache_environment
```

Replace lines 18-31 with:

```python
_PLOTTING_CACHE_ENV_DEFAULTED = configure_test_cache_environment(
    os.environ,
    cache_root=_TEST_CACHE_ROOT,
    worker_id=os.environ.get("PYTEST_XDIST_WORKER"),
)
```

Do not change fast-selection hooks or fixtures.

- [ ] **Step 6: Strengthen the live cache-path contract**

Replace `test_test_environment_uses_writable_plotting_cache_defaults` in `tests/test_tools_components_and_plotting.py` with:

```python
@pytest.mark.fast_always
def test_test_environment_uses_writable_plotting_cache_defaults() -> None:
    temp_root = Path(tempfile.gettempdir())
    worker_id = os.environ.get("PYTEST_XDIST_WORKER")
    worker_suffix = f"-{worker_id}" if worker_id is not None else ""

    assert "MPLBACKEND" in os.environ
    assert "MPLCONFIGDIR" in os.environ
    assert "XDG_CACHE_HOME" in os.environ
    if conftest_module._PLOTTING_CACHE_ENV_DEFAULTED["MPLBACKEND"]:
        assert os.environ["MPLBACKEND"] == "Agg"
    if conftest_module._PLOTTING_CACHE_ENV_DEFAULTED["MPLCONFIGDIR"]:
        assert Path(os.environ["MPLCONFIGDIR"]) == (
            temp_root / f"vercor-matplotlib-cache{worker_suffix}"
        )
    if conftest_module._PLOTTING_CACHE_ENV_DEFAULTED["XDG_CACHE_HOME"]:
        assert Path(os.environ["XDG_CACHE_HOME"]) == (
            temp_root / f"vercor-xdg-cache{worker_suffix}"
        )
```

- [ ] **Step 7: Declare and install pytest-xdist**

Add the same exact entry after `pytest-cov` in both optional dependency lists:

```toml
  "pytest-xdist>=3.7",
```

Install the declared dependency:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pip install "pytest-xdist>=3.7"
```

If the sandbox blocks package-index access, rerun this exact installation with an escalation request. Then record the installed versions:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -c "import importlib.metadata as m; print('pytest', m.version('pytest')); print('pytest-xdist', m.version('pytest-xdist')); print('pytest-cov', m.version('pytest-cov')); print('coverage', m.version('coverage'))"
```

Expected: pytest-xdist is at least 3.7; do not continue with an older release.

- [ ] **Step 8: Run focused GREEN and isolation repetitions**

Run each command separately:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/test_parallel_support.py -v
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/test_distribution_boundaries.py::test_runtime_metadata_separates_test_and_development_dependencies tests/test_tools_components_and_plotting.py::test_test_environment_uses_writable_plotting_cache_defaults -v
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/test_parallel_support.py tests/test_tools_components_and_plotting.py::test_test_environment_uses_writable_plotting_cache_defaults -n2 --dist=loadscope --max-worker-restart=0
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/test_parallel_support.py tests/test_tools_components_and_plotting.py::test_test_environment_uses_writable_plotting_cache_defaults -n2 --dist=loadscope --max-worker-restart=0
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/test_parallel_support.py tests/test_tools_components_and_plotting.py::test_test_environment_uses_writable_plotting_cache_defaults -n2 --dist=loadscope --max-worker-restart=0
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/test_parallel_support.py tests/test_tools_components_and_plotting.py::test_test_environment_uses_writable_plotting_cache_defaults -n2 --dist=loadscope --max-worker-restart=0
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/test_parallel_support.py tests/test_tools_components_and_plotting.py::test_test_environment_uses_writable_plotting_cache_defaults -n2 --dist=loadscope --max-worker-restart=0
```

Expected: the first two focused commands and all five repetitions pass, with no warning, restart, or cache-path collision.

- [ ] **Step 9: Run focused quality gates**

Run each command separately:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m black tests/_parallel_support.py tests/test_parallel_support.py tests/conftest.py tests/test_tools_components_and_plotting.py tests/test_distribution_boundaries.py
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m flake8 tests/_parallel_support.py tests/test_parallel_support.py tests/conftest.py tests/test_tools_components_and_plotting.py tests/test_distribution_boundaries.py --count --max-line-length=120 --statistics
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m mypy tests/_parallel_support.py tests/test_parallel_support.py tests/conftest.py tests/test_tools_components_and_plotting.py tests/test_distribution_boundaries.py
git diff --check
```

Expected: Black leaves formatted files, flake8 reports 0, mypy reports no issues, and the diff check is clean.

- [ ] **Step 10: Commit the independently reviewable isolation batch**

```bash
git add pyproject.toml tests/_parallel_support.py tests/test_parallel_support.py tests/conftest.py tests/test_tools_components_and_plotting.py tests/test_distribution_boundaries.py
git commit -m "test: isolate caches for pytest workers"
```

---

### Task 2: Benchmark worker counts and configure only a proven winner

**Files:**
- Modify on acceptance: `tests/test_distribution_boundaries.py:90-117,242-314,335-366`
- Modify on acceptance: `pyproject.toml:62-67`
- Modify on acceptance: `.github/workflows/python-package.yml:81-102`
- Modify on rejection: `PROGRESS.md`

**Interfaces:**
- Consumes: Task 1 worker isolation, pytest-xdist CLI, baseline count 1,256, and `/private/tmp` timing/log files.
- Produces on acceptance: exactly one default among `-n2`, `-n4`, or `-n auto`, always paired with `--dist=loadscope --max-worker-restart=0`.
- Produces on rejection: original `addopts = "-q"`, no xdist helper/dependency/configuration changes, and durable failed-approach evidence.

- [ ] **Step 1: Measure two contemporaneous serial controls**

Run each command separately:

```bash
/usr/bin/time -p -o /private/tmp/vercor-xdist-serial-1.time /Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -n0 --durations=25 --tb=short > /private/tmp/vercor-xdist-serial-1.log 2>&1
/usr/bin/time -p -o /private/tmp/vercor-xdist-serial-2.time /Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -n0 --durations=25 --tb=short > /private/tmp/vercor-xdist-serial-2.log 2>&1
```

Expected: both runs pass the original 1,256 tests plus Task 1's 5 new helper tests, with identical warnings and no skips or xfails. Record wall, user, and system time from both `.time` files.

- [ ] **Step 2: Measure two complete runs at each candidate worker count**

Run each command separately:

```bash
/usr/bin/time -p -o /private/tmp/vercor-xdist-n2-1.time /Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -n2 --dist=loadscope --max-worker-restart=0 --durations=25 --tb=short > /private/tmp/vercor-xdist-n2-1.log 2>&1
/usr/bin/time -p -o /private/tmp/vercor-xdist-n2-2.time /Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -n2 --dist=loadscope --max-worker-restart=0 --durations=25 --tb=short > /private/tmp/vercor-xdist-n2-2.log 2>&1
/usr/bin/time -p -o /private/tmp/vercor-xdist-n4-1.time /Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -n4 --dist=loadscope --max-worker-restart=0 --durations=25 --tb=short > /private/tmp/vercor-xdist-n4-1.log 2>&1
/usr/bin/time -p -o /private/tmp/vercor-xdist-n4-2.time /Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -n4 --dist=loadscope --max-worker-restart=0 --durations=25 --tb=short > /private/tmp/vercor-xdist-n4-2.log 2>&1
/usr/bin/time -p -o /private/tmp/vercor-xdist-auto-1.time /Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -n auto --dist=loadscope --max-worker-restart=0 --durations=25 --tb=short > /private/tmp/vercor-xdist-auto-1.log 2>&1
/usr/bin/time -p -o /private/tmp/vercor-xdist-auto-2.time /Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -n auto --dist=loadscope --max-worker-restart=0 --durations=25 --tb=short > /private/tmp/vercor-xdist-auto-2.log 2>&1
```

Expected for an eligible candidate: both runs pass exactly the same count as serial; no skips, xfails, retries, worker restarts, crashes, hangs, new warnings, or cleanup failures.

- [ ] **Step 3: Audit raw benchmark evidence and apply the decision rule**

Inspect concise results:

```bash
sed -n '1,4p' /private/tmp/vercor-xdist-serial-1.time /private/tmp/vercor-xdist-serial-2.time /private/tmp/vercor-xdist-n2-1.time /private/tmp/vercor-xdist-n2-2.time /private/tmp/vercor-xdist-n4-1.time /private/tmp/vercor-xdist-n4-2.time /private/tmp/vercor-xdist-auto-1.time /private/tmp/vercor-xdist-auto-2.time
rg -n "passed|failed|skipped|xfailed|warnings summary|worker|crash|restart|INTERNALERROR" /private/tmp/vercor-xdist-serial-*.log /private/tmp/vercor-xdist-n2-*.log /private/tmp/vercor-xdist-n4-*.log /private/tmp/vercor-xdist-auto-*.log
```

Apply these exact rules:

1. Disqualify any candidate with a count mismatch, failure, skip, xfail, retry, restart, crash, hang, cleanup error, or new warning.
2. Among eligible candidates, choose the lowest two-run wall-time mean.
3. When candidate means differ by no more than 2% of the faster mean, choose the smaller fixed worker count; rank `auto` after fixed counts in such a tie.
4. Treat the winner as provisional only when its saving over the serial mean exceeds the absolute difference between the two serial wall times. Otherwise execute the rejection path in Step 9.

- [ ] **Step 4: Validate the provisional candidate a third time, without loadscope reorder, and against a third serial control**

Execute exactly one of the first three commands, matching the provisional winner. Then execute the corresponding reorder command and the serial command.

Two-worker candidate:

```bash
/usr/bin/time -p -o /private/tmp/vercor-xdist-n2-3.time /Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -n2 --dist=loadscope --max-worker-restart=0 --durations=25 --tb=short > /private/tmp/vercor-xdist-n2-3.log 2>&1
/usr/bin/time -p -o /private/tmp/vercor-xdist-n2-no-reorder.time /Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -n2 --dist=loadscope --no-loadscope-reorder --max-worker-restart=0 --durations=25 --tb=short > /private/tmp/vercor-xdist-n2-no-reorder.log 2>&1
```

Four-worker candidate:

```bash
/usr/bin/time -p -o /private/tmp/vercor-xdist-n4-3.time /Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -n4 --dist=loadscope --max-worker-restart=0 --durations=25 --tb=short > /private/tmp/vercor-xdist-n4-3.log 2>&1
/usr/bin/time -p -o /private/tmp/vercor-xdist-n4-no-reorder.time /Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -n4 --dist=loadscope --no-loadscope-reorder --max-worker-restart=0 --durations=25 --tb=short > /private/tmp/vercor-xdist-n4-no-reorder.log 2>&1
```

Automatic-worker candidate:

```bash
/usr/bin/time -p -o /private/tmp/vercor-xdist-auto-3.time /Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -n auto --dist=loadscope --max-worker-restart=0 --durations=25 --tb=short > /private/tmp/vercor-xdist-auto-3.log 2>&1
/usr/bin/time -p -o /private/tmp/vercor-xdist-auto-no-reorder.time /Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -n auto --dist=loadscope --no-loadscope-reorder --max-worker-restart=0 --durations=25 --tb=short > /private/tmp/vercor-xdist-auto-no-reorder.log 2>&1
```

Third serial control:

```bash
/usr/bin/time -p -o /private/tmp/vercor-xdist-serial-3.time /Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -n0 --durations=25 --tb=short > /private/tmp/vercor-xdist-serial-3.log 2>&1
```

Accept only if the two validation runs and third serial run have equivalent results, the three normal candidate runs have a lower mean than the three serial runs, and the saving exceeds the larger of the serial and candidate wall-time ranges. Otherwise execute Step 9.

- [ ] **Step 5: Write the selected-default and CI dependency contracts before configuration**

Add this test after the runtime metadata test, selecting exactly one expected tuple from the three explicit alternatives below:

```python
@pytest.mark.fast_always
def test_pytest_defaults_use_measured_parallel_policy() -> None:
    metadata = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert shlex.split(metadata["tool"]["pytest"]["ini_options"]["addopts"]) == [
        "-q",
        "-n2",
        "--dist=loadscope",
        "--max-worker-restart=0",
    ]
```

For a four-worker winner, the complete expected list is:

```python
    assert shlex.split(metadata["tool"]["pytest"]["ini_options"]["addopts"]) == [
        "-q",
        "-n4",
        "--dist=loadscope",
        "--max-worker-restart=0",
    ]
```

For an automatic-worker winner, the complete expected list is:

```python
    assert shlex.split(metadata["tool"]["pytest"]["ini_options"]["addopts"]) == [
        "-q",
        "-n",
        "auto",
        "--dist=loadscope",
        "--max-worker-restart=0",
    ]
```

In `test_ci_validates_installed_artifacts_across_supported_environments`, add:

```python
    assert "pytest-xdist>=3.7" in installed_tools
```

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/test_distribution_boundaries.py::test_pytest_defaults_use_measured_parallel_policy tests/test_distribution_boundaries.py::test_ci_validates_installed_artifacts_across_supported_environments -n0 -v
```

Expected: both tests fail because `addopts` is still `-q` and the installed-artifact lane does not install pytest-xdist.

- [ ] **Step 6: Configure the measured default and CI support**

For a two-worker winner, set:

```toml
addopts = "-q -n2 --dist=loadscope --max-worker-restart=0"
```

For a four-worker winner, set:

```toml
addopts = "-q -n4 --dist=loadscope --max-worker-restart=0"
```

For an automatic-worker winner, set:

```toml
addopts = "-q -n auto --dist=loadscope --max-worker-restart=0"
```

In `.github/workflows/python-package.yml`, replace the installed-artifact tool-install line with:

```yaml
          python -m pip install --upgrade pip pytest "pytest-xdist>=3.7" mypy pyyaml build "flit_core<4"
```

Do not add explicit xdist flags to workflow commands: source quality and coverage commands inherit the measured default, while `-n0` remains available for diagnosis.

- [ ] **Step 7: Verify GREEN, the serial escape hatch, and CI/release contracts**

Run each command separately:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/test_distribution_boundaries.py::test_pytest_defaults_use_measured_parallel_policy tests/test_distribution_boundaries.py::test_ci_validates_installed_artifacts_across_supported_environments tests/test_distribution_boundaries.py::test_ci_quality_job_enforces_static_full_and_coverage_gates tests/test_api_architecture_review.py::test_release_files_and_metadata_describe_the_built_alpha -n0 -v
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/test_parallel_support.py tests/test_tools_components_and_plotting.py::test_test_environment_uses_writable_plotting_cache_defaults
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/test_parallel_support.py tests/test_tools_components_and_plotting.py::test_test_environment_uses_writable_plotting_cache_defaults -n0
```

Expected: 4 contract tests pass serially; focused cache tests pass with the measured default and with `-n0`. Leave `docs/releasing.md` unchanged when its executable contract passes.

- [ ] **Step 8: Format, statically validate, and commit the accepted configuration**

Run each command separately:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m black tests/test_distribution_boundaries.py
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m flake8 tests/test_distribution_boundaries.py --count --max-line-length=120 --statistics
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m mypy tests/test_distribution_boundaries.py
git diff --check
```

Then commit:

```bash
git add pyproject.toml .github/workflows/python-package.yml tests/test_distribution_boundaries.py
git commit -m "test: enable measured pytest parallelism"
```

- [ ] **Step 9: Execute this forward-revert path instead if no candidate qualifies**

Because no repository commit occurs during Steps 1-4, Task 1 is still `HEAD`. Restore the pre-experiment runtime state with:

```bash
git revert --no-edit HEAD
```

Add a compact `PROGRESS.md` entry that states the three serial wall times, every candidate wall time, the rejection rule that failed, unchanged baseline coverage, and that no parallel default or speedup is retained. Verify and commit it:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/test_api_architecture_review.py::test_active_memory_is_current_and_historical_detail_is_archived -v
git diff --check
git add PROGRESS.md
git commit -m "docs: record rejected pytest parallelization"
```

Stop this plan after the rejection commit. Do not run Task 3 or claim any performance improvement.

---

### Task 3: Prove coverage, determinism, quality, and final performance

**Files:**
- Modify: `PROGRESS.md`
- Verify: `tests/`, `vercor/`, `examples/`, `.github/workflows/python-package.yml`, `docs/releasing.md`

**Interfaces:**
- Consumes: accepted Task 2 default, `-n0`, coverage JSON, coverage arc data, and all `/private/tmp/vercor-xdist-*` evidence.
- Produces: final serial/parallel coverage equivalence, final repeated timing comparison, exact percentage improvement, and a clean committed branch.

- [ ] **Step 1: Run fast gates in default and serial modes**

Run each command separately:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ --fast --tb=short
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ --fast -n0 --tb=short
```

Expected: identical selected/deselected counts and all selected tests pass in both modes.

- [ ] **Step 2: Measure complete serial coverage**

Run:

```bash
env COVERAGE_FILE=/private/tmp/.coverage-vercor-serial /usr/bin/time -p -o /private/tmp/vercor-xdist-coverage-serial.time /Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -n0 --cov=vercor --cov-branch --cov-report=term-missing --cov-report=json:/private/tmp/vercor-xdist-coverage-serial.json --cov-fail-under=90 --tb=short > /private/tmp/vercor-xdist-coverage-serial.log 2>&1
```

Expected: all tests pass; statement, branch, and combined totals meet or exceed the baseline.

- [ ] **Step 3: Measure complete parallel coverage**

Run:

```bash
env COVERAGE_FILE=/private/tmp/.coverage-vercor-parallel /usr/bin/time -p -o /private/tmp/vercor-xdist-coverage-parallel.time /Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ --cov=vercor --cov-branch --cov-report=term-missing --cov-report=json:/private/tmp/vercor-xdist-coverage-parallel.json --cov-fail-under=90 --tb=short > /private/tmp/vercor-xdist-coverage-parallel.log 2>&1
```

Expected: the same test count passes and pytest-cov combines all worker data without warnings or missing files.

- [ ] **Step 4: Compare exact coverage totals and function-entry proxies**

Print serial and parallel JSON totals:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -c "import json; print('serial', json.load(open('/private/tmp/vercor-xdist-coverage-serial.json'))['totals']); print('parallel', json.load(open('/private/tmp/vercor-xdist-coverage-parallel.json'))['totals'])"
```

Run the named-function proxy against serial data:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -c "import ast,json; from pathlib import Path; from coverage import CoverageData; report=json.load(open('/private/tmp/vercor-xdist-coverage-serial.json')); data=CoverageData(basename='/private/tmp/.coverage-vercor-serial'); data.read(); total=covered=0; files=report['files']; exec('for path in Path(\"vercor\").rglob(\"*.py\"):\n tree=ast.parse(path.read_text())\n arcs=data.arcs(str(path.resolve())) or []\n entries={-start for start,end in arcs if start < 0}\n item=files.get(str(path))\n executable=set(item[\"executed_lines\"]+item[\"missing_lines\"]) if item else set()\n for node in ast.walk(tree):\n  if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and any(node.lineno <= line <= (node.end_lineno or node.lineno) for line in executable):\n   total+=1\n   first=min([node.lineno]+[decorator.lineno for decorator in node.decorator_list])\n   covered+=first in entries'); print({'covered':covered,'total':total,'percent':100*covered/total})"
```

Run the same proxy against parallel data:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -c "import ast,json; from pathlib import Path; from coverage import CoverageData; report=json.load(open('/private/tmp/vercor-xdist-coverage-parallel.json')); data=CoverageData(basename='/private/tmp/.coverage-vercor-parallel'); data.read(); total=covered=0; files=report['files']; exec('for path in Path(\"vercor\").rglob(\"*.py\"):\n tree=ast.parse(path.read_text())\n arcs=data.arcs(str(path.resolve())) or []\n entries={-start for start,end in arcs if start < 0}\n item=files.get(str(path))\n executable=set(item[\"executed_lines\"]+item[\"missing_lines\"]) if item else set()\n for node in ast.walk(tree):\n  if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and any(node.lineno <= line <= (node.end_lineno or node.lineno) for line in executable):\n   total+=1\n   first=min([node.lineno]+[decorator.lineno for decorator in node.decorator_list])\n   covered+=first in entries'); print({'covered':covered,'total':total,'percent':100*covered/total})"
```

Acceptance: serial and parallel statement, branch, combined, and function-entry values are identical to one another and no lower than the baseline values in Global Constraints. A decrease triggers a forward revert of Tasks 1-2 and a rejected-result `PROGRESS.md` entry.

- [ ] **Step 5: Run repository-wide static and formatting gates**

Run each command separately:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m black --check vercor examples tests
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m flake8 . --count --max-line-length=120 --statistics
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m mypy vercor examples tests
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m compileall -q vercor examples tests
git diff --check
```

Expected: Black passes without edits, flake8 reports 0, mypy reports no issues, compileall succeeds, and the diff check is clean.

- [ ] **Step 6: Update durable progress evidence without exceeding 180 lines**

Replace the current 13-line `Rejected test-artifact optimization (2026-07-16)` bullet at `PROGRESS.md:9-21` with one dated controlled-parallelization bullet. Retain the prior experiment in one sentence: artifact reuse was forward-reverted because 29.975 seconds regressed from 29.215 seconds. In the same replacement bullet, record all of these measured facts from the named `/private/tmp` files:

- Python, pytest, pytest-xdist, pytest-cov, coverage.py, JAX, platform, and architecture versions.
- Baseline 1,256-test serial mean of 125.91 seconds and the three contemporaneous serial wall times and mean after Task 1's 5 new helper tests.
- Every initial worker-count timing, the selected worker count, the three-run selected mean, the no-reorder result, absolute seconds saved, and `((serial mean - parallel mean) / serial mean) * 100` percentage improvement.
- Final total test count and zero failed/skipped/xfailed/retried/restarted/crashed/flaky tests.
- Serial and parallel statement, branch, combined, and named-function coverage totals.
- Black, flake8, mypy, compileall, fast default/serial, complete default/serial, coverage, and diff-check outcomes.
- Remaining bottlenecks: distribution builds, JAX/coupler runtime, setup subprocesses, flux tests, and public-API subprocess probes.
- Explicit statement that production behavior, assertions, test selection, coverage thresholds, releases, pushes, and publications did not change.

Run the progress size/contract check:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/test_api_architecture_review.py::test_active_memory_is_current_and_historical_detail_is_archived -n0 -v
```

Expected: PASS and `wc -l PROGRESS.md` is at most 180.

- [ ] **Step 7: Run final complete default and serial gates after documentation**

Run each command separately:

```bash
/usr/bin/time -p -o /private/tmp/vercor-xdist-final-parallel.time /Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ --durations=25 --tb=short > /private/tmp/vercor-xdist-final-parallel.log 2>&1
/usr/bin/time -p -o /private/tmp/vercor-xdist-final-serial.time /Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -n0 --durations=25 --tb=short > /private/tmp/vercor-xdist-final-serial.log 2>&1
```

Expected: identical complete counts pass; no skip, xfail, retry, restart, crash, new warning, or cleanup failure. These are correctness confirmations; report the Task 2 three-run means, not a single favorable final sample, as the performance comparison.

- [ ] **Step 8: Inspect cleanup and commit final evidence**

Run:

```bash
git status --short
git diff --check
```

Confirm no tracked cache, coverage, compiled, temporary, or generated artifact entered the worktree. Then commit only the progress evidence:

```bash
git add PROGRESS.md
git commit -m "docs: record pytest parallelization results"
```

- [ ] **Step 9: Request independent final review before claiming completion**

Use `superpowers:requesting-code-review` with the approved design, this plan, all commits from Task 1 onward, benchmark logs, serial/parallel coverage JSON, and final quality outputs. Resolve every Critical or Important finding with a new RED/GREEN cycle and rerun affected plus complete gates before the final report.

The final report must state baseline runtime, contemporaneous serial mean, selected parallel mean, absolute saving, percentage improvement formula/result, baseline/final coverage, serial/parallel equivalence, exact new tests, major remaining bottlenecks, deferred higher-risk work, and the final commit hashes.
