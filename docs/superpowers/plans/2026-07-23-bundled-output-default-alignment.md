# Bundled Output Default Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make omitted slab/data output declarations resolve to `OutputSpec()` while preserving explicit per-component policies and run-level final-field output.

**Architecture:** Replace the step-specific setup helper with one optional-`OutputSpec` resolver shared by slab factories, the data helper, and direct JCM land. Match external configuration defaults by making paired JCM land use `field(default_factory=OutputSpec)`. Leave runtime providers, accumulators, writers, and public signatures unchanged.

**Tech Stack:** Python 3.13, JAX, frozen dataclasses, h5netcdf, pytest, Black, flake8, mypy, coverage.py, Git.

## Global Constraints

- Keep every public `output: OutputSpec | None = None` signature unchanged.
- Omitted output and explicit `None` resolve to `OutputSpec()` with `period=None`.
- Preserve supplied `OutputSpec` objects by identity.
- Keep external native providers and snapshot writers unchanged.
- Keep the paired JCM atmosphere's explicit monthly default unchanged.
- Change paired JCM land's omitted default to `OutputSpec()`.
- Keep `OutputTarget.write_final_fields` independent from period declarations.
- Do not change component kernels, runtime execution, cadence calculations, providers, accumulators, filenames, or NetCDF writers.
- Use TDD and update `README.md`, `DESIGN.md`, `CHANGELOG.md`, `DEPENDENCIES.md`, the API review, and `PROGRESS.md`.
- Run Black, strict flake8, mypy, compileall, focused, fast, full, branch-coverage, installed-artifact, and whitespace checks before completion.

---

## File Structure

- Modify `tests/test_bundled_period_output.py`: default-none, explicit cadence, no-period, and final-field behavior.
- Modify `tests/test_setup_agnostic_api.py`: paired JCM land default and explicit forwarding.
- Modify `tests/test_setup_lifecycle_helpers.py`: default land-output expectations.
- Modify `vercor/setups/_output.py`: generic optional declaration resolver.
- Modify four files under `vercor/setups/_slab/`: use the renamed resolver.
- Modify `vercor/setups/_data/_component_helpers.py`: use the renamed resolver.
- Modify `vercor/setups/_data/jcm_land.py`: use the renamed resolver.
- Modify `vercor/setups/config.py`: use `OutputSpec` as the JCM-land default factory.
- Modify docs and progress files: remove superseded step-default claims and record evidence.

### Task 1: Specify aligned omitted-output behavior

**Files:**
- Modify: `tests/test_bundled_period_output.py`
- Modify: `tests/test_setup_lifecycle_helpers.py`
- Test: `tests/test_bundled_period_output.py`
- Test: `tests/test_setup_lifecycle_helpers.py`

**Interfaces:**
- Consumes: existing public slab/data/JCM factories and `OutputTarget`.
- Produces: failing acceptance tests for `period=None`, no default period files, and preserved final fields.

- [ ] **Step 1: Change slab/data/direct-JCM default assertions**

Replace `_assert_step_period_output` with:

```python
def _assert_output_disabled_by_default(component: Any) -> None:
    assert component.spec.output == OutputSpec()
```

Rename and update the default tests so all slab factories, the shared data
helper, and direct JCM land assert `OutputSpec()`.

- [ ] **Step 2: Add public data-factory default coverage**

Extend the existing public data parameterization:

```python
@pytest.mark.parametrize("factory", _DATA_FACTORIES)
def test_public_data_factory_omitted_output_is_disabled(factory: Any) -> None:
    component = factory()
    assert component.spec.output == OutputSpec()
```

- [ ] **Step 3: Replace the redundant explicit-disable regression**

Change the slab no-period test to omit `output`:

```python
component = make_slab_atmosphere(grid)
```

Assert no `*.averages.*.nc` file is created when the target enables period
output but disables final fields and snapshots.

- [ ] **Step 4: Add final-field independence regression**

Run a default slab component with:

```python
OutputTarget(
    tmp_path,
    write_period=True,
    write_final_fields=True,
    write_snapshots=False,
)
```

Assert no averages file exists and `atm.runtime_fields.nc` exists.

- [ ] **Step 5: Change paired-JCM land default expectations**

In strict paired-JCM lifecycle tests, replace:

```python
OutputSpec(period=PeriodOutput(frequency="step"))
```

with:

```python
OutputSpec()
```

Keep the existing explicit `land_output` forwarding test unchanged.

- [ ] **Step 6: Run tests and verify RED**

Run:

```bash
env CONDA_NO_PLUGINS=true conda run -n scipy pytest \
  tests/test_bundled_period_output.py \
  tests/test_setup_lifecycle_helpers.py \
  -q --tb=short
```

Expected: default slab/data/JCM assertions and paired-JCM expectations fail
because production still injects `PeriodOutput(frequency="step")`.

### Task 2: Implement the generic default resolver

**Files:**
- Modify: `vercor/setups/_output.py`
- Modify: `vercor/setups/_slab/atmosphere.py`
- Modify: `vercor/setups/_slab/land.py`
- Modify: `vercor/setups/_slab/ocean.py`
- Modify: `vercor/setups/_slab/seaice.py`
- Modify: `vercor/setups/_data/_component_helpers.py`
- Modify: `vercor/setups/_data/jcm_land.py`
- Modify: `vercor/setups/config.py`
- Test: `tests/test_bundled_period_output.py`
- Test: `tests/test_setup_lifecycle_helpers.py`
- Test: `tests/test_setup_agnostic_api.py`

**Interfaces:**
- Consumes: `OutputSpec | None`.
- Produces: `resolve_output(output: OutputSpec | None = None) -> OutputSpec`.

- [ ] **Step 1: Replace the private helper**

Implement:

```python
from vercor.output import OutputSpec


def resolve_output(output: OutputSpec | None = None) -> OutputSpec:
    """Return a validated explicit or disabled output declaration."""

    if output is None:
        return OutputSpec()
    if not isinstance(output, OutputSpec):
        raise TypeError("output must be OutputSpec or None")
    return output
```

Remove `step_period_output` and the unused `PeriodOutput` import.

- [ ] **Step 2: Update all resolver consumers**

In the four slab factories, the shared time-interpolated data helper, and
direct JCM land, replace:

```python
from vercor.setups._output import bundled_output
```

with:

```python
from vercor.setups._output import resolve_output
```

and replace each `bundled_output(output)` call with `resolve_output(output)`.

- [ ] **Step 3: Align paired-JCM land configuration**

Remove the private setup-helper import from `vercor/setups/config.py` and
define:

```python
land_output: OutputSpec = field(default_factory=OutputSpec)
```

Do not change `_default_jcm_atmosphere_config`.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```bash
env CONDA_NO_PLUGINS=true conda run -n scipy pytest \
  tests/test_bundled_period_output.py \
  tests/test_setup_lifecycle_helpers.py \
  tests/test_setup_agnostic_api.py \
  -q --tb=short
```

Expected: all tests pass.

- [ ] **Step 5: Run output/setup regressions**

Run:

```bash
env CONDA_NO_PLUGINS=true conda run -n scipy pytest \
  tests/test_v0_4_output_providers.py \
  tests/test_native_period_output.py \
  tests/test_component_models_coverage.py \
  tests/test_setup_boundaries.py \
  tests/test_api_architecture_review.py \
  tests/test_distribution_boundaries.py \
  -q --fast --tb=short
```

Expected: all tests pass; exact public signatures remain unchanged.

- [ ] **Step 6: Commit the behavior**

```bash
git add vercor/setups tests/test_bundled_period_output.py \
  tests/test_setup_lifecycle_helpers.py tests/test_setup_agnostic_api.py
git commit -m "fix: align bundled output defaults"
```

### Task 3: Synchronize documentation

**Files:**
- Modify: `README.md`
- Modify: `DESIGN.md`
- Modify: `CHANGELOG.md`
- Modify: `DEPENDENCIES.md`
- Modify: `docs/api-architecture-review.md`
- Modify: `PROGRESS.md`

**Interfaces:**
- Consumes: final default semantics.
- Produces: one consistent user, architecture, dependency, and progress record.

- [ ] **Step 1: Update user and architecture descriptions**

State that omission means `OutputSpec()` and add an explicit step example:

```python
make_slab_ocean(
    grid,
    output=OutputSpec(period=PeriodOutput(frequency="step")),
)
```

Explain that final-field output remains run-level and that external adapters
retain native snapshots.

- [ ] **Step 2: Update dependency ownership**

Move `vercor/setups/config.py` back into the same dependency layer as
`vercor/output/__init__.py` consumers because it no longer imports the private
setup output resolver. Keep `vercor/setups/_output.py` in the component/output
declaration layer.

- [ ] **Step 3: Record focused evidence**

Add a new top `PROGRESS.md` entry with the red failures, focused pass counts,
and pending full verification. Preserve the previous configurability entry as
history, explicitly noting that its step-default statement was superseded.

- [ ] **Step 4: Run documentation and contract checks**

Run:

```bash
env CONDA_NO_PLUGINS=true conda run -n scipy pytest \
  tests/test_api_architecture_review.py \
  tests/test_distribution_boundaries.py \
  tests/test_versioning_policy.py \
  -q --tb=short
git diff --check
```

Expected: all tests pass and signature fixtures remain unchanged.

- [ ] **Step 5: Commit documentation**

```bash
git add README.md DESIGN.md CHANGELOG.md DEPENDENCIES.md \
  docs/api-architecture-review.md PROGRESS.md
git commit -m "docs: align bundled output defaults"
```

### Task 4: Complete verification

**Files:**
- Modify only deterministic Black output if required.
- Modify: `PROGRESS.md` with exact measured evidence.

**Interfaces:**
- Consumes: complete implementation and documentation.
- Produces: clean, verified commits on `refactor`.

- [ ] **Step 1: Run static gates**

Run:

```bash
env CONDA_NO_PLUGINS=true conda run -n scipy black vercor examples tests
env CONDA_NO_PLUGINS=true conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics
env CONDA_NO_PLUGINS=true conda run -n scipy mypy vercor examples tests
env CONDA_NO_PLUGINS=true conda run -n scipy python -m compileall -q vercor examples tests
git diff --check
```

Expected: Black clean, flake8 `0`, mypy success, compileall exit `0`, and no
whitespace errors.

- [ ] **Step 2: Run fast and full suites**

Run:

```bash
env CONDA_NO_PLUGINS=true conda run -n scipy pytest tests/ -q --fast --tb=short
env CONDA_NO_PLUGINS=true conda run -n scipy pytest tests/ -v -n4 \
  --dist=loadscope --max-worker-restart=0 --durations=25 --tb=short
```

Expected: all tests pass.

- [ ] **Step 3: Run branch coverage**

Run:

```bash
env CONDA_NO_PLUGINS=true conda run -n scipy pytest \
  --cov=vercor --cov-branch --cov-report=term:skip-covered \
  tests/ -q -n4 --dist=loadscope --max-worker-restart=0 --tb=short
```

Expected: all tests pass and total coverage remains at least 90%.

- [ ] **Step 4: Record final evidence and rerun final focus**

Update `PROGRESS.md` with exact counts, warnings, coverage, and static results.
Then rerun Black, flake8, mypy, compileall, the modified test files, and
`git diff --check`.

- [ ] **Step 5: Commit final evidence**

```bash
git add PROGRESS.md
git commit -m "docs: record output default alignment verification"
```

- [ ] **Step 6: Request independent review and inspect repository state**

Review the exact implementation range against the approved spec. Fix all
Critical and Important findings. Then run:

```bash
git status --short
git log -8 --oneline
```

Expected: no review blockers and a clean worktree.
