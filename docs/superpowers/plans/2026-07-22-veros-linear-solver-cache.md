# Veros Linear Solver Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reuse the solver created by one Veros component across its copy-owned runtime states without rebuilding the ILU preconditioner or retaining copied states in Veros' cache.

**Architecture:** `VerosGCMSetupState` captures its setup-created solver once,
removes the setup-state cache key, and passes the solver into the existing
copy-before-mutate step helper. That helper temporarily binds each copied
`VerosState` identity to the component solver in Veros' native cache, calls the
native step, and restores or removes the temporary entry in `finally`. A
single private compatibility helper validates the shared mutable cache exposed
by Veros `>=1.6.2,<1.7` before either operation.

**Tech Stack:** Python 3.13, Veros >=1.6.2,<1.7, SciPy, JAX, pytest, Black, flake8, mypy.

## Global Constraints

- The solver cache is scoped to one `VerosGCMSetupState`; do not share solvers across components or Coupler instances.
- Evolving Veros state remains exclusively in `SetupResult.payload` and `StepResult.payload`.
- Do not change any public VerCOR API, component declaration, runtime payload type, model numerics, output contract, or dependency order.
- Preserve copy-before-mutate behavior for both `jitted=False` and `jitted=True`.
- Use the `scipy` conda environment for every Python, formatting, typing, and test command.
- Write each regression test and verify its expected failure before editing production code.

---

### Task 1: Component-scoped Veros solver reuse

**Files:**
- Modify: `tests/test_external_components_coverage.py:1426`
- Modify: `tests/test_external_components_coverage.py:1900`
- Modify: `vercor/setups/_external/veros_state.py:105`
- Modify: `vercor/setups/_external/veros_gcm_state.py:40`
- Modify: `PROGRESS.md:9`

**Interfaces:**
- Consumes: Veros' untyped `get_linear_solver(state)` accessor and its decorator-owned `cache` mapping keyed by `(state,)`.
- Produces: `get_component_linear_solver(state: VerosState) -> Any` for setup-time solver capture.
- Produces: `pure(state: VerosState, jitted: bool, step: Callable[[VerosState], None], linear_solver: Any) -> VerosState` for copy-before-mutate stepping with scoped solver binding.
- Preserves: `VerosGCMSetupState._step_function: Callable[[Any], Any]` and the native `VerosState` runtime payload.

- [ ] **Step 1: Add hashable fake step state and import the native accessor**

In `tests/test_external_components_coverage.py`, import the native accessor with an unambiguous test-only name:

```python
from veros.core.external.solvers import (
    get_linear_solver as veros_get_linear_solver,
)
```

Add this focused mutable fake beside the existing Veros fake-state helpers:

```python
class _FakeVerosStepState:
    def __init__(self, counter: int) -> None:
        self.counter = counter
```

Do not make it a value-equality dataclass: Veros' defect and the fix both rely on identity hashing.

- [ ] **Step 2: Replace the existing `pure()` test with a two-copy reuse regression**

Replace `test_veros_pure_runs_step_on_copied_state` with:

```python
def test_veros_pure_reuses_component_solver_for_copied_states(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_state = _FakeVerosStepState(counter=1)
    copied_states = iter(
        (_FakeVerosStepState(counter=1), _FakeVerosStepState(counter=1))
    )
    solver = object()
    solver_cache: dict[tuple[Any, ...], Any] = {}

    monkeypatch.setattr(veros_get_linear_solver, "cache", solver_cache)
    monkeypatch.setattr(
        veros_get_linear_solver.__wrapped__, "cache", solver_cache
    )
    monkeypatch.setattr(
        veros_state_module,
        "copy_state",
        lambda state, jitted=True: next(copied_states),
    )

    stepped_states: list[Any] = []

    def fake_step(state: Any) -> None:
        assert veros_get_linear_solver(state) is solver
        state.counter += 1
        stepped_states.append(state)

    results = tuple(
        veros_state_module.pure(
            original_state,
            jitted=False,
            step=fake_step,
            linear_solver=solver,
        )
        for _ in range(2)
    )

    assert results == tuple(stepped_states)
    assert [state.counter for state in results] == [2, 2]
    assert original_state.counter == 1
    assert solver_cache == {}
```

This exercises two different copied identities, proves that the native
accessor returns the same component solver, and proves successful-step cleanup.

- [ ] **Step 3: Add exception cleanup and prior-entry restoration coverage**

Add the following parametrized regression immediately after the reuse test:

```python
@pytest.mark.parametrize("has_prior_entry", (False, True))
def test_veros_pure_restores_solver_cache_when_step_fails(
    monkeypatch: pytest.MonkeyPatch,
    has_prior_entry: bool,
) -> None:
    original_state = _FakeVerosStepState(counter=1)
    copied_state = _FakeVerosStepState(counter=1)
    component_solver = object()
    prior_solver = object()
    solver_cache: dict[tuple[Any, ...], Any] = {}
    key = (copied_state,)
    if has_prior_entry:
        solver_cache[key] = prior_solver

    monkeypatch.setattr(veros_get_linear_solver, "cache", solver_cache)
    monkeypatch.setattr(
        veros_get_linear_solver.__wrapped__, "cache", solver_cache
    )
    monkeypatch.setattr(
        veros_state_module,
        "copy_state",
        lambda state, jitted=True: copied_state,
    )

    def failing_step(state: Any) -> None:
        assert veros_get_linear_solver(state) is component_solver
        raise RuntimeError("native step failed")

    with pytest.raises(RuntimeError, match="native step failed"):
        veros_state_module.pure(
            original_state,
            jitted=False,
            step=failing_step,
            linear_solver=component_solver,
        )

    assert solver_cache == ({key: prior_solver} if has_prior_entry else {})
```

- [ ] **Step 4: Extend the constructor test to require one setup-state solver lookup**

In `test_veros_constructor_builds_jax_backed_grid`, define a solver token and
record the lookup before calling `make_veros_gcm`:

```python
    component_solver = object()
    solver_states: list[Any] = []

    def fake_get_component_linear_solver(veros_state: Any) -> Any:
        solver_states.append(veros_state)
        return component_solver

    monkeypatch.setattr(
        veros_state_module,
        "get_component_linear_solver",
        fake_get_component_linear_solver,
        raising=False,
    )
```

Add this assertion after construction:

```python
    assert solver_states == [state]
```

The `raising=False` argument makes the test fail by assertion on the current
code instead of failing during monkeypatch setup because the function does not
exist yet.

- [ ] **Step 5: Run the exact tests and verify RED**

Run:

```bash
conda run -n scipy pytest \
  tests/test_external_components_coverage.py::test_veros_pure_reuses_component_solver_for_copied_states \
  tests/test_external_components_coverage.py::test_veros_pure_restores_solver_cache_when_step_fails \
  tests/test_external_components_coverage.py::test_veros_constructor_builds_jax_backed_grid \
  -q --tb=short -n0
```

Expected: four failing cases. The three `pure()` cases reject the new
`linear_solver` argument, and the constructor case reports `solver_states == []`.

- [ ] **Step 6: Add setup-time solver capture and exception-safe cache binding**

In `vercor/setups/_external/veros_state.py`, add the setup-time accessor before
`pure()`:

```python
def get_component_linear_solver(state: VerosState) -> Any:
    """Return the native linear solver created for one Veros component."""

    from veros.core.external.solvers import get_linear_solver

    return get_linear_solver(state)
```

Replace `pure()` with the typed solver-aware implementation:

```python
def pure(
    state: VerosState,
    jitted: bool,
    step: Callable[[VerosState], None],
    linear_solver: Any,
) -> VerosState:
    """Copy state and run one native step with the component-owned solver."""

    from veros.core.external.solvers import get_linear_solver

    next_state = copy_state(state, jitted=jitted)
    solver_cache = cast(dict[tuple[VerosState], Any], get_linear_solver.cache)
    cache_key = (next_state,)
    missing = object()
    previous_solver = solver_cache.get(cache_key, missing)
    solver_cache[cache_key] = linear_solver
    try:
        step(next_state)
    finally:
        if previous_solver is missing:
            solver_cache.pop(cache_key, None)
        else:
            solver_cache[cache_key] = previous_solver
    return next_state
```

Add `"get_component_linear_solver"` to `veros_state.py.__all__` because the
sibling setup-state module consumes it. Keep the scoped cache manipulation in
this single Veros-integration owner; do not duplicate it in runtime dispatch.

In `vercor/setups/_external/veros_gcm_state.py`, add the annotation:

```python
    _linear_solver: Any
```

Immediately after `self.model.setup()`, capture the already-created solver and
pass it into the existing partial:

```python
        self._linear_solver = _veros_state.get_component_linear_solver(
            self.model.state
        )
        self._veros_state = _veros_state.copy_state(
            self.model.state,
            jitted=jitted,
        )
        self._step_function = cast(
            Callable[[Any], Any],
            partial(
                _veros_state.pure,
                jitted=jitted,
                step=self.model.step,
                linear_solver=self._linear_solver,
            ),
        )
```

- [ ] **Step 7: Run focused GREEN verification**

Run:

```bash
conda run -n scipy pytest \
  tests/test_external_components_coverage.py::test_veros_pure_reuses_component_solver_for_copied_states \
  tests/test_external_components_coverage.py::test_veros_pure_restores_solver_cache_when_step_fails \
  tests/test_external_components_coverage.py::test_veros_constructor_builds_jax_backed_grid \
  -q --tb=short -n0
conda run -n scipy pytest tests/test_external_components_coverage.py -q --fast --tb=short
```

Expected: the exact selection passes 4/4, and the complete external-component
file passes with no failures.

- [ ] **Step 8: Reproduce two real Veros steps with stable cache size**

Run this in a fresh process so no unrelated Veros state identities are present:

```bash
conda run -n scipy python -c 'from vercor.setups._external.veros_runtime_settings import configure_veros_runtime; configure_veros_runtime(); from vercor.setups._external.veros_gcm_state import VerosGCMSetupState; from vercor.setups._external.veros_state import copy_state; from veros.core.external.solvers import get_linear_solver; owner=VerosGCMSetupState(do_spinup=False,jitted=False); state=owner._veros_state; print("initial_cache", len(get_linear_solver.cache)); state=copy_state(state,jitted=True); state=owner._step_function(state); print("after_step_1", len(get_linear_solver.cache)); state=copy_state(state,jitted=True); state=owner._step_function(state); print("after_step_2", len(get_linear_solver.cache))'
```

Expected: setup emits `Computing ILU preconditioner...` once; neither runtime
step emits it; the printed cache sizes are `1`, `1`, and `1`.

- [ ] **Step 9: Replace the active progress entry with the completed outcome**

Replace the current `IN PROGRESS` block in `PROGRESS.md` with:

```markdown
- Veros component-scoped linear-solver caching completed locally (2026-07-22):
  the setup-created solver is reused across copy-owned native states with
  exception-safe temporary cache binding. TDD RED/GREEN was 4/4; the real
  Veros cache stayed 1→1→1 with no runtime ILU rebuild; external-component,
  fast, full, coverage, formatting, lint, typing, compile, and whitespace gates
  passed. Public APIs, native payloads, and model numerics remain unchanged.
```

Confirm `PROGRESS.md` remains at or below its enforced 180-line limit:

```bash
wc -l PROGRESS.md
```

Expected: at most 180 lines.

- [ ] **Step 10: Run formatting, typing, lint, compile, test, and coverage gates**

Run all commands from the repository root:

```bash
conda run -n scipy black vercor examples tests
conda run -n scipy flake8 . --count --max-line-length=120 --statistics
conda run -n scipy mypy vercor examples tests
conda run -n scipy python -m compileall -q vercor examples tests
conda run -n scipy pytest tests/ -q --fast --tb=short
conda run -n scipy pytest tests/ -q --tb=short
conda run -n scipy pytest tests/ -q --tb=short --cov=vercor --cov-branch --cov-report=term-missing --cov-fail-under=90
git diff --check
```

Expected: Black reports files unchanged after its formatting pass; flake8
reports zero errors; mypy succeeds; compileall and whitespace checks are
silent; fast, full, and coverage suites have zero failures; branch coverage is
at least 90%. Only the already documented third-party Flax and JCM/xarray
warnings may remain.

- [ ] **Step 11: Review scope and commit the implementation**

Inspect the final diff and confirm that it contains only the approved cache
behavior, its tests, the plan, and the final progress entry:

```bash
git diff --stat
git diff -- vercor/setups/_external/veros_state.py vercor/setups/_external/veros_gcm_state.py tests/test_external_components_coverage.py PROGRESS.md docs/superpowers/plans/2026-07-22-veros-linear-solver-cache.md
```

Then stage and verify exactly those files:

```bash
git add \
  vercor/setups/_external/veros_state.py \
  vercor/setups/_external/veros_gcm_state.py \
  tests/test_external_components_coverage.py \
  PROGRESS.md \
  docs/superpowers/plans/2026-07-22-veros-linear-solver-cache.md
git diff --cached --check
git diff --cached --stat
git commit -m "fix: reuse Veros linear solver across runtime steps"
```

Expected: one implementation commit is created after every required gate has
passed; no push, tag, PR, release, or publication occurs.
