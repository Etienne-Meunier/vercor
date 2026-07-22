# Veros Payload Copy Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the bundled native Veros runtime payload compatible with VerCOR's copy-owned component boundary.

**Architecture:** Keep the generic component adapter and public lifecycle API unchanged. Normalize the one Veros-owned live `dict_keys` metadata view to an immutable tuple in the existing `copy_state()` construction boundary.

**Tech Stack:** Python 3.13, Veros, NumPy, pytest, JAX.

## Global Constraints

- No public API, component contract, payload type, stepping behavior, or output behavior changes.
- Preserve VerCOR's defensive payload-copy and runtime-state isolation guarantees.
- Use the `scipy` conda environment for every Python and pytest command.
- Write and verify the regression test before editing production code.

---

### Task 1: Normalize copied Veros settings metadata

**Files:**
- Modify: `tests/test_external_components_coverage.py`
- Modify: `vercor/setups/_external/veros_state.py`
- Modify: `PROGRESS.md`

**Interfaces:**
- Consumes: `copy_state(tree: VerosState, jitted: bool = True) -> VerosState`
- Produces: the same `copy_state` interface, returning a native state whose settings field-name collection is immutable and `deepcopy`-compatible.

- [x] **Step 1: Make the fake Veros settings reproduce the dependency behavior**

Add the live field-name view to `_FakeSettings.__init__`:

```python
object.__setattr__(self, "__fields__", metadata.keys())
```

- [x] **Step 2: Add the failing regression test**

```python
@pytest.mark.parametrize("jitted", (False, True))
def test_veros_copy_state_returns_deepcopy_compatible_state(
    monkeypatch: pytest.MonkeyPatch,
    jitted: bool,
) -> None:
    monkeypatch.setattr(veros_state_module, "VerosState", _ConstructedVerosState)
    source = _make_copyable_fake_veros_state()

    copied = veros_state_module.copy_state(source, jitted=jitted)
    copied_again = deepcopy(copied)

    assert isinstance(copied.settings.__fields__, tuple)
    assert copied_again is not copied
    assert copied_again.settings is not copied.settings
    assert copied_again.variables is not copied.variables
```

- [x] **Step 3: Run the regression test and verify RED**

Run:

```bash
conda run -n scipy pytest tests/test_external_components_coverage.py::test_veros_copy_state_returns_deepcopy_compatible_state -q --tb=short
```

Expected: FAIL because `copied.settings.__fields__` remains `dict_keys`, causing `deepcopy(copied)` to raise `TypeError: cannot pickle 'dict_keys' object`.

- [x] **Step 4: Implement the minimal Veros-specific normalization**

After the existing jitted/non-jitted branch in `copy_state()` and before its
return, add:

```python
object.__setattr__(
    state_copy.settings,
    "__fields__",
    tuple(state_copy.settings.__fields__),
)
```

This preserves field order and native state type without changing generic payload handling.

- [x] **Step 5: Run focused GREEN verification**

Run:

```bash
conda run -n scipy pytest tests/test_external_components_coverage.py -q --fast --tb=short
conda run -n scipy pytest tests/test_v0_4_component_contracts.py -q --fast --tb=short
```

Expected: both commands pass with zero failures.

- [x] **Step 6: Re-run the reported example through preparation**

Run:

```bash
conda run -n scipy python examples/run_veros_with_era5data.py
```

Expected: coupler preparation passes the former `deepcopy` failure. The long 365-step run may be interrupted after observing at least one Veros runtime step; interruption is not evidence for the full test gates.

- [x] **Step 7: Run formatting, static, and regression gates**

Run:

```bash
conda run -n scipy black --check vercor/setups/_external/veros_state.py tests/test_external_components_coverage.py
conda run -n scipy flake8 vercor/setups/_external/veros_state.py tests/test_external_components_coverage.py --count --max-line-length=120 --statistics
conda run -n scipy mypy vercor/setups/_external/veros_state.py tests/test_external_components_coverage.py
conda run -n scipy pytest tests/ -q --fast --tb=short
```

Expected: static commands report zero errors and pytest reports zero test failures, apart from any explicitly documented environment-only Git/Xcode test limitation.

- [x] **Step 8: Update progress and perform final verification**

Add a dated `PROGRESS.md` entry with the root cause, RED/GREEN evidence, example result, test totals, and any environment limitation. Re-run the focused regression test and whitespace check before reporting completion.

- [ ] **Step 9: Commit only if the complete unit suite passes and the user requests a commit**

Do not commit based only on focused or fast evidence. If authorized after full-suite success:

```bash
git add docs/superpowers/specs/2026-07-22-veros-payload-copy-design.md docs/superpowers/plans/2026-07-22-veros-payload-copy.md tests/test_external_components_coverage.py vercor/setups/_external/veros_state.py PROGRESS.md
git commit -m "Fix Veros runtime payload copying"
```
