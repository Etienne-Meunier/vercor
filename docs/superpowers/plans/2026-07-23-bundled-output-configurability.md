# Bundled Output Configurability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let every bundled slab and data factory accept a complete per-component `OutputSpec` while preserving step-period output when the argument is omitted.

**Architecture:** A private setup-layer resolver owns the default and validates optional overrides. Slab factories and direct JCM land resolve their arguments into `ComponentSpec.output`; shared time-interpolated data factories forward the argument through one helper. The paired JCM setup owns its land policy in `JCMLandAtmosphereConfig`, while the existing output session remains the sole sampler, accumulator, and writer.

**Tech Stack:** Python 3.13, JAX, frozen dataclasses, h5netcdf, pytest, Black, flake8, mypy, coverage.py, Git.

## Global Constraints

- Preserve all existing positional factory arguments and make only `output` keyword-only.
- The public factory parameter is `output: OutputSpec | None = None`.
- Omitted output retains `OutputSpec(period=PeriodOutput(frequency="step"))`.
- A supplied `OutputSpec` is preserved as the component's complete declaration.
- `OutputSpec()` disables period and snapshot output for that component.
- `OutputTarget` remains run-level enablement and never overrides component cadence.
- Do not change providers, accumulation, cadence calculation, NetCDF writing, component kernels, or execution backends.
- Keep optional JCM imports lazy and preserve dependency-free slab imports.
- Use test-driven development and update `DESIGN.md`, `PROGRESS.md`, `DEPENDENCIES.md`, README, and signature fixtures.
- Run focused, fast, full, coverage, formatting, linting, typing, compile, distribution, and whitespace gates before the final implementation commit.

---

## File Structure

- Modify `vercor/setups/_output.py`: resolve omitted or supplied bundled output declarations and validate overrides.
- Modify `vercor/setups/_slab/atmosphere.py`: accept and forward a keyword-only output declaration.
- Modify `vercor/setups/_slab/land.py`: accept and forward a keyword-only output declaration.
- Modify `vercor/setups/_slab/ocean.py`: accept and forward a keyword-only output declaration.
- Modify `vercor/setups/_slab/seaice.py`: accept and forward a keyword-only output declaration.
- Modify `vercor/setups/_data/_component_helpers.py`: accept and resolve the shared data-factory output declaration.
- Modify `vercor/setups/_data/era5_atmosphere.py`: expose and forward `output`.
- Modify `vercor/setups/_data/era5_land.py`: expose and forward `output`.
- Modify `vercor/setups/_data/era5_ocean.py`: expose and forward `output`.
- Modify `vercor/setups/_data/erainterim_ocean.py`: expose and forward `output`.
- Modify `vercor/setups/_data/jcm_land.py`: expose and resolve `output`.
- Modify `vercor/setups/config.py`: add independently configurable `JCMLandAtmosphereConfig.land_output`.
- Modify `vercor/setups/_jcm.py`: forward paired JCM land output.
- Modify `tests/test_bundled_period_output.py`: declaration, forwarding, validation, disabling, and monthly integration tests.
- Modify `tests/contracts/vercor-0.4.0a1-public-signatures.json`: record the additive keyword-only API and new paired-JCM config field.
- Modify `README.md`, `DESIGN.md`, `DEPENDENCIES.md`, and `PROGRESS.md`: document the API, ownership, dependency order, and verification.

### Task 1: Specify the configurable output contract

**Files:**
- Modify: `tests/test_bundled_period_output.py`
- Test: `tests/test_bundled_period_output.py`

**Interfaces:**
- Consumes: public `OutputSpec`, `PeriodOutput`, slab factories, the shared data helper, direct JCM land factory, and paired JCM setup.
- Produces: executable acceptance criteria for the resolver and all public factory boundaries.

- [ ] **Step 1: Add failing slab/default/validation tests**

Add imports for `inspect.Parameter`, `inspect.signature`, and the private resolver module. Add:

```python
_SLAB_FACTORIES = (
    make_slab_atmosphere,
    make_slab_land,
    make_slab_ocean,
    make_slab_seaice,
)


@pytest.mark.parametrize("factory", _SLAB_FACTORIES)
def test_slab_factory_accepts_keyword_only_output_spec(factory: Any) -> None:
    parameter = signature(factory).parameters["output"]
    assert parameter.kind is Parameter.KEYWORD_ONLY
    assert parameter.default is None

    custom = OutputSpec(
        period=PeriodOutput(
            frequency="month",
            variables=(factory(make_test_grid()).spec.outputs[0],),
        )
    )
    component = factory(make_test_grid(name="configured-slab"), output=custom)
    assert component.spec.output is custom


@pytest.mark.parametrize("factory", _SLAB_FACTORIES)
def test_slab_factory_omitted_output_retains_step_default(factory: Any) -> None:
    component = factory(make_test_grid(name="default-slab"))
    assert component.spec.output == OutputSpec(
        period=PeriodOutput(frequency="step")
    )


def test_bundled_output_resolver_rejects_invalid_override() -> None:
    with pytest.raises(TypeError, match="output must be OutputSpec or None"):
        bundled_output(cast(Any, "month"))
```

- [ ] **Step 2: Add failing data helper and JCM tests**

Import the four public time-interpolated data factories and add:

```python
@pytest.mark.parametrize(
    "factory",
    (
        make_era5_atmosphere,
        make_era5_land,
        make_era5_ocean,
        make_erainterim_ocean,
    ),
)
def test_public_data_factory_forwards_output_spec(factory: Any) -> None:
    parameter = signature(factory).parameters["output"]
    assert parameter.kind is Parameter.KEYWORD_ONLY
    custom = OutputSpec(period=PeriodOutput(frequency="month"))
    component = factory(output=custom)
    assert component.spec.output is custom
```

Extend the shared helper test to pass a custom declaration and assert identity:

```python
custom = OutputSpec(
    period=PeriodOutput(
        frequency="month",
        variables=("temperature",),
    )
)
component = time_interpolated_data_component(
    name="DATA",
    grid=grid,
    fields={"temperature": jnp.full(grid.shape, 280.0)},
    outputs=("temperature",),
    output=custom,
)
assert component.spec.output is custom
```

Extend direct JCM land construction with `output=custom` and assert identity.
Add a `JCMLandAtmosphereConfig` assertion:

```python
config = JCMLandAtmosphereConfig(land_output=custom)
assert config.land_output is custom
```

- [ ] **Step 3: Add failing disabling and monthly-mean tests**

Add a one-component slab run with `output=OutputSpec()` and assert no average
file is created. Add a two-step monthly slab-atmosphere run ending at a month
boundary. Obtain the post-step `temperature_2m` reference values from matching
one-step and two-step output-free runs, then assert the monthly NetCDF value
equals their arithmetic mean. Use the ordinary slab factory output override
and the existing runtime-field provider; do not call private accumulator
functions.

- [ ] **Step 4: Run the new tests and verify RED**

Run:

```bash
env CONDA_NO_PLUGINS=true conda run -n scipy pytest tests/test_bundled_period_output.py -q --tb=short
```

Expected: failures report missing `output` parameters, missing `bundled_output`,
and missing `land_output`.

### Task 2: Implement the shared resolver and slab overrides

**Files:**
- Modify: `vercor/setups/_output.py`
- Modify: `vercor/setups/_slab/atmosphere.py`
- Modify: `vercor/setups/_slab/land.py`
- Modify: `vercor/setups/_slab/ocean.py`
- Modify: `vercor/setups/_slab/seaice.py`
- Test: `tests/test_bundled_period_output.py`

**Interfaces:**
- Consumes: `OutputSpec | None`.
- Produces: `bundled_output(output: OutputSpec | None = None) -> OutputSpec` and four public slab factory `output` parameters.

- [ ] **Step 1: Replace the fixed helper with a resolver**

Implement:

```python
def bundled_output(output: OutputSpec | None = None) -> OutputSpec:
    """Return a validated bundled output declaration."""

    if output is None:
        return OutputSpec(period=PeriodOutput(frequency="step"))
    if not isinstance(output, OutputSpec):
        raise TypeError("output must be OutputSpec or None")
    return output
```

Keep `step_period_output()` only if a dataclass `default_factory` needs the
zero-argument constructor; implement it by delegating to `bundled_output()` so
there remains one default declaration.

- [ ] **Step 2: Add the slab parameters**

Use these signatures:

```python
def make_slab_atmosphere(
    grid: RectilinearGrid,
    name: str = "ATM",
    *,
    output: OutputSpec | None = None,
) -> Component:
```

```python
def make_slab_land(
    grid: RectilinearGrid,
    name: str = "LND",
    *,
    output: OutputSpec | None = None,
) -> Component:
```

```python
def make_slab_ocean(
    grid: RectilinearGrid,
    name: str = "OCN",
    mixed_layer_depth: float = 30.0,
    *,
    output: OutputSpec | None = None,
) -> Component:
```

```python
def make_slab_seaice(
    grid: RectilinearGrid,
    name: str = "ICE",
    *,
    output: OutputSpec | None = None,
) -> Component:
```

Import `OutputSpec` for annotations and store:

```python
output=bundled_output(output),
```

- [ ] **Step 3: Run the slab and validation tests**

Run:

```bash
env CONDA_NO_PLUGINS=true conda run -n scipy pytest tests/test_bundled_period_output.py -q --tb=short
```

Expected: slab/default/validation tests pass; data and JCM forwarding tests
remain red.

- [ ] **Step 4: Commit the slab unit**

```bash
git add vercor/setups/_output.py vercor/setups/_slab tests/test_bundled_period_output.py
git commit -m "feat: configure slab period output"
```

### Task 3: Implement data and JCM overrides

**Files:**
- Modify: `vercor/setups/_data/_component_helpers.py`
- Modify: `vercor/setups/_data/era5_atmosphere.py`
- Modify: `vercor/setups/_data/era5_land.py`
- Modify: `vercor/setups/_data/era5_ocean.py`
- Modify: `vercor/setups/_data/erainterim_ocean.py`
- Modify: `vercor/setups/_data/jcm_land.py`
- Modify: `vercor/setups/config.py`
- Modify: `vercor/setups/_jcm.py`
- Test: `tests/test_bundled_period_output.py`
- Test: `tests/test_setup_agnostic_api.py`

**Interfaces:**
- Consumes: `bundled_output`, public `OutputSpec`, and existing data/JCM construction inputs.
- Produces: keyword-only output overrides for every bundled data factory and `JCMLandAtmosphereConfig.land_output`.

- [ ] **Step 1: Extend the shared data helper**

Add:

```python
output: OutputSpec | None = None,
```

to `time_interpolated_data_component`, import `OutputSpec`, and construct the
spec with:

```python
output=bundled_output(output),
```

- [ ] **Step 2: Extend every public time-interpolated data factory**

Preserve all current positional parameters and append:

```python
*,
output: OutputSpec | None = None,
```

to `make_era5_atmosphere`, `make_era5_land`, `make_era5_ocean`, and
`make_erainterim_ocean`. Import `OutputSpec` and pass:

```python
output=output,
```

to `time_interpolated_data_component`.

- [ ] **Step 3: Extend direct JCM land**

Append a keyword-only `output: OutputSpec | None = None` parameter to
`make_jcm_land`, import `OutputSpec`, and set:

```python
output=bundled_output(output),
```

in its `ComponentSpec`.

- [ ] **Step 4: Extend paired JCM configuration**

Add to `JCMLandAtmosphereConfig`:

```python
land_output: OutputSpec = field(default_factory=step_period_output)
```

and pass it in `_jcm.py`:

```python
land = make_jcm_land(
    coords,
    forcing,
    ocn_grid,
    name=config.land_name,
    output=config.land_output,
)
```

- [ ] **Step 5: Run the complete focused module**

Run:

```bash
env CONDA_NO_PLUGINS=true conda run -n scipy pytest tests/test_bundled_period_output.py -q --tb=short
```

Expected: all parameter-expanded tests pass.

- [ ] **Step 6: Run bundled and output regressions**

Run:

```bash
env CONDA_NO_PLUGINS=true conda run -n scipy pytest \
  tests/test_bundled_period_output.py \
  tests/test_v0_4_output_providers.py \
  tests/test_native_period_output.py \
  tests/test_slab_kernels.py \
  tests/test_data_component_kernels.py \
  tests/test_component_models_coverage.py \
  tests/test_setup_boundaries.py \
  -q --fast --tb=short
```

Expected: pass with no failures.

- [ ] **Step 7: Commit the data/JCM unit**

```bash
git add vercor/setups/_data vercor/setups/config.py vercor/setups/_jcm.py tests/test_bundled_period_output.py
git commit -m "feat: configure data component output"
```

### Task 4: Synchronize public contracts and documentation

**Files:**
- Modify: `tests/contracts/vercor-0.4.0a1-public-signatures.json`
- Modify: `README.md`
- Modify: `DESIGN.md`
- Modify: `DEPENDENCIES.md`
- Modify: `PROGRESS.md`
- Test: `tests/test_api_architecture_review.py`
- Test: `tests/test_distribution_boundaries.py`

**Interfaces:**
- Consumes: final public signatures and verified behavior.
- Produces: synchronized API contracts and user/developer documentation.

- [ ] **Step 1: Run signature tests and capture exact failures**

Run:

```bash
env CONDA_NO_PLUGINS=true conda run -n scipy pytest \
  tests/test_api_architecture_review.py \
  tests/test_distribution_boundaries.py \
  tests/test_v0_4_public_api.py \
  -q --tb=short
```

Expected: exact signature fixture failures for the changed public factories.

- [ ] **Step 2: Update exact signature fixtures**

Edit only these JSON entries: `vercor.setups.make_slab_atmosphere`,
`make_slab_land`, `make_slab_ocean`, `make_slab_seaice`,
`make_era5_atmosphere`, `make_era5_land`, `make_era5_ocean`,
`make_erainterim_ocean`, `make_jcm_land`, and
`vercor.setups.JCMLandAtmosphereConfig`. Each factory signature gains its final
keyword-only `output: vercor.output.OutputSpec | None = None`; the config
signature gains `land_output: vercor.output.OutputSpec = <factory>`. Preserve
every unrelated public signature byte for byte.

- [ ] **Step 3: Update user and architecture documentation**

Document:

```python
output=OutputSpec(
    period=PeriodOutput(
        frequency="month",
        variables=("sea_surface_temperature",),
    )
)
```

in `README.md`; state in `DESIGN.md` that bundled slab/data defaults are
overridable per component; add the `_output.py` ownership edge to
`DEPENDENCIES.md`; and record behavior plus current focused evidence in
`PROGRESS.md`.

- [ ] **Step 4: Run contract and documentation tests**

Run:

```bash
env CONDA_NO_PLUGINS=true conda run -n scipy pytest \
  tests/test_api_architecture_review.py \
  tests/test_distribution_boundaries.py \
  tests/test_v0_4_public_api.py \
  tests/test_bundled_period_output.py \
  -q --tb=short
```

Expected: pass with no failures.

- [ ] **Step 5: Commit contracts and documentation**

```bash
git add tests/contracts README.md DESIGN.md DEPENDENCIES.md PROGRESS.md
git commit -m "docs: describe configurable bundled output"
```

### Task 5: Complete verification

**Files:**
- Modify only files changed by deterministic Black formatting.
- Modify: `PROGRESS.md` with final measured evidence.

**Interfaces:**
- Consumes: complete implementation and tests.
- Produces: a clean, fully verified branch and final evidence record.

- [ ] **Step 1: Format and run static gates**

Run:

```bash
env CONDA_NO_PLUGINS=true conda run -n scipy black vercor examples tests
env CONDA_NO_PLUGINS=true conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics
env CONDA_NO_PLUGINS=true conda run -n scipy mypy vercor examples tests
env CONDA_NO_PLUGINS=true conda run -n scipy python -m compileall -q vercor examples tests
git diff --check
```

Expected: Black reports no remaining changes after formatting, flake8 reports
`0`, mypy reports success, compileall exits `0`, and whitespace checks are
clean.

- [ ] **Step 2: Run the fast suite**

Run:

```bash
env CONDA_NO_PLUGINS=true conda run -n scipy pytest tests/ -q --fast --tb=short
```

Expected: all tests pass.

- [ ] **Step 3: Run the full suite**

Run:

```bash
env CONDA_NO_PLUGINS=true conda run -n scipy pytest tests/ -q -n4 --dist=loadscope --max-worker-restart=0 --durations=25 --tb=short
```

Expected: all tests pass.

- [ ] **Step 4: Run branch coverage**

Run:

```bash
env CONDA_NO_PLUGINS=true conda run -n scipy pytest \
  --cov=vercor --cov-branch --cov-report=term-missing:skip-covered \
  tests/ -q -n4 --dist=loadscope --max-worker-restart=0 --tb=short
```

Expected: all tests pass and total branch coverage remains at least 90%.

- [ ] **Step 5: Record exact evidence and re-run documentation/static checks**

Update `PROGRESS.md` with actual test counts, warnings, coverage, and tool
results. Re-run Black, flake8, mypy, compileall, the focused test module, and
`git diff --check` after that documentation edit.

- [ ] **Step 6: Commit final verification evidence**

```bash
git add PROGRESS.md
git commit -m "docs: record bundled output verification"
```

- [ ] **Step 7: Verify repository state**

Run:

```bash
git status --short
git log -8 --oneline
```

Expected: clean worktree with the design, plan, feature, documentation, and
verification commits visible.
