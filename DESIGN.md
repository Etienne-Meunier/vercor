# VerCOR: Design Specification

A fully differentiable coupler in JAX for different Earth system models written in JAX.

---

## 1. Goals and Non-Goals

### Goals

- **End-to-end differentiable**: exact gradients of any output with respect to any
  coupled models parameters, via JAX reverse-mode AD.
- **No global arrays or mutable state**: all data is passed explicitly via function arguments and return values.
- **Accelerated**: pure functions are JIT-compiled with `jax.jit` decorator, 
single device parallelism via `jax.vmap` where applicable, use `jax.lax.scan` for iterative methods/solvers, 
use `jax.lax.fori_loop`, `jax.numpy.where` and  `jax.lax.cond` etc. to avoid Python control flow.
- **Modularity**: clean, modular code structure for easy maintenance and extension.
- **Documentation**: comprehensive docstrings and usage examples.
- **Testing**: extensive unit tests for correctness and regression prevention.

## 2. Architecture Overview

### Modular design

The codebase is organized into modules corresponding to physical, numerical and different coupled models/components.

Interpolation, exchangers, grids, model components, output routines etc. are all separate modules with well-defined interfaces. 
This allows different agents to work on different components in parallel and makes testing easier.

The output module handles all data saving and logging, ensuring a clean separation between computation and I/O.

### Pure functional style

All functions are pure, jitted with `jax.jit` decorator and stateless. No mutable global state. No side effects.
This is critical for JAX compatibility and makes reasoning about the code easier.
Each function takes explicit inputs and returns explicit outputs, which can be easily tested and debugged.

### Compile cache hits and safe buffer donation

To ensure good performance, we need to design the code to maximize JIT cache hits and enable safe buffer donation.
This means avoiding dynamic shapes, using static arguments for control parameters, and ensuring that arrays are not mutated in-place.

**Keep JIT compile keys stable (avoid surprise recompiles):**
- Define model/containers at module top‑level so identities don’t change between runs.
- Mark non‑array metadata as static so it isn’t traced.
- Keep argument pytrees small and consistent (e.g., NamedTuple with fixed fields).
- If you pass constants/flags, make them static args.

**Anti‑pattern to avoid:** constructing fresh containers every call with changing non‑array fields (e.g., dicts with varying keys or dataclasses whose __eq__ changes) → recompiles.

**Donate buffers safely (lower peak memory, speed up):**
Donation lets XLA reuse input buffers for outputs.
- Rule of thumb: donate only arrays you won’t read again after the call.
- Practical boundary: donate at the outer step (not deep internals) so the contract is easy to respect.

**Small patterns that add up**
1) Stable run‑state wrapper
Keep “run‑level” scalars (step, time, dt) in a fixed NamedTuple; keep big arrays (model params) in plain pytrees (tuples/dicts) with stable keys.
2) Keep non‑array metadata out of traces
3) Reduce variant explosion: prefer fixed‑shape boundary tuples over dicts whose keys appear/disappear:
4) Donation audit at the callsite.
5) Deterministic RNG: split once per step at the boundary; don’t split inside inner kernels (helps compile stability).

### Input / Output

All I/O is handled by a dedicated module that reads/writes from/to disk.
The core computational modules are completely decoupled from file formats and storage details.
This allows us to easily swap out the I/O layer if needed, and keeps the core logic clean and focused on the physics.

The output is done in a structured format, such as NetCDF, HDF5, that can be easily read by visualization tools and post-processing scripts.

Model restart files are supported and written in compact HDF5 format using `h5py`.

Current example output snapshots are also written in HDF5. NetCDF output for broader
VerCOR workflows remains future work.

### Data flow: PyTree-based result objects

Create a JAX helper module for common PyTree utilities & classes (to be reused by many modules),
such as flattening, tree mapping, etc.

Every module returns a frozen dataclass (registered as a JAX PyTree) containing
arrays and objects. 

No mutable state. No side effects.

```python
@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class RectilinearGrid:
    longitude: Array
    latitude: Array
    longitude_edges: Array
    latitude_edges: Array
    binary_mask: Array
    ...
```

### Public and runtime API boundary

VerCOR intentionally separates user-facing orchestration objects from the
immutable runtime containers used during traced integration.

- Public orchestration API: `Coupler`, `Exchange`, `RunSequence`, `Clock`,
  grids, regridders, and bundled concrete components are the objects users
  compose when configuring a coupled run.
- Component-author API: `Component`, `DataComponent`, and
  `HostRuntimeComponent` are the stable extension points. All custom adapters
  seed setup-time fields on `Component.data` and call the base constructor so
  `name`, `grid`, `data`, and `settings` are available during initialization,
  execution, and finalization. `Component.data` is a grid-field store, not a
  general metadata store: all entries must use one of the canonical layouts
  `(nLat, nLon)`, `(nTime, nLat, nLon)`, `(nLev, nLat, nLon)`, or
  `(nTime, nLev, nLat, nLon)`. Setup and runtime-state creation validate this
  contract before traced execution, and non-grid metadata such as hybrid-level
  coefficients belongs on component attributes or runtime payloads. Use
  `Component` for differentiable active models and implement
  `step_runtime_state()`. Use `DataComponent` for forcing/static data adapters
  that intentionally keep the shared no-op runtime step and do not create
  plotting-only runtime fields. Derived diagnostics, such as a combined land/sea
  surface temperature used only for plots, belong in diagnostics or examples.
  Use `HostRuntimeComponent` for non-differentiable adapters and implement
  `step_host_runtime_state()`; host-backed adapters must run through
  `Coupler.run()` so VerCOR can select the Python host runtime path. Optional
  hooks include `initialize()`, `create_runtime_payload()`,
  `prefill_runtime_state_fields()`, and `validate_runtime_state()`.
- Internal runtime API: the `vercor.runtime` package owns
  `RuntimeFieldStore`, `RuntimeComponentState`, `RuntimeCouplerState`, runtime
  contexts, dispatch contexts, and runtime helper functions. These containers
  carry immutable arrays and static metadata through JAX tracing. They are
  required for differentiability and stable scan carry structure, but they are
  not exported from the package top level.

### Logging across JAX runtime transforms

The coupler logger is callback-backed through `jax.debug.callback`, so runtime
hooks can emit diagnostics inside `jax.lax.scan`, `jax.jit`, and automatic
differentiation transforms. Coupler logging levels are configured at
instantiation with `Coupler(..., log_level=...)`; disabled levels are filtered
before callbacks enter the traced graph. Runtime hooks should pass traced values
as logger arguments, for example `logger.info("Mean SST: {}", jnp.mean(sst))`,
instead of converting tracers with `float(...)` or `int(...)`.

---

## 3. Module Specifications

### 3.1 Constants and Parameters

**File**: `constants.py`

For physical constants, such as gravitational acceleration, gas constant, etc.

**File**: `parameters.py`

For runtime parameters, such as coupled run identifier, precision, time interpolation type,
type of year (leap, noleap, 360day), etc.

Two parameter containers:

```python
@dataclass(frozen=True)
class PhysicsParameters:
    """All fields are JAX-traceable floats."""
    # Scalars
    gravity: float
    rhoAir: float
    rgas: float
    latvap: float
    zref: float
    mwdair: float
    ...


@dataclass(frozen=True)
class ControlParameters:
    """Control parameters. NOT traced by JAX (static)."""
    get_field_time_slice: bool
    apply_time_interpolation: bool
    enable_x64: bool
    identifier: str
    ...
```

The split between `PhysicsParameters` (traced) and `ControlParameters` (static) is critical:
JAX traces through `PhysicsParameters` for AD, while `ControlParameters` controls array
shapes and solver settings that must be compile-time constants.

## 4. Validation and Testing

### Test design philosophy

The test harness is the most important part of this project. Without high-quality
tests, autonomous agents will solve the wrong problem.

1. **Tests must be nearly perfect.** Agents will optimize for whatever the tests
   measure. If a test is wrong or has loose tolerances, agents will produce code
   that passes the bad test but gives wrong physics. Invest more time in the test
   harness than in the code it tests.

2. **Tests must give concise, actionable feedback.** Print the max relative error
   and where it occurs, not full arrays. Pre-compute aggregate statistics.
   Log details to files, not stdout, to avoid context window pollution.

3. **Tests must be fast by default.** Every test file supports a `--fast` mode
   (~10% subsample) for rapid iteration. Full validation runs before commits.

4. **Tests must decompose monolithic tasks.** Test sub-components independently:
   - Regridding with mock meshes and fields
   - Different clock functionalities and options
   - Coupler stepping with mock models and fields
   - Fluxes computations with mock meshes and fields
   - Exchanges of fields between models with mock models, meshes and fields
   - Input / Output
   - etc.
   This lets different agents work on different subsystems.

5. **Tests must enable bisection.** When solution disagrees, we need to find the
   first module in the pipeline that diverges from original code. The test suite
   should make this easy by testing every intermediate quantity, not just
   the final output. This is the "oracle bisection" pattern.

### Test hierarchy

We use a layered testing approach, from unit tests to full pipeline validation.

**level 1**: Unit tests (fast)

**level 2**: Module tests.
For each module, pre-generate reference data and check agreement.

**level 3**: Gradient tests

For each module, verify that AD gradients match finite-difference gradients.

**level 5**: End-to-end integration tests (from examples in `examples/`)

---

## 5. Performance Strategy

### JIT compilation

The entire `run_simulation()` function should be JIT-compiled:

Since `ControlParameters` is static (controls array shapes), it should be passed via
`static_argnums` or as a `static_field` in an Equinox module.

First call will be slow (~30-60s for XLA compilation). Subsequent calls with the
same shapes will be fast.
