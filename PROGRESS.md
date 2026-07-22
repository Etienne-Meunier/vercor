# VerCOR Progress

This is the bounded orientation log for active development. Detailed history is
preserved in `docs/progress-archive-2026-04-23-to-2026-05-15.md`,
`docs/progress-archive-2026-05-16-to-2026-07-14.md`, and
`docs/progress-archive-2026-07-22.md`.

## Current Status

- Bundled slab/data step-period output completed locally (2026-07-22): one private setup declaration enables the existing generic output provider across all slab factories, shared time-interpolated data factories, and the direct JCM land data factory. Period files contain declared outputs only; custom components remain opt-in and `output=None` remains I/O-free. Focused tests passed 9/9 with no warnings; fast passed 662/662 with four Flax/JAX third-party deprecation warnings; full and branch-coverage runs exited 0 with those four warnings plus one JCM/xarray future warning, and fresh collection reported 1,280 tests. Branch coverage was 91.08% across 7,383 statements and 1,556 branches. Black left 236 files unchanged with its known Python 3.13/target-Python-3.15 safety-parse advisory; strict flake8 reported 0, mypy passed 236 source files, compileall passed, and whitespace checks passed. Independent final review remains assigned to the controller.
- Veros component-scoped linear-solver caching completed locally (2026-07-22): the setup-created solver is reused across copy-owned native states with exception-safe temporary cache binding, setup-key release, owner isolation, and validated Veros >=1.6.2,<1.7 cache ABI. TDD RED/GREEN, real-cache, external-component, fast, full, coverage, formatting, lint, typing, compile, and whitespace gates passed. Public APIs, native payloads, and numerics stay unchanged.
- Final-review corrections completed locally (2026-07-21): lifecycle identity, CAMulator calendar rejection, and API review evidence are complete. Focused, fast, full, static, prior 91.01% coverage, and artifact gates passed. Deferred private cleanup remains runtime-schema and output-owner unification.
- Installed plugin evidence and author guidance completed locally (2026-07-21): the fixture accepts `0.4.0a1` through `<0.5`, its non-default configuration produces independently observable JAX, host, exchange, and backend results, and CI now resolves the plugin against the installed VerCOR wheel. The new public-only author guide executes configuration, payload, regridding, topology, workflow/backend, output, and fake-testing examples in order. The reviewed guide composes through a plugin-owned factory, samples payload state, and release instructions preserve normal plugin dependency resolution.
- Stable extension and factory typing completed locally (2026-07-21):
  `RegridderFactory` is one runtime-checkable public protocol, the plugin
  fixture may import only the six-symbol root/stable extension tier, and the
  review/design distinguish that tier from retained alpha inventory and JAX
  integration hooks. RED was the expected `TYPE_CHECKING` source-boundary and
  extension-import failures; focused GREEN and strict plugin mypy passed, as
  did the full suite. The user-approved follow-up narrowed the factory protocol
  to its two required grids, restoring built-in/default factory compatibility.
  No public export was removed.
- CAMulator atmosphere payload ownership completed locally (2026-07-21): setup seeds a frozen native payload; functional stepping clones and advances payload-owned model state, cursor, forecast hour, and predictions; providers and snapshots sample only context payloads. TDD RED was 5 expected failures; exact focus passed 18/18, complete focus passed 90/90, static gates passed, and full passed 1,242/1,242 with five known third-party warnings.
- Explicit CAMulator forcing alignment and functional land cursor completed locally (2026-07-21): `strict` now rejects coupler/forcing start mismatches, `forcing_start` opts in without warnings, typed configuration carries the policy, the immutable cursor advances functionally, and land owns its cursor in runtime payload; TDD RED was 4 expected failures and focused GREEN passed all selected tests.
- Veros payload ownership completed locally (2026-07-21): setup seeds the native payload; runtime returns `StepResult` without mutating setup resources; provider and snapshot output read context payloads. RED was 4 transition/provider plus 2 setup failures; focused GREEN passed 79/79 (8/8 non-fast ownership), and full passed 1,242/1,242 with five known third-party warnings.
- Static component identity through setup completed locally (2026-07-21): setup cannot replace declared name, grid, or spec; the adapter revalidates after the hook before examining its result. TDD RED was 3/3 expected failures; focused GREEN passed 42/42.
- CI artifact and NetCDF backend stability completed locally (2026-07-21): the quality job now reuses the build-once artifact bundle, forcing fixtures explicitly use h5netcdf, and JCM packaged input loading temporarily prefers h5netcdf without leaking xarray configuration. After the formatting follow-up, the static gates passed: Black left 234 files unchanged (with its known Python 3.13/target-Python-3.15 safety-parse advisory), flake8 was 0, mypy passed 234 source files, and compileall/whitespace checks were clean. The exact direct-`scipy`-interpreter workflow contract command with `-n0` passed 1/1. Fast passed 638/638 with four Flax/JAX-effect deprecation warnings; full and branch coverage passed 1,235/1,235 without NetCDF/HDF failures, with those four warnings plus one JCM/xarray merge-default future warning. Branch coverage was 90.75% across 7,287 statements and 1,524 branches.
- JAXGCM runtime dtype warning fixed locally (2026-07-21): the adapter applies the runtime-owned `DTypePolicy` before pressure and altitude calculations, preventing `float64` promotion and the incompatible JAX scatter into `float32`; the warning-as-error and mapped-field dtype regression had 1/1 RED, focused GREEN passed 11/11, fast/full passed 638/638 and 1,235/1,235 without the scatter warning, Black left 234 files unchanged, flake8 was 0, mypy passed 234 source files, and whitespace checks were clean.
- CI fixes completed locally (2026-07-20): plugin smoke uses installed artifacts; strict mypy checks a copied installed plugin outside the checkout with Python safe-path isolation. TDD and isolated checks passed.
- Final-review alignment completed locally (2026-07-20): removed stale `runner`
  documentation and clarified the pre-stable versioning label after Graphify
  exposed ambiguous generated wording; focused checks passed 49/49.
- Codebase simplification completed locally (2026-07-20): Tasks 1-8 reduced
  private grid/regridding/runtime/component state and focused setup paths while
  preserving public behavior; all task focuses, fast/full/coverage, and static
  gates passed. Detailed evidence is retained in the task reports and archive.
- Period-average window-start identity completed locally (2026-07-17): filenames and NetCDF times use each schema's actual start across partial/subsequent periods, mixed cadences, and Gregorian/no-leap/360-day clocks while preserving calendar ISO formats, post-step provider times, means, and incomplete-period behavior. TDD RED/GREEN was 4/4; output focus 25/25.
  Black/flake8/mypy/compileall passed; fast passed 636/636 and full/coverage 1224/1224 (90.50% across 7,361 statements and 1,538 branches), with known third-party warnings.
- VerCOR 0.4 deprecation cleanup completed locally (2026-07-17) in commits `f82588f` through `04e6f45`. Obsolete evidence,
  mutable helpers, absence-only guards, and old adapter tests are gone; positive contracts cover public ownership, lifecycle,
  immutable state, artifacts, output, numerics, JIT, and gradients. Supported foreign-state, calendar, transform, lazy-import,
  payload-copy, and offline-artifact behavior remains. Focused gates passed 34/34, 94/94, 136/136, and 301/301; docs 175/175;
  fast 632/632; full/coverage 1220/1220 at 90.49%. Black, flake8, mypy, compileall, and whitespace passed.
- Controlled pytest parallelization completed locally (2026-07-16): fixed n4
  loadscope reduced the measured 124.38s serial mean to 61.66s (50.43%) while
  preserving selection, warnings, and 90.52% combined coverage. Fast/full,
  Black, flake8, mypy, compileall, and whitespace gates passed; production and
  release behavior did not change.
- Calendar-owned runtime year metadata completed locally (2026-07-15):
  commits `9ade80c` and `a9b079c`. Runtime forcing metadata now derives year
  type and duration per timestamp; the duplicate runtime owner and private
  mapper are gone, and the common-year stdlib `datetime` boundary is fixed.
  Focused GREEN was 136/136; mutation/restoration was 1 failure then 1 pass;
  final fast was 660/660 with 596 deselected and full was 1256/1256. Earlier
  implementation coverage was 90.52% across 7,355 statements and 1,534
  branches; Black, flake8, mypy, compileall, and whitespace gates passed.
- Matcher-level versioning review completed locally (2026-07-15): ordered,
  explicit repository-release contexts replace the broad proximity heuristic.
  RED isolated 4 incorrect cases among 17 parameters; matcher GREEN is 17/17
  and the complete policy/architecture focus is 28/28. External dependency,
  plugin, action, schema, and numerical labels remain accepted.
- Versioning-review follow-up completed locally (2026-07-15): the repository
  policy now rejects contextual major-series shorthand without matching
  numerical values or external versions. RED reported exactly 10 remaining
  repository-owned labels; the corrected policy/documentation focus passes
  11/11 and the progress-archive checksum is refreshed.
- Completed locally (2026-07-15): corrected the unsupervised historical
  release labels to the approved pre-1.0 sequence, ending at `0.4.0a1`. The
  policy and architecture focus passes 30/30 and the fast suite passes 524/524.
  The Conda launcher panic occurred before pytest, so actual checks used the
  direct `scipy` environment interpreter. The approved repository-wide scope is recorded in
  `docs/superpowers/specs/2026-07-15-vercor-versioning-design.md`; the execution
  sequence is in `docs/superpowers/plans/2026-07-15-vercor-versioning.md`. No
  tag, push, publication, or Git-history rewrite is authorized.
- VerCOR 0.4.0a1 Task 10 candidate preparation was completed and committed in
  repository history on 2026-07-14. Tagging, pushing, and publication remain
  intentionally unperformed pending separate authority.
  Task 9 was explicitly skipped: no legacy adapter namespace is implemented.
- Tasks 1-8 plus Task 10 form the complete alpha series. The current API has a six-symbol root,
  protocol-first components, constructor-only coupling, traced physical
  constants, stable route IDs, strict state validation, workflow-planned chunk
  execution, unified output providers, migrated bundled setups/examples, and a
  public-only installed 0.4 plugin.
- VerCOR 0.4.0a1 release verification completed locally (2026-07-15) from
  build HEAD `31e803c06a4e65e8e72ee77937b056eac540eb44`. Black warned Python 3.13
  cannot perform its safety parse for configured Python 3.15, while exit
  remained 0 and 242 files were unchanged; strict flake8 reported 0; mypy
  passed 238 source files; compileall and whitespace checks were clean.
  The fast suite passed 543/543; the full and branch-coverage suites passed
  1139/1139; coverage was 90.51% (7,352 statements and 1,532 branches). The
  full and coverage runs emitted five third-party warning instances: one Flax
  JAX-effect deprecation, one JAX scatter-cast future warning, one JCM/xarray
  merge-default future warning, and two xarray NumPy-shape deprecations. The
  optional JCM/Veros focus passed 9/9 with only the Flax warning; output-free
  JVP/reverse differentiation passed 3/3; supplied-artifact boundaries passed
  16/16. Fresh offline no-isolation builds are in
  `/private/tmp/vercor-0.4.0a1-dist/` with SHA-256 values:
  `vercor-0.4.0a1-py3-none-any.whl`
  `a713f10c3722145d1dd0e0886c266e264d098dc7f30276b99bb027fdc246bff7`;
  `vercor-0.4.0a1.tar.gz`
  `119717648950a04d89fe28a2522a2c6ae5fc699d8725ae0cdc788691c6c529a2`;
  `vercor_public_plugin-0.1.0-py3-none-any.whl`
  `198a7e2d7d4873d3550ff3ffe41aa8b6c41ab38e80347b501e6f04e43766db74`.
  JCM 1.1.1 and Veros 1.6.2 remain the installed optional-model versions.
  Tag, push, publication, and upload remain unperformed.
- Post-review version-policy hardening completed locally (2026-07-15): the
  ownership matrix preserves qualified external/independent identifiers while
  rejecting exact and shorthand VerCOR labels. Policy passed 141/141,
  policy/architecture passed 161/161, and the full suite passed 1261/1261 with
  the same five third-party warning instances. Independent final review found
  no Critical, Important, or Minor issues. Release artifacts and hashes above
  remain unchanged; no tag, push, publication, or upload was performed.

## Implemented 0.4 Architecture

- `vercor.__all__` is exactly `Clock`, `Coupler`, `Exchange`,
  `RectilinearGrid`, `RunState`, and `RuntimeOptions`.
- `Component` is structural. `ComponentSpec` owns fields, lifecycle, execution,
  transfer, and output; `CallableComponent` and `DataComponent` are the only
  convenience adapters.
- `PhysicalConstants` is the frozen traced PyTree; `RuntimeOptions.dtype` is
  the sole precision policy.
- `Coupler(...)` owns complete immutable assembly. Reconfiguration constructs a
  new coupler.
- Exchange and topology identity is the stable route ID. Ambiguous target-field
  fan-in is rejected.
- Workflows produce exact plans; the core owns chunks and validates every
  backend driver call and returned state.
- `RunState` exposes only component views and immutable field replacement.
- One output coordinator owns all enabled provider selection, accumulation,
  cadence, host transfer, paths, period files, final fields, and snapshots.
  `output=None` performs no I/O and remains differentiable.
- JCM, Veros, and CAMulator imports remain lazy. CAMulator is not installed or
  pinned.

## Release Candidate Handoff

- The executable review validates the exact eight sections, all canonical
  public manifests, central/root signatures, all 119 non-public modules,
  runnable README/migration snippets, archive SHA-256, Task 9 absence, and
  release metadata.
- CI encodes Python 3.12/3.13 base/JCM/Veros artifact lanes, Python 3.12/3.13
  native-v0.4 plugin lanes, and a macOS installed-plugin smoke. GitHub-hosted
  jobs have not run locally.
- The Task 10 documentation/release commit is present in repository history.
  Do not tag, push, or publish without separate authority.

## Durable Constraints

- Never add numerical fudge factors; trace discrepancies to their first source.
- Write behavior/contract tests before implementation changes.
- Preserve exact public owner manifests and keep primary 0.4 modules alias-free.
- Keep output opt-in and optional-model imports lazy.
- No registry, entry-point discovery, Pydantic, fan-in reducer, public prepared
  graph, fractional subcycling, or CAMulator dependency pin.
- Do not tag, push, publish, or create a release without separate authority.
- The Conda launcher can panic through `conda-rattler`; use the direct `scipy`
  environment executable when that occurs.

## Validation Policy

Use concise pytest output. Run focused tests while iterating, then Black,
strict flake8, mypy, compileall, fast and full pytest, branch coverage, build,
installed wheel/sdist/plugin smokes, and `git diff --check` before a release
candidate commit. Put detailed command evidence in the active task report and
archive only durable outcomes here.
