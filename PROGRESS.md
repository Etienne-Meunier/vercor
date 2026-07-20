# VerCOR Progress

This is the bounded orientation log for active development. Detailed history is
preserved in `docs/progress-archive-2026-04-23-to-2026-05-15.md` and
`docs/progress-archive-2026-05-16-to-2026-07-14.md`.

## Current Status

- Installed-artifact CI fixes completed locally (2026-07-20): plugin mypy checks
  source outside `site-packages`; JCM/Veros lanes install both extras required
  to collect their shared test module. TDD and isolated checks passed.
- Final-review alignment completed locally (2026-07-20): removed stale `runner`
  documentation and clarified the pre-stable versioning label after Graphify
  exposed ambiguous generated wording; focused checks passed 49/49.
- Codebase simplification completed locally (2026-07-20): Tasks 1-8 collapsed
  the private grid hierarchy and scalar regridding, removed unread component/data/store/flux
  state, centralized output dimensions and immutable accumulator reconstruction,
  returned topology maps directly, removed the one-use runtime runner, narrowed
  CAMulator initialization to supported named checkpoints, reduced representative
  bilinear-interpolator PyTree leaves from 28 to 26, and changed each Veros forcing
  update from four state copies/unlocks to one. Task RED/GREEN evidence was 3
  failures then 45 passes; 7 then 66; 2 then 72; 7 then 108; 2 then 107; 2 then
  90; 2 then 28 (46 final bilinear/interpolator-regrid passes); and 1 then 45
  (65 final Veros/physics passes), respectively. The final 14-file subsystem
  focus passed 324/324 with four Flax warning instances. Fast passed 638/638 of
  1,235 with six known third-party warning instances: four Flax/JAX-effect
  deprecations and two xarray/NumPy-shape deprecations. Full and coverage each
  passed 1,235/1,235 with eight known third-party instances: those six plus one
  JAX scatter-cast future warning and one JCM/xarray merge-default future warning.
  Branch coverage was 90.71% across 7,265 statements and 1,522 branches. Black
  left 234 files unchanged and emitted its known Python 3.13/configured Python
  3.15 safety-parse advisory; final strict flake8 was 0, mypy passed 234 source
  files, compileall was silent, and whitespace checks were clean. Release-gate
  review also removed one stale Task 2 `Any` import and applied the repository's
  typing-only `cast(object, ...)` pattern to two Task 3 `PyTreeDef` comparisons;
  the affected static and focused gates were rerun. The Task 7 plan's
  `tests/test_gradients.py` path was stale because that file does not exist;
  reverse-gradient coverage remains in
  `tests/test_bilinear_rectilinear_interpolator.py`, and the explicit
  flatten/unflatten, JIT, JVP, and reverse-gradient probe passed. Incremental
  Graphify completed from the direct `scipy` interpreter after the sandboxed
  attempt was denied: 82/82 uncached code files produced 4,281 nodes, 12,503
  edges, and 175 communities. Relative to the saved graph it added 814 nodes
  and 884 edges and removed 133 nodes and 244 edges. Graphify warned that the
  public-signatures JSON produced no AST nodes; the canonical HTML, JSON,
  report, labels, and manifest were refreshed.
- Period-average window-start identity completed locally (2026-07-17): filenames and NetCDF times now use each schema's actual start.
  Tests cover partial/subsequent periods, mixed cadences, and Gregorian/no-leap/360-day clocks while preserving calendar ISO formats,
  post-step provider times, means, and incomplete-period behavior. TDD RED was 4/4; clarified GREEN was 4/4; output focus was 25/25.
  Black reformatted 2 files and left 234 unchanged with its Python 3.13/target-3.15 advisory; flake8 was 0, mypy passed 236 files,
  and compileall was clean. Fast passed 636/636 with six third-party warnings; full and coverage passed 1224/1224 with eight.
  Coverage was 90.50% across 7,361 statements and 1,538 branches.
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
