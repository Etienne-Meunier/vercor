# VerCOR Progress

This is the bounded orientation log for active development. Detailed history is
preserved in `docs/progress-archive-2026-04-23-to-2026-05-15.md` and
`docs/progress-archive-2026-05-16-to-2026-07-14.md`.

## Current Status

- VerCOR 0.4 cleanup completed locally (2026-07-17): historical 0.3
  distribution evidence, the mutable component test helper, and the mutable
  output test adapter are removed. The output adapter deletion produced the
  required two-module collection RED; CAMulator, JAXGCM, and Veros tests now
  exercise immutable `OutputFrame`/`_OutputAccumulator` accumulation and direct
  NetCDF writes. Cleanup gates include the earlier 34/34 and 94/94 focused
  GREENs, the output 136/136 focus, targeted mypy, 657/657 fast, and 1256/1256
  full suites with only known third-party warnings and no production behavior
  change.
- Controlled pytest parallelization completed locally (2026-07-16) on Python 3.13.13,
  pytest 9.1.1, pytest-xdist 3.8.0, pytest-cov 7.1.0, coverage.py 7.15.0, JAX 0.10.2,
  macOS-26.5.2-arm64-arm-64bit-Mach-O/arm64. The 1,256-test serial baseline mean was 125.91s; artifact reuse was forward-reverted because 29.975s regressed from 29.215s.
  After Task 1's five helper tests, serial wall times were 122.23/128.87/122.05s (mean 124.3833s).
  Initial n2 walls were 86.83/77.58s (mean 82.205s); n4 walls were 62.82/61.68s,
  with 60.47s validation (three-run mean 61.6567s); auto was 62.59/66.48s (mean 64.535s).
  Fixed n4 with `--dist=loadscope --max-worker-restart=0` was selected; no-reorder passed at 55.66s. Saving was 124.3833-61.6567=62.7267s, and `(124.3833-61.6567)/124.3833*100=50.43%`.
  Final collection was 1,262/1,262 passed with zero failed, skipped, xfailed, retried, restarted, crashed, or flaky tests; serial/parallel warnings had identical sources/messages, with only the known Flax collection warning duplicated per worker.
  Serial and parallel coverage were exactly equal: statements 6,844/7,355 (93.05%), branches 1,202/1,534 (78.36%), combined 90.52%, and named-function entries 671/729 (92.04%).
  Fast default/serial passed the same 662 selected tests (600 deselected); complete default/serial and coverage passed 1,262/1,262.
  Black left 244 files unchanged; strict flake8 reported 0; mypy passed 240 files; compileall and `git diff --check` were clean.
  Remaining bottlenecks are distribution builds, JAX/coupler runtime, setup subprocesses, flux tests, and public-API subprocess probes.
  Production behavior, assertions, test selection, coverage thresholds, releases, pushes, and publications did not change.
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
  The frozen 0.3.2 manifest and frozen 0.3 plugin remain historical artifact
  evidence only and are not installed or executed against 0.4.
- Tasks 1-8 plus Task 10 form the complete alpha series. The current API has a six-symbol root,
  protocol-first components, constructor-only coupling, traced physical
  constants, stable route IDs, strict state validation, workflow-planned chunk
  execution, unified output providers, migrated bundled setups/examples, and a
  public-only installed 0.4 plugin.
- Task 8 baseline: focused setup/example/plugin and JAXGCM/Veros selection
  114/114; fast suite 481/481; full suite 1067/1067. Black, strict flake8,
  mypy, compileall, installed plugin/default-slab artifacts, and whitespace
  checks passed. Two third-party `FutureWarning`s were known in the full suite.
- Task 10 documentation contract RED: 6 failed and 1 passed before the 0.4
  review, migration/release documents, version metadata, and archive rewrite.
  Metadata/CI contract RED: 2 failed for version 0.3.2 and the absent plugin
  lane. No production behavior changed.
- Final Task 10 gates: documentation/release contracts 9/9; distribution
  boundaries 16/16; Black 239 files; strict flake8 0; mypy 235 files;
  compileall clean; fast suite 479/479 with 587 deselected; full suite
  1066/1066; branch coverage 90.41%; installed final-artifact wheel/sdist,
  plugin, supplied-artifact, and slab probes 4/4; focused JCM/Veros lanes 9/9;
  output-free JVP/reverse acceptance 3/3. The full and coverage suites emitted
  only the two known third-party FutureWarnings.
- Task 10 controller follow-up closed the incomplete signature sample. RED was
  the absent complete static contract; GREEN freezes and checks all 147
  concrete callable exports from canonical non-root owner manifests plus 55
  public class/protocol methods against source and an isolated installed
  wheel. The focused API/distribution files passed 18/18, fast passed 479/479,
  and full passed 1066/1066 with only the two known third-party warnings.
- Final whole-branch Important findings were closed on 2026-07-14. One private
  normalizer now rejects bare `str`/`bytes` at every audited public name-sequence
  boundary, `RuntimeOptions.model_year_seconds` eagerly requires and
  canonicalizes a finite positive real scalar, and prepared bindings no longer
  delegate private author markers. Primary RED was 37 failed/1 passed; the
  specialist follow-up RED was 5 failed/38 passed; final focused GREEN is
  43/43. A first full run caught and localized one private-helper namespace
  leak before commit; the source and installed-wheel regressions then passed.
- Final-review fix gates: Black 240 files; strict flake8 0; mypy 236 files;
  compileall clean; fast suite 522/522 with 587 deselected; full suite
  1109/1109; branch coverage 90.49%; distribution boundaries 16/16 from fresh
  source-built artifacts; focused JCM/Veros lanes 9/9; output-free JVP/reverse
  acceptance 3/3; whitespace clean. Full and coverage runs emitted only the
  two known third-party `FutureWarning`s. Task 9, version `0.4.0a1`, and release
  publication state remain unchanged.
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
  `198a7e2d7d4873d3550ff3ffe41aa8b6c41ab38e80347b501e6f04e43766db74`;
  and `vercor_compat_plugin_0_3-0.1.0-py3-none-any.whl`
  `740a3fa64f0af5ae18ec497469d53159b52834faf092f8148239bc73c18a2ad4`.
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
  native-v0.4 plugin lanes, metadata-only frozen-0.3 inspection, and a macOS
  installed-plugin smoke. GitHub-hosted jobs have not run locally.
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
