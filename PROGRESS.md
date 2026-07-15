# VerCOR Progress

This is the bounded orientation log for active development. Detailed history is
preserved in `docs/progress-archive-2026-04-23-to-2026-05-15.md` and
`docs/progress-archive-2026-05-16-to-2026-07-14.md`.

## Current Status

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
- Prior retained candidate bundles and their hashes are stale because their
  metadata predates the pre-1.0 version correction. Fresh artifact evidence is
  intentionally deferred to the next release-verification task. JCM 1.1.1 and
  Veros 1.6.2 remain the installed versions used for optional focused gates.

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
