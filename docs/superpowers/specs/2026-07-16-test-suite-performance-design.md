# Test-suite performance optimization design

**Date:** 2026-07-16  
**Status:** Approved direction; implementation pending written-spec review

## Objective

Reduce the wall-clock time of the complete `pytest tests/` gate without changing
production behavior, test selection, assertion strength, meaningful behavioral
coverage, or the 90% branch-aware coverage threshold. The complete gate includes
the repository's artifact-build, installed-distribution, optional-dependency,
JAX transformation, and numerical contracts even where they extend beyond a
strict unit-test boundary.

## Baseline

The clean `refactor` branch at `39178a0` passes 1,256 tests. Two complete serial
runs took 126.77 and 125.04 seconds, for a 125.91-second mean. A complete
coverage run took 134.73 seconds and measured:

- 6,844 of 7,355 executable statements/lines: 93.05%;
- 1,202 of 1,534 branches: 78.36%;
- 90.52% combined branch-aware coverage; and
- 671 of 729 named functions/methods entered: 92.04%, using a documented
  coverage call-arc proxy because the configured coverage tool does not report
  function coverage directly.

The second serial run attributed 112.04 seconds to test calls, 1.35 seconds to
setup, effectively zero to teardown, and the remainder to collection, imports,
and runner overhead. Collection/import cost ranged from 6.53 seconds warm to
13.23 seconds cold.

The leading file totals were 21.52 seconds for distribution boundaries, 16.89
seconds for coupler runtime, 11.09 seconds for optional setup boundaries, 8.46
seconds for flux utilities, and 7.27 seconds for the public API boundary.

## Considered approaches

### 1. Reuse immutable built artifacts across test modules (selected)

Build the VerCOR wheel/source distribution and both plugin wheels once per
serial pytest session, then give every consumer the same immutable artifact
paths. Each installed-distribution test retains its own temporary installation
directory and fresh interpreter probe.

This removes a known duplicate VerCOR wheel build from
`test_v0_4_public_api.py` while preserving the independent build contract in
`test_distribution_boundaries.py`. It is the lowest-risk measured opportunity
because it changes test setup ownership rather than tested behavior.

### 2. Batch fresh-process import probes

Combining optional-dependency scenarios could save repeated Python/JAX startup
time, but it would let earlier imports affect later assertions. Resetting
`sys.modules` is not equivalent to a clean interpreter. This approach is not
selected because process isolation is part of the behavior being protected.

### 3. Share JAX-compiled callables or reduce numerical inputs

JAX and gradient tests account for substantial time, but careless sharing can
hide retracing defects, while smaller grids can remove numerical edge cases.
This remains a later, separately measured opportunity only when the exact
equivalence classes and transformation guarantees can be enumerated first.

## Selected architecture

Move the `built_distributions` fixture from one test module to the root test
configuration and give it session scope. The fixture continues to call the
existing `build_distributions` helper, including all current environment-based
CI artifact reuse and validation. No artifact is modified after construction.

The installed public-boundary test will request this fixture and install its
prebuilt VerCOR wheel into that test's unique `tmp_path`. Its isolated Python
probe, complete public manifest checks, removed-module checks, and installation
assertions remain unchanged. The redundant inline `python -m build` invocation
and its now-unused build-environment setup are removed from that test only.

The distribution-boundary tests continue to request the same fixture. They
still validate that a clean checkout can build the wheel, source distribution,
and plugin wheels, and they retain fresh target directories for every install
or execution probe. Explicit-artifact helper tests continue to construct their
own controlled paths and monkeypatch build behavior independently.

No production module changes are planned for this batch.

## Behavioral equivalence

The rewritten public-boundary test originally protected two behaviors:

1. the repository can produce a wheel; and
2. an installed wheel exposes the exact public API and excludes removed or
   private boundaries.

Behavior 1 remains covered by the shared fixture's real build and the existing
distribution-boundary tests that consume all generated artifacts. Behavior 2
remains in the same public-boundary test with the same fresh installation,
interpreter isolation, and assertions. Only the duplicate execution of behavior
1 is removed.

No test is deleted, merged, skipped, deselected, or marked expected-failure.
No assertion is weakened. Test count and statement, line, branch, combined, and
function-entry coverage are required to remain at least baseline-equivalent.

## Failure handling and isolation

Artifact build failures still fail fixture setup with the original captured
subprocess diagnostics. A missing or incorrectly named artifact still fails
`_existing_distributions`. Install and probe failures remain local to the
requesting test.

Session reuse is safe for serial execution because `BuiltDistributions` is a
frozen dataclass of paths and consumers only read or copy the files. Unique
`tmp_path` installation roots prevent package-state sharing. No environment
variable, import cache, installed tree, database, port, or mutable model state
is shared between tests.

The fixture is worker-local under pytest-xdist. Consequently this batch does
not claim a parallel speedup and does not introduce xdist. Cross-worker artifact
coordination requires a later design because file locking and failure recovery
would add shared mutable filesystem state.

## Implementation and validation sequence

1. Add a focused static performance contract that requires the installed
   public-boundary test to request `built_distributions` and forbids its former
   private build-environment helper. This fails against the baseline and guards
   the artifact-reuse boundary without adding a timing-sensitive assertion.
2. Move the immutable artifact fixture to `tests/conftest.py` and update both
   consumers.
3. Run the focused regression and the two affected test modules.
4. Repeat the affected tests to check order independence and leaked state.
5. Measure focused before/after timings with identical commands.
6. Run the complete serial suite at least twice and compare its mean with the
   125.91-second baseline.
7. Run complete branch coverage and compare every recorded metric.
8. Run Black, strict flake8, mypy, compileall, `git diff --check`, and the
   repository's fast suite.
9. Revert the optimization if it produces no justified timing improvement,
   reduces meaningful coverage, weakens behavior, or introduces instability.

After this batch, remaining bottlenecks will be re-ranked using fresh duration
data. Later batches may address repeated immutable source/AST indexing or
carefully proven JAX setup reuse. Fresh-process import isolation will remain
intact unless a separate behavioral-equivalence design is approved.

## Reporting

The implementation report will record exact commands, environment, focused and
complete before/after timings, all coverage metrics, behavioral-equivalence
evidence, static-quality results, remaining bottlenecks, and the percentage
improvement:

`((baseline runtime - final runtime) / baseline runtime) * 100`

Durable results will be condensed into `PROGRESS.md` without exceeding its
executable 180-line limit. No tag, push, publication, or release action is in
scope.
