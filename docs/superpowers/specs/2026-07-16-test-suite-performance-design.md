# Test-suite performance optimization design

**Date:** 2026-07-16
**Status:** Attempted, rejected by the timing gate, and forward-reverted

## Historical objective

Reduce the wall-clock time of the complete `pytest tests/` gate without changing
production behavior, test selection, assertion strength, meaningful behavioral
coverage, or the 90% branch-aware coverage threshold. The complete gate includes
the repository's artifact-build, installed-distribution, optional-dependency,
JAX transformation, and numerical contracts even where they extend beyond a
strict unit-test boundary.

## Outcome

The artifact-reuse experiment was implemented in `20ac416`, but its required
focused two-run wall-time mean increased from 29.215s to 29.975s. An alternating
archived-base/current comparison also favored the base in wall time while showing
essentially unchanged user CPU. The experiment therefore triggered the planned
failure path and was forward-reverted in `242bbe7`; the three test files match
pre-experiment commit `0d86341`. Independent distribution builds remain the
current architecture, and no test-suite speedup is claimed.

## Focused timing record

The original samples used the direct SciPy-environment interpreter and this
command, with `before` or `after` and sample number substituted in the output
path:

```bash
/usr/bin/time -p -o /private/tmp/vercor-artifact-focused-<before-or-after>-<n>.time \
  /Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest \
  tests/test_distribution_boundaries.py tests/test_v0_4_public_api.py \
  --durations=15
```

| Configuration/sample | Tests | Real (s) | User (s) | Sys (s) | Installed-boundary call (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base `0d86341`, 1 | 103 | 29.83 | 24.87 | 3.63 | 1.28 |
| Base `0d86341`, 2 | 103 | 28.60 | 24.35 | 3.39 | 1.28 |
| Attempt `20ac416`, 1 | 104 | 29.73 | 25.03 | 3.65 | 1.02 |
| Attempt `20ac416`, 2 | 104 | 30.22 | 25.08 | 3.74 | 1.06 |

The dominant public-plugin probe was 12.60s and 13.11s across the attempted
pair; its per-sample association and the base-pair plugin durations were not
retained, so they are not reconstructed here.

The controller then alternated the same pytest command between the archived
checkout `/private/tmp/vercor-perf-base-0d86341-20260716` and the working
checkout, recording `/private/tmp/vercor-alt-<base-or-current>-<n>.time` and
the corresponding `.log` file:

```bash
(cd <archived-base-or-working-checkout> && \
  /usr/bin/time -p -o /private/tmp/vercor-alt-<base-or-current>-<n>.time \
  /Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest \
  tests/test_distribution_boundaries.py tests/test_v0_4_public_api.py \
  --durations=15 > /private/tmp/vercor-alt-<base-or-current>-<n>.log)
```

| Alternating sample | Tests | Real (s) | User (s) | Sys (s) | Target call (s) | Plugin call (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base 1 | 103 | 30.13 | 24.96 | 3.66 | 1.27 | 13.23 |
| Attempt 1 | 104 | 33.10 | 25.06 | 4.19 | 1.11 | 14.12 |
| Base 2 | 103 | 28.85 | 24.69 | 3.42 | 1.27 | 12.51 |
| Attempt 2 | 104 | 28.99 | 24.53 | 3.45 | 1.01 | 12.74 |
| Base mean | 103 | 29.490 | 24.825 | 3.540 | 1.27 | 12.87 |
| Attempt mean | 104 | 31.045 | 24.795 | 3.820 | 1.06 | 13.43 |

The target call reduction did not produce an aggregate improvement. User CPU
was effectively unchanged, and the wall-time difference tracked variance in
the much larger plugin probe, so the acceptance gate remained failed.

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

### 1. Reuse immutable built artifacts across test modules (rejected experiment)

Build the VerCOR wheel/source distribution and both plugin wheels once per
serial pytest session, then give every consumer the same immutable artifact
paths. Each installed-distribution test retains its own temporary installation
directory and fresh interpreter probe.

This was selected as an experiment intended to remove a known duplicate VerCOR
wheel build from
`test_v0_4_public_api.py` while preserving the independent build contract in
`test_distribution_boundaries.py`. It appeared to be the lowest-risk measured
opportunity because it changed test setup ownership rather than tested
behavior.

### 2. Batch fresh-process import probes

Combining optional-dependency scenarios could save repeated Python/JAX startup
time, but it would let earlier imports affect later assertions. Resetting
`sys.modules` is not equivalent to a clean interpreter. This approach was not
selected because process isolation was part of the behavior being protected.

### 3. Share JAX-compiled callables or reduce numerical inputs

JAX and gradient tests account for substantial time, but careless sharing can
hide retracing defects, while smaller grids can remove numerical edge cases.
This was left for a separate design requiring enumerated equivalence classes
and transformation guarantees; this experiment selected no such follow-up.

## Rejected experimental architecture

The experiment moved the `built_distributions` fixture from one test module to
the root test configuration and gave it session scope. The fixture continued
to call the existing `build_distributions` helper, including all then-current
environment-based CI artifact reuse and validation. No artifact was modified
after construction.

The installed public-boundary test requested this fixture and installed its
prebuilt VerCOR wheel into that test's unique `tmp_path`. Its isolated Python
probe, complete public manifest checks, removed-module checks, and installation
assertions remained unchanged. The redundant inline `python -m build`
invocation and its now-unused build-environment setup were removed from that
test only during the experiment.

The distribution-boundary tests requested the same fixture. They still
validated that a clean checkout could build the wheel, source distribution,
and plugin wheels, and retained fresh target directories for every install or
execution probe. Explicit-artifact helper tests continued to construct their
own controlled paths and monkeypatch build behavior independently.

No production module was changed. This architecture is retained only as a
record of the rejected experiment; it is not the current direction.

## Behavioral equivalence

The rewritten public-boundary test originally protected two behaviors:

1. the repository can produce a wheel; and
2. an installed wheel exposes the exact public API and excludes removed or
   private boundaries.

During the experiment, behavior 1 remained covered by the shared fixture's real
build and the existing distribution-boundary tests that consumed all generated
artifacts. Behavior 2 remained in the same public-boundary test with the same
fresh installation, interpreter isolation, and assertions. Only the duplicate
execution of behavior 1 was removed.

No test was deleted, merged, skipped, deselected, or marked expected-failure,
and no assertion was weakened during the experiment. The intended success path
required baseline-equivalent test count and coverage, but it was not reached.

## Failure handling and isolation

The experimental failure model kept artifact build failures in fixture setup
with the original captured subprocess diagnostics. A missing or incorrectly
named artifact still failed `_existing_distributions`. Install and probe
failures remained local to the requesting test.

Session reuse was considered safe for serial execution because
`BuiltDistributions` is a frozen dataclass of paths and consumers only read or
copy the files. Unique `tmp_path` installation roots prevented package-state
sharing. No environment variable, import cache, installed tree, database,
port, or mutable model state was shared between tests.

The fixture would have remained worker-local under pytest-xdist. The experiment
therefore made no parallel-speedup claim and did not introduce xdist.

## Planned implementation and validation sequence

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

Execution reached the failure condition in step 9 after the focused timing gate
failed. The success-path full-suite, coverage, and repository-wide quality
measurements were not run for a retained optimization. No subsequent
optimization direction was selected by this experiment.

## Reporting policy

The original plan required exact commands, environment, focused and complete
before/after timings, coverage metrics, behavioral-equivalence evidence,
static-quality results, remaining bottlenecks, and percentage improvement:

`((baseline runtime - final runtime) / baseline runtime) * 100`

Because the failure path triggered before those success-path gates, the durable
record contains the focused rejection evidence and explicitly identifies the
unrun gates. `PROGRESS.md` remains within its executable 180-line limit. No tag,
push, publication, or release action occurred.
