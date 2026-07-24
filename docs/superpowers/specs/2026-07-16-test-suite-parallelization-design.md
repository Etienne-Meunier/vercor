# Controlled pytest parallelization design

**Date:** 2026-07-16
**Status:** Approved direction; implementation pending written-spec review

## Objective

Reduce the wall-clock time of the complete `pytest tests/` gate through
controlled local-process parallelism without changing production behavior,
test selection, assertion strength, coverage thresholds, or meaningful
behavioral guarantees.

This is a new optimization batch. The earlier cross-module artifact-reuse
experiment was rejected by its aggregate timing gate and forward-reverted. It
is not a prerequisite and must not be reintroduced.

## Baseline and acceptance metrics

The unchanged serial suite contains 1,256 tests. Its pre-optimization wall
times were 126.77 and 125.04 seconds, with a 125.91-second mean. A fresh
post-revert verification passed 1,256 tests in 122.13 seconds of pytest time and
128.68 seconds wall time; this variation is baseline noise, not a speedup.

Coverage is the immutable comparison baseline:

- statement/line coverage: 6,844/7,355, or 93.05%;
- branch coverage: 1,202/1,534, or 78.36%;
- combined branch-aware coverage: 90.52%; and
- named-function entry proxy: 671/729, or 92.04%.

A parallel configuration is accepted only if complete repeated runs pass all
1,256 existing tests plus any new configuration-contract tests, produce no new
skip/xfail/retry behavior, preserve or improve every coverage metric, and
reduce the repeated complete-suite mean relative to a contemporaneous serial
control.

## Supported pytest-xdist contracts

The design relies only on documented pytest-xdist behavior:

- [`--dist=loadscope`](https://pytest-xdist.readthedocs.io/en/stable/distribution.html)
  groups test functions by module, keeping module fixtures in one worker;
- `-n0` disables distributed execution and remains the serial escape hatch;
- workers expose `PYTEST_XDIST_WORKER` and `PYTEST_XDIST_WORKER_COUNT` for
  per-worker resource isolation; and
- pytest-cov documents
  [xdist support](https://pytest-cov.readthedocs.io/en/stable/readme.html) and
  combines worker coverage.

Use `pytest-xdist>=3.7`, the first documented release with Python 3.13 support.
Add it to both `test` and `dev` optional dependencies. Do not add an unrelated
parallelism, retry, or randomization package.

## Considered approaches

### 1. Controlled pytest-xdist execution (selected)

Benchmark two, four, and automatic worker counts using `loadscope`. Isolate
default plotting and XDG cache directories per worker, disable worker restarts,
and retain `-n0` for serial diagnosis. Enable a default parallel command only
after repeated complete-suite and coverage equivalence gates pass.

This has the largest plausible aggregate effect because the suite contains
independent subprocess, filesystem, numerical, and JAX-heavy test modules.

### 2. Shared JAX-compiled test fixtures

This could reduce compilation work but risks hiding retracing or transformation
defects and requires behavior-by-behavior analysis. It is deferred.

### 3. Cached source/AST indexes

Repository-wide static tests repeat some reads and parses. A shared immutable
index is lower risk but expected to save much less than process parallelism. It
is deferred unless xdist is rejected.

## Worker isolation architecture

`tests/conftest.py` currently assigns shared default Matplotlib and XDG cache
directories under the system temporary root. In distributed execution every
worker must instead receive its own directory when, and only when, VerCOR
supplied the default.

Refactor the cache setup into a small private pure helper that accepts an
environment mapping and worker ID. Its behavior is:

1. Preserve user-supplied `MPLBACKEND`, `MPLCONFIGDIR`, and `XDG_CACHE_HOME`.
2. Continue defaulting `MPLBACKEND` to `Agg`.
3. In the controller or serial process, retain the existing
   `vercor-matplotlib-cache` and `vercor-xdg-cache` paths.
4. In worker `gwN`, replace only VerCOR's inherited controller defaults with
   `vercor-matplotlib-cache-gwN` and `vercor-xdg-cache-gwN`.
5. Never derive mutable state from test order or share a writable worker cache.

Add focused tests for serial defaults, worker suffixes, inherited controller
defaults, and preservation of explicit user paths before changing the helper.

Pytest's own `tmp_path`/`tmp_path_factory` resources are worker-isolated. The
suite has no real network calls or shared database. Actual signal tests execute
inside their worker processes. Environment/module monkeypatches are process
local.

## Scheduling and failure policy

Use `--dist=loadscope` so every test module stays in one worker. This preserves
the existing module-scoped distribution build and other module fixture reuse.
Do not introduce cross-worker file locks or shared artifact caches.

Use `--max-worker-restart=0`. A crashed worker must fail the run immediately;
the optimization must not hide failures through automatic replacement.

Do not use `--dist=each`, which would duplicate the full suite, or default
`load` scheduling, which can split module fixtures across workers. Do not mark
tests into serial groups until a reproducible concurrency failure demonstrates
that isolation is necessary. Any such failure triggers systematic debugging
before a marker is considered.

## Benchmark protocol

Install the declared dependency in the existing `scipy` environment, then run
identical complete-suite commands with output captured outside the repository:

1. serial control: `-n0`, twice;
2. two workers: `-n2 --dist=loadscope --max-worker-restart=0`, twice;
3. four workers: `-n4 --dist=loadscope --max-worker-restart=0`, twice; and
4. automatic workers: `-n auto --dist=loadscope --max-worker-restart=0`, twice.

Record wall, user, and system time, pass/fail/skip/xfail counts, warnings,
worker crashes, and the slowest 25 phases. Compare means and retain all raw
logs in `/private/tmp` during the task.

The candidate is the fastest worker count that passes both repetitions without
new warnings, crashes, retries, test-count differences, or cleanup failures.
More workers are not preferred when a smaller count is equally fast within
measurement noise.

Run the candidate at least one additional time, plus one
`--no-loadscope-reorder` run. Both must pass with equivalent results. Then run
the complete serial command again to confirm the parallel changes did not
introduce a serial regression.

## Default command decision

Do not change pytest `addopts` before benchmark evidence exists. If a candidate
passes every gate and has a justified lower repeated mean, configure the exact
worker count and these options in `pyproject.toml`:

```toml
addopts = "-q -n <measured-count> --dist=loadscope --max-worker-restart=0"
```

The implementation plan must replace `<measured-count>` with the measured
integer before editing configuration; no placeholder may reach the repository.
Users retain `-n0` for serial execution.

If no worker count produces a justified improvement, retain serial `addopts`,
remove unnecessary xdist-specific code/dependency changes with a forward patch,
document the failed approach in `PROGRESS.md`, and claim no improvement.

## Coverage equivalence

Run full coverage once serially with `-n0` and once with the chosen parallel
configuration. Both runs must pass the configured 90% floor and match or exceed
the baseline statement/line, branch, combined, and named-function entry
metrics.

Equal percentages alone are insufficient: the same 1,256 baseline tests and
all new worker-configuration contracts must execute. Inspect coverage totals,
test counts, skips, and failures explicitly.

## Determinism and state-leak checks

Validation must include:

- repeated candidate runs;
- `loadscope` with and without scope reordering;
- complete serial execution with `-n0`;
- focused repetition of any test implicated by a concurrency failure;
- checks that no worker cache path collides;
- checks that explicit user cache paths are unchanged; and
- tracked-worktree and temporary-resource cleanup inspection.

`pytest-randomly` remains absent, so no randomized-order result may be claimed.
The loadscope reorder comparison provides an available order-dependence probe
without adding another dependency.

## Quality and release gates

Before accepting the batch, run Black, strict flake8, mypy, compileall, the fast
suite in default and serial modes, complete serial and candidate parallel
suites, serial and parallel coverage, `git diff --check`, and the executable
distribution/CI configuration contracts.

Update `PROGRESS.md` within its 180-line limit with exact serial/parallel means,
absolute and percentage savings, selected worker count, coverage metrics,
warnings, quality gates, and remaining bottlenecks. Update CI/release commands
only if executable contract tests demonstrate they need an explicit serial or
parallel mode.

No production code, release tag, push, publication, or upload is in scope.

## Revert gates

Forward-revert the parallelization batch and document it as rejected if any of
the following occurs:

- complete test count or behavior differs between serial and parallel runs;
- any test becomes flaky, order-dependent, skipped, xfailed, or retried;
- a worker crashes, hangs, or leaks a resource;
- coverage decreases in any required metric;
- the candidate is not faster than the contemporaneous serial control; or
- safe isolation requires weakening assertions or material production changes.

## Expected retained deliverables

If accepted, retain only:

- the pytest-xdist test/development dependency;
- focused worker-cache isolation code and tests;
- measured default pytest configuration;
- any necessary executable CI/release command updates;
- the approved design and implementation plan; and
- exact durable performance/coverage evidence.

If rejected, retain only the design/plan and failed-approach evidence; all
runtime test configuration changes must be forward-reverted.
