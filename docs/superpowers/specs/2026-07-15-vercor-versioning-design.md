# VerCOR Pre-1.0 Versioning Correction Design

**Date:** 2026-07-15

## Purpose

VerCOR may publish only minor and patch releases until an externally supervised
and approved first stable release. The repository currently describes several
pre-stable API generations as major releases. This correction makes the live
candidate `0.4.0a1` and rewrites the repository's earlier VerCOR release labels
to their approved pre-1.0 equivalents.

This is a release-contract and historical-documentation correction. It does not
restore an older API, add compatibility adapters, or change numerical or
runtime behavior.

## Canonical mapping

Every reference to the affected VerCOR releases uses this exact mapping:

| Incorrect label | Correct label |
| --- | --- |
| `1.0.0` | `0.2.0` |
| `2.0.0` | `0.2.1` |
| `3.0.0` | `0.3.0` |
| `3.1.0` | `0.3.1` |
| `3.1.1` | `0.3.2` |
| `4.0.0a1` | `0.4.0a1` |

The corresponding prose and identifier forms follow the same release series:
VerCOR 4/v4 becomes VerCOR 0.4/v0.4, and historical VerCOR 3/v3 release
references become VerCOR 0.3/v0.3. References to the former 1.0 and 2.0
releases become 0.2.0 and 0.2.1 respectively.

## Scope

The correction covers every tracked repository artifact that names these
VerCOR releases:

- package metadata and VerCOR source docstrings;
- unit tests, test names, support helpers, and executable release contracts;
- public-signature and historical-API JSON contract filenames and contents;
- the frozen historical plugin fixture, its distribution metadata, package
  identifiers, use site, and CI lane;
- GitHub workflow artifact names, paths, assertions, and human-readable lane
  labels;
- the changelog, root documentation, release procedure, design, dependency and
  progress records, migration guide, archived progress, and Superpowers design
  and plan records; and
- version-bearing paths and cross-references, including migration documents,
  test modules, contract files, and plugin fixture directories.

Renamed Python modules and fixture packages must leave no imports or build paths
pointing at their former names. Renamed Markdown and JSON files must leave no
stale links or executable-contract paths.

## Boundaries

Only VerCOR's own release and API-generation labels are in scope. External or
independent version identifiers remain unchanged, including:

- Python, JAX, NumPy, SciPy, JCM, Veros, and other dependency versions;
- GitHub Action revisions such as `actions/checkout@v4`;
- the Apache License, Version 2.0;
- schema versions and independently versioned plugin distributions; and
- dates, hashes, numerical values, and field names that merely contain similar
  digits.

The work updates the checked-out repository contents. It does not rewrite Git
history, create or move tags, push commits, publish packages, upload artifacts,
or create a GitHub release.

## Artifact evidence

Hashes recorded for `vercor-4.0.0a1` archives do not prove the corrected
`vercor-0.4.0a1` artifacts and must not be relabeled. Obsolete candidate hash
records are removed. Fresh `0.4.0a1` hashes may be recorded only after new
wheel and source-distribution artifacts have been built from the corrected
metadata and have passed the installed-artifact gates.

The historical compatibility fixture is renamed from the 3.0 release label to
the 0.3 release label. Its dependency interval becomes `vercor>=0.3,<0.4`, and
the frozen public API baseline moves from `3.1.1` to `0.3.2`. These changes
correct evidence labels without claiming that the historical fixture executes
against the current alpha.

## Testing strategy

Implementation begins with failing executable contracts for the approved
mapping and current `0.4.0a1` metadata. A repository-wide version-policy test
then checks tracked source, tests, workflows, and Markdown paths and contents
for forbidden VerCOR release labels. The test uses VerCOR-specific patterns or
explicitly scoped assertions so unrelated external versions remain valid.

After the repository migration, focused documentation, distribution, and
compatibility contracts must pass. Final verification follows the project
release policy: Black, strict flake8, mypy, compileall, fast and full pytest,
branch coverage, fresh wheel/source-distribution and plugin builds, installed
artifact boundary tests, optional focused lanes, output-free gradient checks,
and `git diff --check`.

## Success criteria

The correction is complete when:

1. package and artifact metadata report exactly `0.4.0a1`;
2. historical VerCOR release evidence uses the canonical mapping above;
3. no tracked filename, source/test/workflow content, or Markdown content uses
   a forbidden VerCOR major-release label;
4. unrelated external version identifiers are unchanged;
5. all renamed paths, imports, links, builds, and executable contracts work;
6. fresh artifacts use `vercor-0.4.0a1` filenames and pass installed-boundary
   validation; and
7. no tag, push, publication, or release upload is performed.
