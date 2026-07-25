# Pyproject Version Source Design

## Goal

Make `pyproject.toml` the only executable source of VerCOR's package version.
Unit tests and GitHub Actions must derive their version-dependent expectations
from `[project].version`, so a future package-version bump does not require
editing test or workflow literals.

Also replace the current license classifier with exactly:

```text
License :: OSI Approved :: Apache Software License
```

## Scope

The change covers:

- the project license classifier in `pyproject.toml`;
- VerCOR package-version literals in `.github/workflows/python-package.yml`;
- VerCOR package-version literals and tautological version assertions in
  executable tests; and
- regression coverage for the single-source contract.

Historical records retain their literal release identities. This includes the
changelog, release notes, archived plans, and progress evidence. Independent
versions, such as supported Python versions and the external extension fixture
version, are not VerCOR package-version duplicates and remain unchanged.

No tag, push, publication, release creation, or other remote mutation is in
scope.

## Version Ownership

`pyproject.toml` remains the sole owner of the package version. Test code reads
that value through the existing distribution-support metadata boundary and
derives:

- the expected package version;
- the expected `v<version>` tag;
- the expected wheel name; and
- the expected source-distribution name.

Tests may compare generated artifacts, installed metadata, workflow structure,
and active release safeguards with those derived values. A test must not assert
that the value read from `pyproject.toml` equals another literal or constant
whose only purpose is to repeat the same value.

Historical-document tests may verify structure and release-safety behavior, but
their executable expectations must not embed the current VerCOR version.

## GitHub Actions Data Flow

The `build-artifacts` job reads `[project].version` once, before validating the
tag or building distributions. A metadata step writes the version, wheel name,
and source-distribution name to `GITHUB_OUTPUT`. The job exposes those values as
job outputs.

Within `build-artifacts`, later steps consume the metadata step outputs.
Downstream jobs consume the corresponding `needs.build-artifacts.outputs`
values. This keeps upload paths, install paths, checksums, and exact artifact
inventory checks aligned with the same parsed version.

The workflow continues to fail closed:

- a release tag must equal `v<project version>`;
- a build must produce exactly two files;
- both derived artifact names must exist;
- the checksum manifest must name and verify those exact files; and
- all installed-artifact, extension, macOS, quality, and deployment jobs must
  consume the derived names rather than globs.

Artifact discovery by unrestricted wildcard is rejected because it would weaken
the existing exact two-file release boundary.

## Test Changes

Test-driven implementation begins with regression tests that fail against the
current repository:

1. metadata validation requires the exact Apache Software License classifier;
2. workflow validation executes a project-metadata output step against a
   synthetic project version and requires exposed job outputs plus downstream
   consumption of those outputs.

After the RED evidence is recorded:

- reuse the version already parsed by `tests/_distribution_support.py`;
- derive tags, release paths, titles, URLs, wheel names, and sdist names with
  formatted strings;
- remove the standalone approved-version test because comparing the parsed
  value with a duplicate constant is tautological;
- remove duplicate literal-version assertions from broader metadata tests; and
- retain meaningful checks for stable classifiers, dependencies, artifact
  contents, release safety, and installed metadata.

The final verification uses `rg` to prove that Python tests and GitHub workflow
YAML contain no literal copy of the current project version. This remains a
verification gate rather than a source-text unit test.

## Error Handling

Metadata extraction uses Python's standard-library `tomllib` and fails the job
immediately if `pyproject.toml`, `[project]`, or `version` is missing or
malformed. Bash steps retain `set -euo pipefail` where metadata is prepared or
validated. Missing or wrongly named artifacts continue to fail before upload or
installation.

## Verification

Verification will include:

- focused RED/GREEN tests for metadata, workflow, and version-literal policy;
- YAML parsing and `bash -n` validation of every workflow Bash block;
- Black, strict flake8, mypy, compileall, and `git diff --check`;
- `pytest tests/ -q --fast`;
- the full test suite; and
- focused distribution/workflow tests that prove exact dynamically named
  artifacts remain enforced.

`PROGRESS.md` will record the completed change and concise verification
evidence.
