# External Extension Test Fixture Design

## Goal

Remove the independently built extension fixture from every VerCOR release and
shared distribution artifact while preserving its installed, public-API-only
compatibility coverage under an unambiguous test-only name.

## Naming

The fixture is renamed consistently:

- source directory:
  `tests/fixtures/external_extension_test_fixture`;
- distribution:
  `external-extension-test-fixture`;
- import package:
  `external_extension_test_fixture`;
- wheel:
  `external_extension_test_fixture-0.1.0-py3-none-any.whl`; and
- active CI jobs, tests, helper names, variables, and documentation:
  “external extension test fixture.”

Generic documentation may continue to call independently authored VerCOR
extensions “plugins.” Dated plans, specifications, and archived progress
records remain unchanged because they describe historical repository states.

## Artifact boundary

VerCOR release artifacts contain exactly:

1. `vercor-0.4.0-py3-none-any.whl`; and
2. `vercor-0.4.0.tar.gz`.

Only these files are built into `dist/`, uploaded as the shared
`vercor-distributions` CI artifact, recorded in `dist/SHA256SUMS`, checked by
Twine for release, uploaded to PyPI, or attached to a GitHub release.

The fixture wheel remains installation evidence, not a release artifact. Tests
and CI build it into a temporary directory, install it beside a built VerCOR
distribution, run its smoke test outside the checkout, and run strict mypy
against the installed fixture package and an external use site. Temporary
fixture wheels are not uploaded or retained as release evidence.

## Build-helper responsibilities

`tests/_distribution_support.py` separates two concerns:

- `build_distributions(...)` builds or reuses only the VerCOR wheel and source
  distribution; and
- `build_external_extension_fixture(...)` builds the fixture wheel into an
  explicitly supplied temporary directory.

`BuiltDistributions` contains only the VerCOR wheel, source distribution, and
offline build-backend path. Fixture-consuming tests receive a separate fixture
wheel path. Installation helpers accept that path explicitly when a composed
extension check is required.

This separation makes it impossible for an externally supplied VerCOR artifact
directory to imply that a fixture wheel belongs in the release bundle.

## CI and release workflow

The `build-artifacts` CI job builds and uploads only VerCOR distributions.
Installed-artifact tests reuse those distributions and build the extension
fixture locally when needed.

The renamed extension-contract job and macOS smoke job:

1. check out the triggering commit;
2. download the VerCOR distribution artifact;
3. build the fixture wheel under the runner's temporary directory;
4. install VerCOR and the fixture through normal dependency resolution; and
5. execute the installed smoke and applicable strict typing checks.

The release guide builds the fixture only in a `mktemp` directory for the
installed-extension acceptance check. Its checksum manifest and all publication
commands mention only the two VerCOR release distributions.

## Tests

Development follows red-green-refactor:

1. change contract tests to require the new names and two-file release bundle;
2. run the focused tests and confirm they fail against the old structure;
3. rename the fixture and minimally refactor build helpers, CI, and release
   commands;
4. rerun the focused tests until green; and
5. run formatting, strict linting, mypy, compileall, fast and full pytest,
   branch coverage, installed wheel/source-distribution/fixture checks, and
   `git diff --check`.

Tests explicitly verify that:

- the old fixture distribution and import names are absent from active source,
  CI, tests, and current documentation;
- `dist/` and the uploaded CI bundle contain no fixture wheel;
- `SHA256SUMS` covers exactly the VerCOR wheel and source distribution;
- the renamed fixture imports only the stable public extension tier; and
- the temporarily built fixture still installs and runs outside the checkout.

## Documentation and coordination

`DESIGN.md`, `DEPENDENCIES.md`, active architecture and release documentation,
release notes, and `PROGRESS.md` use the new terminology and describe the
temporary-only artifact boundary. `PROGRESS.md` records the red-green evidence
and final verification results.

## Non-goals

- No VerCOR runtime API or numerical behavior changes.
- No plugin registry, entry-point discovery, or new extension framework.
- No publication of the fixture under its new name.
- No unrelated refactoring of distribution or release-validation code.
