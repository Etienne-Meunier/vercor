# Automated Release Deployment Design

**Date:** 2026-07-24

## Purpose

Publish VerCOR package distributions to PyPI and create the corresponding
GitHub Release automatically after an authorized version tag is pushed. The
deployment must reuse the exact wheel and source distribution built and tested
by the existing GitHub Actions workflow.

## Scope

This change extends `.github/workflows/python-package.yml`, updates its
executable workflow contracts, and revises `docs/releasing.md` and
`PROGRESS.md`. It does not create or push a tag, publish a package, create a
hosted release, or change the package version.

The repository administrator must separately configure a protected GitHub
Actions environment named `release`. The existing repository secret
`PYPI_API_TOKEN` will authenticate the production upload. The distinct
`TEST_PYPI_API_TOKEN` secret will not be used.

## Trigger and authorization

The existing workflow will keep its `main` push and pull-request triggers and
add a push trigger for tags matching `v*.*.*`.

A tag push authorizes the workflow to attempt deployment only when:

- every build, installed-artifact, extension, macOS, quality, and coverage
  gate succeeds;
- the triggering ref is a tag;
- the tag is exactly `v` followed by the version in `pyproject.toml`;
- the checked-out commit is the triggering commit;
- the release environment's protection rules permit the job to start;
- PyPI does not already contain that version; and
- GitHub does not already contain a release for that tag.

Branch pushes and pull requests continue to run CI but never run deployment.
Tag protection and release-environment reviewers remain repository settings,
not workflow-controlled policy.

## Workflow architecture

`build-artifacts` remains the sole distribution builder. It will continue to
create and upload exactly:

- `dist/vercor-0.4.0-py3-none-any.whl`; and
- `dist/vercor-0.4.0.tar.gz`.

A new `publish-release` job will depend on all validation jobs rather than
rebuilding either distribution. It will have job-scoped `contents: write`
permission and use the protected `release` environment. No other job receives
publication permissions or a reference to the PyPI credential.

The job will:

1. Check out the exact triggering commit.
2. Download the `vercor-distributions` artifact into `dist/`.
3. Derive the project version from `pyproject.toml`.
4. Verify the tag/version relationship and exact two-file inventory.
5. Run distribution metadata checks.
6. Fail closed unless both the PyPI version and GitHub Release are absent.
7. Publish both distributions through
   `pypa/gh-action-pypi-publish@release/v1`, authenticating as `__token__`
   with `secrets.PYPI_API_TOKEN`.
8. Revalidate the local artifact inventory.
9. Create the GitHub Release with `gh release create`, the version-specific
   tracked release notes, and the same two distributions.

PyPI publication precedes GitHub Release creation, preserving the existing
release order. If PyPI succeeds but GitHub Release creation fails, the workflow
will not silently skip or overwrite either public state on rerun. The existing
fail-closed recovery procedure in `docs/releasing.md` remains the authorized
way to complete a partial release after exact hash and namespace validation.

## Artifact and data flow

```text
tag push
  -> build wheel and sdist once
  -> run all CI gates against the uploaded artifact bundle
  -> protected publish-release job downloads that bundle
  -> verify tag, version, metadata, inventory, and public namespace absence
  -> publish bundle to PyPI with the repository's production token
  -> attach the same bundle to a new GitHub Release
```

The temporary external-extension fixture distribution remains outside
`dist/`, is not uploaded with the VerCOR artifact bundle, and is never
published.

## Failure behavior

The deployment job fails before publication when the tag is malformed, the
tag and package version differ, a required release-notes file is missing, the
artifact inventory differs from the exact wheel and sdist, metadata checks
fail, a public version or release already exists, a remote API returns an
unexpected status, or a required CI job fails.

The job will not use PyPI's `skip-existing` behavior. Treating an existing file
as success without verifying its digest could hide a conflicting publication.
GitHub Release creation likewise will not overwrite an existing release or
asset.

## Testing and verification

Tests will be written before the workflow change and will parse the workflow to
require:

- the version-tag trigger;
- a tag-only deployment condition;
- dependencies on every validation job;
- the protected `release` environment;
- job-scoped GitHub contents permission and no OIDC permission;
- exact triggering-commit checkout and artifact download;
- tag-to-project-version validation;
- exact wheel/sdist inventory and metadata checks;
- authenticated, fail-closed PyPI and GitHub preflights;
- production publication through `secrets.PYPI_API_TOKEN`, without exposing
  the token to any other step or job; and
- GitHub Release creation with the exact two tested artifacts and tracked
  release notes.

Documentation contracts will require the administrator setup instructions and
replace the ordinary manual publication path with the automated tag workflow,
while retaining explicit partial-release recovery instructions.

Verification will run the focused release and distribution contract tests,
workflow YAML parsing, the fast test suite, static checks appropriate to the
changed Python tests, and `git diff --check`. No tag, push, package upload, or
GitHub Release creation is part of local verification.

## Alternatives considered

### Separate release workflow

A dedicated `release.yml` would either duplicate the substantial existing
build/test workflow or require a broader reusable-workflow refactor. That adds
unnecessary release risk and makes it harder to prove that publication uses
the exact artifacts already validated by CI.

### Publish after a manually created GitHub Release

Triggering on `release.published` is a common PyPI pattern, but it leaves GitHub
Release creation manual and therefore does not satisfy the requested automated
deployment.

### Extend the existing build workflow

This is the selected approach. A tag run builds once, validates once, and
publishes those same bytes. Publication permissions remain confined to one
protected, tag-only job.
