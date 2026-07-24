# Automated Release Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a validated `v*.*.*` tag push publish the exact CI-tested VerCOR wheel and sdist to PyPI and create the matching GitHub Release automatically.

**Architecture:** Extend the existing build-once workflow with a tag trigger and one tag-only `publish-release` job. The job waits for every CI lane, downloads the existing `vercor-distributions` artifact, validates the tag, package version, exact artifact inventory, metadata, and public namespace absence, publishes with the existing `PYPI_API_TOKEN` repository secret, and then creates the GitHub Release from the same files.

**Tech Stack:** GitHub Actions YAML, `actions/checkout`, `actions/setup-python`, workflow artifacts, `pypa/gh-action-pypi-publish`, GitHub CLI, PyPI/GitHub REST APIs, PyYAML, pytest.

## Global Constraints

- A pushed version tag matching `v*.*.*` is the only deployment trigger.
- The tag must equal `v` plus the version in `pyproject.toml`.
- Publication must reuse the exact wheel and sdist produced by `build-artifacts`.
- The publish job must wait for all Linux, optional-model, extension, macOS, quality, and coverage gates.
- Production PyPI authentication must use repository secret `PYPI_API_TOKEN`.
- `TEST_PYPI_API_TOKEN` and OIDC Trusted Publishing must not be used.
- PyPI `skip-existing` behavior and GitHub Release overwrites are forbidden.
- Only the tag-only publish job may reference `PYPI_API_TOKEN` or receive `contents: write`.
- The external-extension fixture wheel must never enter the release artifact bundle.
- Do not create or push a tag, upload to PyPI, create a GitHub Release, or push repository changes during local implementation.

---

### Task 1: Add executable automated-deployment contracts

**Files:**
- Modify: `tests/test_distribution_boundaries.py:288-431`
- Modify: `tests/test_api_architecture_review.py:526-585`

**Interfaces:**
- Consumes: `.github/workflows/python-package.yml` parsed through `yaml.safe_load`; `docs/releasing.md` section extraction through `_section`.
- Produces: static tests defining the exact trigger, job dependencies, permissions, secret boundary, validation commands, publish action, GitHub Release command, checkout count, and automated documentation contract.

- [ ] **Step 1: Import the release-section helper**

Change the existing import in `tests/test_distribution_boundaries.py` to:

```python
from tests.test_api_architecture_review import (
    _public_signature_contract,
    _section,
)
```

- [ ] **Step 2: Add the failing workflow deployment test**

Add this test after
`test_ci_validates_installed_artifacts_across_supported_environments`:

```python
@pytest.mark.fast_always
def test_version_tag_deploys_exact_tested_distributions() -> None:
    workflow_path = PROJECT_ROOT / ".github/workflows/python-package.yml"
    workflow_source = workflow_path.read_text(encoding="utf-8")
    workflow = yaml.safe_load(workflow_source)
    triggers = workflow[True]
    publish = workflow["jobs"]["publish-release"]

    assert triggers["push"] == {
        "branches": ["main"],
        "tags": ["v*.*.*"],
    }
    assert publish["if"] == (
        "github.event_name == 'push' && "
        "startsWith(github.ref, 'refs/tags/v')"
    )
    assert publish["needs"] == [
        "build-artifacts",
        "installed-artifact-tests",
        "external-extension-contract-tests",
        "macos-smoke",
        "quality",
    ]
    assert publish["runs-on"] == "ubuntu-latest"
    assert publish["environment"] == {
        "name": "release",
        "url": "https://pypi.org/p/vercor",
    }
    assert publish["permissions"] == {"contents": "write"}
    assert "id-token" not in publish["permissions"]

    checkout = next(
        step for step in publish["steps"] if step.get("uses") == "actions/checkout@v4"
    )
    setup = next(
        step
        for step in publish["steps"]
        if step.get("uses") == "actions/setup-python@v5"
    )
    download = next(
        step
        for step in publish["steps"]
        if step.get("uses") == "actions/download-artifact@v4"
    )
    assert checkout["with"]["ref"] == (
        "${{ github.event.pull_request.head.sha || github.sha }}"
    )
    assert setup["with"]["python-version"] == "3.12"
    assert download["with"] == {
        "name": "vercor-distributions",
        "path": "dist/",
    }

    commands = "\n".join(
        step.get("run", "") for step in publish["steps"] if isinstance(step, dict)
    )
    for required in (
        'test "$GITHUB_REF_TYPE" = "tag"',
        'test "$GITHUB_REF_NAME" = "v${VERSION}"',
        'test -f "docs/release-notes-${VERSION}.md"',
        "DIST_ARTIFACTS=(dist/*)",
        'test "${#DIST_ARTIFACTS[@]}" -eq 2',
        'WHEEL="dist/vercor-${VERSION}-py3-none-any.whl"',
        'SDIST="dist/vercor-${VERSION}.tar.gz"',
        'python -m twine check "$WHEEL" "$SDIST"',
        "https://pypi.org/pypi/vercor/${VERSION}/json",
        'test "$PYPI_STATUS" = "404"',
        "https://api.github.com/repos/${GITHUB_REPOSITORY}",
        'test "$REPO_STATUS" = "200"',
        "releases/tags/${GITHUB_REF_NAME}",
        'test "$RELEASE_STATUS" = "404"',
        'sha256sum "$WHEEL" "$SDIST"',
        "sha256sum -c",
        'gh release create "$GITHUB_REF_NAME"',
        '--notes-file "docs/release-notes-${RELEASE_VERSION}.md"',
        '"$RELEASE_WHEEL" "$RELEASE_SDIST"',
    ):
        assert required in commands

    pypi_publish = next(
        step
        for step in publish["steps"]
        if step.get("uses")
        == (
            "pypa/gh-action-pypi-publish@"
            "ba38be9e461d3875417946c167d0b5f3d385a247"
        )
    )
    assert pypi_publish["with"] == {
        "user": "__token__",
        "password": "${{ secrets.PYPI_API_TOKEN }}",
        "packages-dir": "dist/",
        "skip-existing": False,
        "attestations": False,
    }
    assert "TEST_PYPI_API_TOKEN" not in workflow_source
    assert workflow_source.count("secrets.PYPI_API_TOKEN") == 1
```

- [ ] **Step 3: Update the exact-checkout contract for the new job**

In `test_release_workflow_checks_out_the_exact_triggering_commit`, change:

```python
assert len(checkout_steps) == 5
```

to:

```python
assert len(checkout_steps) == 6
```

- [ ] **Step 4: Tighten the release-guide contract around automation**

In `test_release_bundle_contains_only_vercor_distributions`, parse the ordinary
publication section and replace the two assertions requiring manual Twine and
`gh release create` commands with:

```python
publish = _section(
    releasing,
    "## 7. Publish packages and create the hosted release",
)
assert "Pushing the annotated `v0.4.0` tag" in publish
assert "`PYPI_API_TOKEN`" in publish
assert "python-package.yml" in publish
assert "gh run watch" in publish
assert "python -m twine upload" not in publish
assert "gh release create" not in publish
```

Keep the exact local build/inventory assertions and
`assert "dist/external_extension_test_fixture" not in releasing`.

- [ ] **Step 5: Extend the publication preflight test across the workflow**

Replace the publication-specific half of
`test_release_publication_preflights_are_authenticated_and_fail_closed` after
the `for section in (prepare, tag)` loop with:

```python
assert prepare.index(repo_url) < guide.index("git tag -a")

workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
publish_steps = workflow["jobs"]["publish-release"]["steps"]
publish_commands = "\n".join(
    step.get("run", "") for step in publish_steps if isinstance(step, dict)
)
publish_action_index = next(
    index
    for index, step in enumerate(publish_steps)
    if step.get("uses")
    == (
        "pypa/gh-action-pypi-publish@"
        "ba38be9e461d3875417946c167d0b5f3d385a247"
    )
)
github_release_index = next(
    index
    for index, step in enumerate(publish_steps)
    if "gh release create" in step.get("run", "")
)

assert "https://pypi.org/pypi/vercor/${VERSION}/json" in publish_commands
assert 'test "$PYPI_STATUS" = "404"' in publish_commands
assert "https://api.github.com/repos/${GITHUB_REPOSITORY}" in publish_commands
assert 'test "$REPO_STATUS" = "200"' in publish_commands
assert "releases/tags/${GITHUB_REF_NAME}" in publish_commands
assert publish_commands.count('test "$RELEASE_STATUS" = "404"') == 2
assert publish_action_index < github_release_index
```

Remove the now-unused local variable `publish` from this test while retaining
the manual preflight checks for Sections 5 and 6.

- [ ] **Step 6: Run the focused tests to verify RED**

Run:

```bash
conda run -n scipy pytest \
  tests/test_distribution_boundaries.py::test_version_tag_deploys_exact_tested_distributions \
  tests/test_distribution_boundaries.py::test_release_bundle_contains_only_vercor_distributions \
  tests/test_api_architecture_review.py::test_release_publication_preflights_are_authenticated_and_fail_closed \
  -q --tb=short
```

Expected: three failures reporting the absent `publish-release` job and the
still-manual Section 7.

- [ ] **Step 7: Commit the executable contracts**

```bash
git add tests/test_distribution_boundaries.py tests/test_api_architecture_review.py
git commit -m "test: define automated release deployment"
```

---

### Task 2: Implement the tag-only deployment job

**Files:**
- Modify: `.github/workflows/python-package.yml:3-7`
- Modify: `.github/workflows/python-package.yml:9-190`

**Interfaces:**
- Consumes: workflow artifact `vercor-distributions`; repository secret `PYPI_API_TOKEN`; `pyproject.toml` project version; `docs/release-notes-${VERSION}.md`; GitHub-provided `GITHUB_REF_TYPE`, `GITHUB_REF_NAME`, `GITHUB_REPOSITORY`, `GITHUB_TOKEN`, and runner paths.
- Produces: PyPI release files and a GitHub Release only on a validated version-tag run after all CI jobs pass.

- [ ] **Step 1: Add the version-tag push trigger**

Change the trigger block to:

```yaml
on:
  push:
    branches: ["main"]
    tags: ["v*.*.*"]
  pull_request:
    branches: ["main"]
```

- [ ] **Step 2: Add the protected publish job**

Append this job after `quality`:

```yaml
  publish-release:
    if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')
    needs:
      - build-artifacts
      - installed-artifact-tests
      - external-extension-contract-tests
      - macos-smoke
      - quality
    runs-on: ubuntu-latest
    environment:
      name: release
      url: https://pypi.org/p/vercor
    permissions:
      contents: write
    steps:
      - uses: actions/checkout@v4
        with:
          ref: ${{ github.event.pull_request.head.sha || github.sha }}
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - uses: actions/download-artifact@v4
        with:
          name: vercor-distributions
          path: dist/
      - name: Validate release context and public namespaces
        env:
          GH_TOKEN: ${{ github.token }}
        run: |
          set -euo pipefail
          test "$GITHUB_REF_TYPE" = "tag"
          VERSION="$(python -c 'import pathlib, tomllib; print(tomllib.loads(pathlib.Path("pyproject.toml").read_text())["project"]["version"])')"
          test "$GITHUB_REF_NAME" = "v${VERSION}"
          test -f "docs/release-notes-${VERSION}.md"
          WHEEL="dist/vercor-${VERSION}-py3-none-any.whl"
          SDIST="dist/vercor-${VERSION}.tar.gz"
          shopt -s nullglob dotglob
          DIST_ARTIFACTS=(dist/*)
          test "${#DIST_ARTIFACTS[@]}" -eq 2
          test -f "$WHEEL"
          test -f "$SDIST"
          python -m pip install --upgrade pip twine
          python -m twine check "$WHEEL" "$SDIST"
          sha256sum "$WHEEL" "$SDIST" > "$RUNNER_TEMP/vercor-SHA256SUMS"
          sha256sum -c "$RUNNER_TEMP/vercor-SHA256SUMS"
          STATE_DIR="$(mktemp -d)"
          PYPI_STATUS="$(curl -sS -L -o "$STATE_DIR/pypi.json" -w '%{http_code}' "https://pypi.org/pypi/vercor/${VERSION}/json")"
          test "$PYPI_STATUS" = "404"
          REPO_STATUS="$(curl -sS -L -o "$STATE_DIR/repository.json" -w '%{http_code}' -H 'Accept: application/vnd.github+json' -H "Authorization: Bearer $GH_TOKEN" "https://api.github.com/repos/${GITHUB_REPOSITORY}")"
          test "$REPO_STATUS" = "200"
          RELEASE_STATUS="$(curl -sS -L -o "$STATE_DIR/release.json" -w '%{http_code}' -H 'Accept: application/vnd.github+json' -H "Authorization: Bearer $GH_TOKEN" "https://api.github.com/repos/${GITHUB_REPOSITORY}/releases/tags/${GITHUB_REF_NAME}")"
          test "$RELEASE_STATUS" = "404"
          echo "RELEASE_VERSION=${VERSION}" >> "$GITHUB_ENV"
          echo "RELEASE_WHEEL=${WHEEL}" >> "$GITHUB_ENV"
          echo "RELEASE_SDIST=${SDIST}" >> "$GITHUB_ENV"
      - name: Publish package distributions to PyPI
        uses: pypa/gh-action-pypi-publish@ba38be9e461d3875417946c167d0b5f3d385a247
        with:
          user: __token__
          password: ${{ secrets.PYPI_API_TOKEN }}
          packages-dir: dist/
          skip-existing: false
          attestations: false
      - name: Create GitHub Release from published distributions
        env:
          GH_TOKEN: ${{ github.token }}
        run: |
          set -euo pipefail
          shopt -s nullglob dotglob
          DIST_ARTIFACTS=(dist/*)
          test "${#DIST_ARTIFACTS[@]}" -eq 2
          test -f "$RELEASE_WHEEL"
          test -f "$RELEASE_SDIST"
          python -m twine check "$RELEASE_WHEEL" "$RELEASE_SDIST"
          sha256sum -c "$RUNNER_TEMP/vercor-SHA256SUMS"
          STATE_DIR="$(mktemp -d)"
          REPO_STATUS="$(curl -sS -L -o "$STATE_DIR/repository.json" -w '%{http_code}' -H 'Accept: application/vnd.github+json' -H "Authorization: Bearer $GH_TOKEN" "https://api.github.com/repos/${GITHUB_REPOSITORY}")"
          test "$REPO_STATUS" = "200"
          RELEASE_STATUS="$(curl -sS -L -o "$STATE_DIR/release.json" -w '%{http_code}' -H 'Accept: application/vnd.github+json' -H "Authorization: Bearer $GH_TOKEN" "https://api.github.com/repos/${GITHUB_REPOSITORY}/releases/tags/${GITHUB_REF_NAME}")"
          test "$RELEASE_STATUS" = "404"
          gh release create "$GITHUB_REF_NAME" \
            --repo "$GITHUB_REPOSITORY" \
            --verify-tag \
            --title "VerCOR ${RELEASE_VERSION}" \
            --notes-file "docs/release-notes-${RELEASE_VERSION}.md" \
            "$RELEASE_WHEEL" "$RELEASE_SDIST"
```

- [ ] **Step 3: Run the workflow-focused tests**

Run:

```bash
conda run -n scipy pytest \
  tests/test_distribution_boundaries.py::test_version_tag_deploys_exact_tested_distributions \
  tests/test_api_architecture_review.py::test_release_workflow_checks_out_the_exact_triggering_commit \
  tests/test_api_architecture_review.py::test_release_publication_preflights_are_authenticated_and_fail_closed \
  -q --tb=short
```

Expected: all three workflow-focused tests pass. The separate release-guide
contract remains red until Task 3.

- [ ] **Step 4: Parse and inspect the workflow**

Run:

```bash
conda run -n scipy python -c 'import pathlib, yaml; workflow=yaml.safe_load(pathlib.Path(".github/workflows/python-package.yml").read_text()); assert "publish-release" in workflow["jobs"]; print(sorted(workflow["jobs"]))'
git diff --check
```

Expected: the parser prints all six job names including `publish-release`, and
`git diff --check` exits 0.

- [ ] **Step 5: Commit the deployment workflow**

```bash
git add .github/workflows/python-package.yml
git commit -m "ci: automate tagged package releases"
```

---

### Task 3: Replace ordinary manual publication with the automated handoff

**Files:**
- Modify: `docs/releasing.md:142-320`

**Interfaces:**
- Consumes: the tag-triggered `python-package.yml` workflow and existing
  fail-closed recovery sections.
- Produces: maintainer instructions for configuring the `release` environment,
  using `PYPI_API_TOKEN`, pushing the annotated tag, watching the exact tag
  workflow run, verifying both public targets, and recovering manually only
  after partial failure.

- [ ] **Step 1: Document repository setup before the release procedure**

After the opening paragraph in `docs/releasing.md`, add:

```markdown
## Repository deployment configuration

Tagged deployment uses the existing production repository secret
`PYPI_API_TOKEN`. Keep `TEST_PYPI_API_TOKEN` reserved for TestPyPI; the release
workflow never references it. Configure a GitHub Actions environment named
`release` and add the required reviewers or other deployment protection rules
before pushing a version tag. The workflow grants `contents: write` and exposes
the production token only in the tag-only `publish-release` job.
```

- [ ] **Step 2: Correct the hosted-workflow description**

Replace the paragraph immediately before Section 5 with:

```markdown
The hosted workflow repeats base, JCM, Veros, wheel/sdist, external-extension,
mypy, and macOS lanes on its configured Python matrix. Pull requests and
`main` pushes run validation only. A pushed version tag matching `v*.*.*` runs
the same gates and, after all pass, the protected `publish-release` job
publishes the tested artifact bundle.
```

In Section 5, replace “The workflow file triggers only on pushes to `main` and
pull requests targeting `main`” with:

```markdown
The workflow file runs validation on pushes to `main`, pull requests targeting
`main`, and version tags. Only a version tag can satisfy the deployment job's
condition. A push to `refactor` alone does not run it.
```

- [ ] **Step 3: Make the tag push the automated deployment boundary**

After the final remote-tag verification in Section 6, add:

```markdown
Pushing the annotated `v0.4.0` tag starts `python-package.yml`. That tag push
is the publication authorization: after every CI lane passes, the protected
deployment job validates the tag against `pyproject.toml`, downloads the exact
two-file `vercor-distributions` artifact, checks both public namespaces, uses
`PYPI_API_TOKEN` to publish to PyPI, and creates the GitHub Release with the
same files. An existing local or remote tag is a stop condition. Never
overwrite or repoint a published release tag.
```

- [ ] **Step 4: Replace Section 7 with the automated run transcript**

Replace the ordinary manual upload/create procedure under
`## 7. Publish packages and create the hosted release` with:

````markdown
Pushing the annotated tag starts the automated deployment. Do not run a second
local Twine upload or `gh release create` during the ordinary release path.
The repository secret `PYPI_API_TOKEN` is supplied only to the production
publish action in `python-package.yml`.

Select the `push` run for the exact release commit and tag, wait for every CI
and deployment job, and mechanically verify its result:

```text
set -euo pipefail
test -n "${RELEASE_COMMIT:-}"
test "$(git rev-parse 'v0.4.0^{commit}')" = "$RELEASE_COMMIT"
RELEASE_RUN_ID="$(gh run list --repo nutrik/vercor --workflow python-package.yml --event push --commit "$RELEASE_COMMIT" --limit 20 --json databaseId,event,headBranch,headSha --jq 'map(select(.event == "push" and .headBranch == "v0.4.0" and .headSha == env.RELEASE_COMMIT)) | sort_by(.databaseId) | last | .databaseId // empty')"
export RELEASE_RUN_ID
test -n "${RELEASE_RUN_ID:-}"
gh run watch "$RELEASE_RUN_ID" --repo nutrik/vercor --exit-status
test "$(gh run view "$RELEASE_RUN_ID" --repo nutrik/vercor --json headSha --jq .headSha)" = "$RELEASE_COMMIT"
test "$(gh run view "$RELEASE_RUN_ID" --repo nutrik/vercor --json event --jq .event)" = "push"
test "$(gh run view "$RELEASE_RUN_ID" --repo nutrik/vercor --json headBranch --jq .headBranch)" = "v0.4.0"
test "$(gh run view "$RELEASE_RUN_ID" --repo nutrik/vercor --json conclusion --jq .conclusion)" = "success"
```

If the tag run has not appeared, is waiting for approval in the protected
`release` environment, or has not completed, stop and inspect that exact run.
Do not select a branch-push or pull-request run. If PyPI publication succeeds
but GitHub Release creation fails, use the exact-state recovery procedure
below; rerunning the ordinary deployment must fail on the existing PyPI
version rather than silently skipping it.
````

- [ ] **Step 5: Run the documentation/release focused tests**

Run:

```bash
conda run -n scipy pytest \
  tests/test_distribution_boundaries.py::test_version_tag_deploys_exact_tested_distributions \
  tests/test_distribution_boundaries.py::test_release_bundle_contains_only_vercor_distributions \
  tests/test_api_architecture_review.py::test_release_transcripts_are_well_formed_and_shell_syntax_valid \
  tests/test_api_architecture_review.py::test_release_workflow_checks_out_the_exact_triggering_commit \
  tests/test_api_architecture_review.py::test_release_publication_preflights_are_authenticated_and_fail_closed \
  tests/test_api_architecture_review.py::test_release_recovery_commands_verify_exact_state_before_mutation \
  -q --tb=short
```

Expected: all six tests pass.

- [ ] **Step 6: Commit the automated release instructions**

```bash
git add docs/releasing.md
git commit -m "docs: describe automated tagged releases"
```

---

### Task 4: Run final verification and finalize the progress evidence

**Files:**
- Modify: `PROGRESS.md:9`

**Interfaces:**
- Consumes: all workflow, test, and documentation changes from Tasks 1-3.
- Produces: concise, reproducible verification evidence and a clean release-automation change set.

- [ ] **Step 1: Run formatting and static checks**

Run:

```bash
conda run -n scipy black --check vercor examples tests
conda run -n scipy flake8 . --count --max-line-length=120 --statistics
conda run -n scipy mypy vercor examples tests
conda run -n scipy python -m compileall -q vercor examples tests
```

Expected: all commands exit 0.

- [ ] **Step 2: Run the focused release boundary**

Run:

```bash
conda run -n scipy pytest \
  tests/test_distribution_boundaries.py \
  tests/test_api_architecture_review.py \
  tests/test_release_state_validator.py \
  -q --tb=short
```

Expected: all selected tests pass.

- [ ] **Step 3: Run the fast suite**

Run:

```bash
conda run -n scipy pytest tests/ -q --fast --tb=short
```

Expected: all tests pass with only the repository's documented third-party
warnings.

- [ ] **Step 4: Recheck workflow structure and source hygiene**

Run:

```bash
conda run -n scipy python -c 'import pathlib, yaml; workflow=yaml.safe_load(pathlib.Path(".github/workflows/python-package.yml").read_text()); publish=workflow["jobs"]["publish-release"]; assert publish["permissions"] == {"contents": "write"}; assert publish["steps"][4]["with"]["password"] == "${{ secrets.PYPI_API_TOKEN }}"; print("release workflow contract: OK")'
rg -n "TEST_PYPI_API_TOKEN|id-token|skip-existing: true" .github/workflows/python-package.yml
git diff --check
git status --short
```

Expected: the Python check prints `release workflow contract: OK`; `rg` has no
matches and exits 1; whitespace validation exits 0; status contains only the
intended progress update, if not yet committed.

- [ ] **Step 5: Record the exact verification evidence**

After the checks above produce their expected results, insert this entry at the
top of `PROGRESS.md` under `## Current Status`:

```markdown
- Automated tagged release deployment completed locally (2026-07-24):
  `python-package.yml` now runs on `v*.*.*`, gates one protected
  `publish-release` job on every CI lane, validates tag/version and exact
  artifact/public-namespace state, publishes with repository secret
  `PYPI_API_TOKEN`, and creates the GitHub Release from the same tested wheel
  and sdist. TDD RED was 3 expected failures; the release boundary passed
  48/48 and fast passed 676/676. Black left 237 files unchanged, flake8
  reported 0, mypy passed 237 source files, compileall and workflow parsing
  passed, and `git diff --check` was clean. `TEST_PYPI_API_TOKEN`, OIDC,
  `skip-existing`, tag creation, pushing, publication, and hosted release
  creation were not used locally.
```

If the observed counts differ because the repository changes before execution,
record the actual observed counts while preserving every behavioral statement.

- [ ] **Step 6: Commit the final verification record**

```bash
git add PROGRESS.md
git diff --cached --check
git commit -m "docs: record release automation verification"
```

- [ ] **Step 7: Inspect the completed change set**

Run:

```bash
git status --short
git log -5 --oneline
git diff HEAD~4..HEAD --check
git diff HEAD~4..HEAD --stat
```

Expected: the worktree is clean; the design, tests, workflow, documentation,
and verification commits are present; the combined diff has no whitespace
errors. Do not push, tag, publish, or create a hosted release.
