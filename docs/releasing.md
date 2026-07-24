# Releasing VerCOR

This is the verification procedure for a release candidate. Preparing a
candidate does not authorize a commit, pull request, tag, push, upload,
publication, merge, or hosted release.

## 1. Confirm and review the candidate

- Work from the intended `refactor` branch with complete history.
- Confirm `pyproject.toml` and `CHANGELOG.md` use the intended version.
- Confirm the package root and canonical owner manifests match live signatures.
- Confirm optional JCM and Veros versions in the verification environment.
- Leave CAMulator uninstalled and unpinned until an exact compatible release is
  verified.

Perform the read-only review before requesting commit authority:

```bash
set -euo pipefail
test "$(git branch --show-current)" = "refactor"
git status --short --untracked-files=all
git diff --check
git diff
git diff --cached --check
git diff --cached
git ls-files --others --exclude-standard
```

Only after explicit commit authorization may a maintainer stage the completely
reviewed release state and create the release commit. Run this transcript in one
shell so later gates retain `RELEASE_COMMIT`:

```text
set -euo pipefail
test "$(git branch --show-current)" = "refactor"
git add -A
git diff --cached --check
git diff --cached
git commit -m "Release 0.4.0"
RELEASE_COMMIT="$(git rev-parse HEAD)"
export RELEASE_COMMIT
test -n "${RELEASE_COMMIT:-}"
printf 'Release commit: %s\n' "$RELEASE_COMMIT"
test -z "$(git status --porcelain --untracked-files=all)"
```

Do not infer the release SHA from a later moving branch name.

## 2. Run source gates from the release commit

Use the supported environment binaries directly if the Conda launcher is
unavailable. This single-shell transcript fails closed and binds every gate to
the recorded commit:

```bash
set -euo pipefail
test -n "${RELEASE_COMMIT:-}"
test "$(git branch --show-current)" = "refactor"
test "$(git rev-parse HEAD)" = "$RELEASE_COMMIT"
test -z "$(git status --porcelain --untracked-files=all)"
python -m black --check vercor examples tests
python -m flake8 . --count --max-line-length=120 --statistics
python -m mypy vercor examples tests
python -m compileall -q vercor examples tests
python -m pytest tests/ -q --fast --tb=short
python -m pytest tests/ -q --tb=short
python -m pytest tests/ -q --cov=vercor --cov-branch --cov-report=term-missing --cov-fail-under=90
git diff --check
test "$(git rev-parse HEAD)" = "$RELEASE_COMMIT"
test -z "$(git status --porcelain --untracked-files=all)"
```

The clean-tree checks protect the source/ref state. They intentionally do not
inspect ignored `dist/` bytes; the checksum manifest below protects those.

## 3. Build once and create the checksum manifest

Build the two publishable VerCOR distributions from the exact clean release
commit, inspect the bundle, and create the ignored `dist/SHA256SUMS` manifest:

```bash
set -euo pipefail
test -n "${RELEASE_COMMIT:-}"
test "$(git rev-parse HEAD)" = "$RELEASE_COMMIT"
test -z "$(git status --porcelain --untracked-files=all)"
python -m build --outdir dist
unzip -p dist/vercor-0.4.0-py3-none-any.whl vercor-0.4.0.dist-info/METADATA
tar -xOf dist/vercor-0.4.0.tar.gz vercor-0.4.0/PKG-INFO
python -m twine check dist/vercor-0.4.0-py3-none-any.whl dist/vercor-0.4.0.tar.gz
VERCOR_ARTIFACT_DIR="$(pwd)/dist" python -m pytest tests/test_distribution_boundaries.py -q --tb=short
(
  cd dist
  shasum -a 256 vercor-0.4.0-py3-none-any.whl vercor-0.4.0.tar.gz > SHA256SUMS
  shasum -a 256 -c SHA256SUMS
)
python -c 'import importlib.metadata as m; print("JCM", m.version("jcm")); print("Veros", m.version("veros"))'
test "$(git rev-parse HEAD)" = "$RELEASE_COMMIT"
test -z "$(git status --porcelain --untracked-files=all)"
```

`dist/SHA256SUMS` is ignored release evidence, not a source-cleanliness signal.
Record it against `RELEASE_COMMIT` and retain it through hosted CI, package
publication, hosted asset upload, and recovery.

## 4. Run local installed-artifact acceptance

Run the bounded optional-model nodes and explicit output-free gradient
acceptance against the exact source commit:

```bash
set -euo pipefail
test -n "${RELEASE_COMMIT:-}"
test "$(git rev-parse HEAD)" = "$RELEASE_COMMIT"
python -m pytest tests/test_setup_lifecycle_helpers.py::test_make_jcm_land_atmosphere_replaces_only_missing_forcing tests/test_external_components_coverage.py::test_jax_gcm_initialize_uses_provided_forcing_and_can_spin_up tests/test_external_components_coverage.py::test_jax_gcm_initialize_builds_default_forcing_when_missing tests/test_setup_boundaries.py::test_veros_implementation_import_does_not_configure_runtime tests/test_setup_boundaries.py::test_veros_factory_configures_once_before_implementation_import tests/test_external_components_coverage.py::test_veros_initialize_spinup_follows_enabled_only -q --tb=short
python -m pytest tests/test_v0_4_workflow_execution.py::test_output_free_workflow_preserves_jvp_and_reverse_mode_gradients tests/test_v0_4_workflow_execution.py::test_payload_dependent_multi_step_scan_preserves_treedef_jvp_and_grad tests/test_v0_4_output_providers.py::test_all_disabled_target_remains_jit_and_gradient_compatible -q --tb=short
(
  smoke_dir="$(mktemp -d)"
  external_extension_fixture_dir="$(mktemp -d)"
  python -m build --wheel \
    --outdir "$external_extension_fixture_dir" \
    tests/fixtures/external_extension_test_fixture
  python -m pip install --target "$smoke_dir/site" "dist/vercor-0.4.0-py3-none-any.whl"
  python -m pip install --target "$smoke_dir/site" \
    "$external_extension_fixture_dir/external_extension_test_fixture-0.1.0-py3-none-any.whl"
  cd "$smoke_dir"
  PYTHONPATH="$smoke_dir/site" \
    python -m external_extension_test_fixture.smoke \
    --output-dir "$smoke_dir/extension-output"
)
(cd dist && shasum -a 256 -c SHA256SUMS)
```

The hosted workflow repeats base, JCM, Veros, wheel/sdist, external-extension,
mypy, and macOS lanes on its configured Python matrix.

## 5. Prepare the required release pull request

The workflow file triggers only on pushes to `main` and pull requests targeting
`main`. A push to `refactor` alone does not run it. Before CI or a push, fetch
the current protected branch, prove it is an ancestor of the reviewed release
commit, and distinguish repository access (HTTP 200) from release absence
(HTTP 404). PyPI 0.4.0 must also be absent:

```text
set -euo pipefail
test -n "${RELEASE_COMMIT:-}"
test "$(git rev-parse HEAD)" = "$RELEASE_COMMIT"
git fetch --no-tags origin main
MAIN_COMMIT="$(git rev-parse refs/remotes/origin/main)"
export MAIN_COMMIT
test -n "${MAIN_COMMIT:-}"
git merge-base --is-ancestor "$MAIN_COMMIT" "$RELEASE_COMMIT"
GITHUB_TOKEN="$(gh auth token)"
export GITHUB_TOKEN
test -n "${GITHUB_TOKEN:-}"
PREFLIGHT_DIR="$(mktemp -d)"
REPO_STATUS="$(curl -sS -L -o "$PREFLIGHT_DIR/repository.json" -w '%{http_code}' -H 'Accept: application/vnd.github+json' -H "Authorization: Bearer $GITHUB_TOKEN" https://api.github.com/repos/nutrik/vercor)"
export REPO_STATUS
test "$REPO_STATUS" = "200"
RELEASE_STATUS="$(curl -sS -L -o "$PREFLIGHT_DIR/release.json" -w '%{http_code}' -H 'Accept: application/vnd.github+json' -H "Authorization: Bearer $GITHUB_TOKEN" https://api.github.com/repos/nutrik/vercor/releases/tags/v0.4.0)"
export RELEASE_STATUS
test "$RELEASE_STATUS" = "404"
PYPI_STATUS="$(curl -sS -L -o "$PREFLIGHT_DIR/pypi.json" -w '%{http_code}' https://pypi.org/pypi/vercor/0.4.0/json)"
export PYPI_STATUS
test "$PYPI_STATUS" = "404"
gh pr list --repo nutrik/vercor --state open --base main --head refactor --json number,url,headRefName,baseRefName,headRefOid
```

If no authorized pull request exists, this is the exact preparation command.
Run it only with explicit pull-request creation authority:

```text
set -euo pipefail
test -n "${RELEASE_COMMIT:-}"
test "$(git rev-parse HEAD)" = "$RELEASE_COMMIT"
test -n "${MAIN_COMMIT:-}"
git merge-base --is-ancestor "$MAIN_COMMIT" "$RELEASE_COMMIT"
gh pr create --repo nutrik/vercor --base main --head refactor --title "Release 0.4.0" --body "Prepare VerCOR 0.4.0 from commit $RELEASE_COMMIT."
```

Confirm exactly one open matching pull request, push the reviewed commit, select
the `pull_request` run of `python-package.yml` at the exact SHA, watch it, and
mechanically recheck the run:

```text
set -euo pipefail
test -n "${RELEASE_COMMIT:-}"
test "$(git rev-parse HEAD)" = "$RELEASE_COMMIT"
test -z "$(git status --porcelain --untracked-files=all)"
git fetch --no-tags origin main
MAIN_COMMIT="$(git rev-parse refs/remotes/origin/main)"
export MAIN_COMMIT
test -n "${MAIN_COMMIT:-}"
git merge-base --is-ancestor "$MAIN_COMMIT" "$RELEASE_COMMIT"
RELEASE_PR_NUMBER="$(gh pr list --repo nutrik/vercor --state open --base main --head refactor --json number --jq 'if length == 1 then .[0].number else empty end')"
export RELEASE_PR_NUMBER
test -n "${RELEASE_PR_NUMBER:-}"
git push origin refactor
RELEASE_RUN_ID="$(gh run list --repo nutrik/vercor --workflow python-package.yml --event pull_request --branch refactor --commit "$RELEASE_COMMIT" --limit 20 --json databaseId,event,headSha --jq 'map(select(.event == "pull_request" and .headSha == env.RELEASE_COMMIT)) | sort_by(.databaseId) | last | .databaseId // empty')"
export RELEASE_RUN_ID
test -n "${RELEASE_RUN_ID:-}"
gh run watch "$RELEASE_RUN_ID" --repo nutrik/vercor --exit-status
test "$(gh run view "$RELEASE_RUN_ID" --repo nutrik/vercor --json headSha --jq .headSha)" = "$RELEASE_COMMIT"
test "$(gh run view "$RELEASE_RUN_ID" --repo nutrik/vercor --json event --jq .event)" = "pull_request"
test "$(gh run view "$RELEASE_RUN_ID" --repo nutrik/vercor --json conclusion --jq .conclusion)" = "success"
(cd dist && shasum -a 256 -c SHA256SUMS)
```

If the run has not appeared yet, stop and rerun the selection transcript later.
Do not select a `push` run, a run for another workflow, or a run at another SHA.

## 6. Create and verify the annotated tag

Immediately before tagging, fetch `main` again, repeat the ancestry and
authenticated public-namespace preflights, and confirm the local and remote tag
are absent. A repository HTTP 404 is never accepted as evidence that the
release is absent:

```text
set -euo pipefail
test -n "${RELEASE_COMMIT:-}"
test "$(git rev-parse HEAD)" = "$RELEASE_COMMIT"
test -z "$(git status --porcelain --untracked-files=all)"
git fetch --no-tags origin main
MAIN_COMMIT="$(git rev-parse refs/remotes/origin/main)"
export MAIN_COMMIT
test -n "${MAIN_COMMIT:-}"
git merge-base --is-ancestor "$MAIN_COMMIT" "$RELEASE_COMMIT"
GITHUB_TOKEN="$(gh auth token)"
export GITHUB_TOKEN
test -n "${GITHUB_TOKEN:-}"
TAG_PREFLIGHT_DIR="$(mktemp -d)"
REPO_STATUS="$(curl -sS -L -o "$TAG_PREFLIGHT_DIR/repository.json" -w '%{http_code}' -H 'Accept: application/vnd.github+json' -H "Authorization: Bearer $GITHUB_TOKEN" https://api.github.com/repos/nutrik/vercor)"
export REPO_STATUS
test "$REPO_STATUS" = "200"
RELEASE_STATUS="$(curl -sS -L -o "$TAG_PREFLIGHT_DIR/release.json" -w '%{http_code}' -H 'Accept: application/vnd.github+json' -H "Authorization: Bearer $GITHUB_TOKEN" https://api.github.com/repos/nutrik/vercor/releases/tags/v0.4.0)"
export RELEASE_STATUS
test "$RELEASE_STATUS" = "404"
PYPI_STATUS="$(curl -sS -L -o "$TAG_PREFLIGHT_DIR/pypi.json" -w '%{http_code}' https://pypi.org/pypi/vercor/0.4.0/json)"
export PYPI_STATUS
test "$PYPI_STATUS" = "404"
test -z "$(git tag --list v0.4.0)"
REMOTE_TAG_PRECHECK="$(git ls-remote --tags origin refs/tags/v0.4.0 'refs/tags/v0.4.0^{}')"
export REMOTE_TAG_PRECHECK
test -z "$REMOTE_TAG_PRECHECK"
git tag -a v0.4.0 "$RELEASE_COMMIT" -m "VerCOR 0.4.0"
test "$(git cat-file -t v0.4.0)" = "tag"
test "$(git rev-parse 'v0.4.0^{commit}')" = "$RELEASE_COMMIT"
git show --stat v0.4.0
git push origin refs/tags/v0.4.0
REMOTE_TAG_STATE="$(git ls-remote --tags origin refs/tags/v0.4.0 'refs/tags/v0.4.0^{}')"
export REMOTE_TAG_STATE
REMOTE_TAG_COMMIT="$(printf '%s\n' "$REMOTE_TAG_STATE" | awk '$2 == "refs/tags/v0.4.0^{}" {print $1}')"
export REMOTE_TAG_COMMIT
test -n "${REMOTE_TAG_COMMIT:-}"
test "$REMOTE_TAG_COMMIT" = "$RELEASE_COMMIT"
```

An existing local or remote tag is a stop condition. Never overwrite or repoint
a published release tag.

## 7. Publish packages and create the hosted release

Run only with explicit package-publication and hosted-release authority. The
checksum manifest is verified immediately before each external artifact upload:

```text
set -euo pipefail
test -n "${RELEASE_COMMIT:-}"
test "$(git rev-parse 'v0.4.0^{commit}')" = "$RELEASE_COMMIT"
REMOTE_TAG_COMMIT="$(git ls-remote origin 'refs/tags/v0.4.0^{}' | awk '{print $1}')"
export REMOTE_TAG_COMMIT
test "$REMOTE_TAG_COMMIT" = "$RELEASE_COMMIT"
python -m twine check dist/vercor-0.4.0-py3-none-any.whl dist/vercor-0.4.0.tar.gz
(cd dist && shasum -a 256 -c SHA256SUMS)
PUBLISH_STATE_DIR="$(mktemp -d)"
PYPI_STATUS="$(curl -sS -L -o "$PUBLISH_STATE_DIR/pypi.json" -w '%{http_code}' https://pypi.org/pypi/vercor/0.4.0/json)"
export PYPI_STATUS
test "$PYPI_STATUS" = "404"
python -m twine upload --repository-url https://upload.pypi.org/legacy/ dist/vercor-0.4.0-py3-none-any.whl dist/vercor-0.4.0.tar.gz
(cd dist && shasum -a 256 -c SHA256SUMS)
GITHUB_TOKEN="$(gh auth token)"
export GITHUB_TOKEN
test -n "${GITHUB_TOKEN:-}"
REPO_STATUS="$(curl -sS -L -o "$PUBLISH_STATE_DIR/repository.json" -w '%{http_code}' -H 'Accept: application/vnd.github+json' -H "Authorization: Bearer $GITHUB_TOKEN" https://api.github.com/repos/nutrik/vercor)"
export REPO_STATUS
test "$REPO_STATUS" = "200"
RELEASE_STATUS="$(curl -sS -L -o "$PUBLISH_STATE_DIR/release.json" -w '%{http_code}' -H 'Accept: application/vnd.github+json' -H "Authorization: Bearer $GITHUB_TOKEN" https://api.github.com/repos/nutrik/vercor/releases/tags/v0.4.0)"
export RELEASE_STATUS
test "$RELEASE_STATUS" = "404"
gh release create v0.4.0 --repo nutrik/vercor --title "VerCOR 0.4.0" --notes-file docs/release-notes-0.4.0.md dist/vercor-0.4.0-py3-none-any.whl dist/vercor-0.4.0.tar.gz
```

The temporary external-extension fixture wheel is neither checksummed nor
published to the VerCOR package index or hosted release.

## 8. Verify the published package and hosted release

```bash
set -euo pipefail
test -n "${RELEASE_COMMIT:-}"
published_check_dir="$(mktemp -d)"
python -m venv "$published_check_dir/venv"
"$published_check_dir/venv/bin/python" -m pip install --upgrade pip
"$published_check_dir/venv/bin/python" -m pip install --no-cache-dir "vercor==0.4.0"
"$published_check_dir/venv/bin/python" -c 'import importlib.metadata as m; assert m.version("vercor") == "0.4.0"; print(m.version("vercor"))'
"$published_check_dir/venv/bin/python" -c 'from vercor import Clock, Coupler, Exchange, RectilinearGrid, RunState, RuntimeOptions'
gh release view v0.4.0 --repo nutrik/vercor --json tagName,name,isDraft,isPrerelease,assets
release_verify_dir="$(mktemp -d)"
gh release download v0.4.0 --repo nutrik/vercor --dir "$release_verify_dir"
test "$(shasum -a 256 "$release_verify_dir/vercor-0.4.0-py3-none-any.whl" | awk '{print $1}')" = "$(awk '$2 == "vercor-0.4.0-py3-none-any.whl" {print $1}' dist/SHA256SUMS)"
test "$(shasum -a 256 "$release_verify_dir/vercor-0.4.0.tar.gz" | awk '{print $1}')" = "$(awk '$2 == "vercor-0.4.0.tar.gz" {print $1}' dist/SHA256SUMS)"
```

## 9. Query public state before recovery

Before any destructive tag action, query PyPI and the hosted-release API,
accept only HTTP 200 or 404, and compare every published PyPI file digest with
`dist/SHA256SUMS`:

```text
set -euo pipefail
test -n "${RELEASE_COMMIT:-}"
(cd dist && shasum -a 256 -c SHA256SUMS)
RECOVERY_STATE_DIR="$(mktemp -d)"
export RECOVERY_STATE_DIR
PYPI_JSON="$RECOVERY_STATE_DIR/pypi.json"
export PYPI_JSON
PYPI_STATUS="$(curl -sS -L -o "$PYPI_JSON" -w '%{http_code}' https://pypi.org/pypi/vercor/0.4.0/json)"
export PYPI_STATUS
case "$PYPI_STATUS" in 200|404) ;; *) printf 'Unexpected PyPI HTTP status: %s\n' "$PYPI_STATUS" >&2; exit 1 ;; esac
if [ "$PYPI_STATUS" = 200 ]; then
  python - "$PYPI_JSON" dist/SHA256SUMS <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
manifest = {}
for line in pathlib.Path(sys.argv[2]).read_text(encoding="utf-8").splitlines():
    digest, name = line.split(maxsplit=1)
    manifest[name.lstrip("*")] = digest
expected = {"vercor-0.4.0-py3-none-any.whl", "vercor-0.4.0.tar.gz"}
urls = payload.get("urls", [])
if not urls:
    raise SystemExit("PyPI returned 200 without release files")
for item in urls:
    name = item["filename"]
    if name not in expected or name not in manifest:
        raise SystemExit(f"unexpected PyPI file: {name}")
    if item["digests"]["sha256"] != manifest[name]:
        raise SystemExit(f"PyPI digest mismatch: {name}")
    print(name, item["digests"]["sha256"])
PY
fi
GITHUB_TOKEN="$(gh auth token)"
export GITHUB_TOKEN
test -n "${GITHUB_TOKEN:-}"
REPOSITORY_JSON="$RECOVERY_STATE_DIR/repository.json"
export REPOSITORY_JSON
REPO_STATUS="$(curl -sS -L -H 'Accept: application/vnd.github+json' -H "Authorization: Bearer $GITHUB_TOKEN" -o "$REPOSITORY_JSON" -w '%{http_code}' https://api.github.com/repos/nutrik/vercor)"
export REPO_STATUS
test "$REPO_STATUS" = "200"
HOSTED_JSON="$RECOVERY_STATE_DIR/hosted-release.json"
export HOSTED_JSON
HOSTED_STATUS="$(curl -sS -L -H 'Accept: application/vnd.github+json' -H "Authorization: Bearer $GITHUB_TOKEN" -o "$HOSTED_JSON" -w '%{http_code}' https://api.github.com/repos/nutrik/vercor/releases/tags/v0.4.0)"
export HOSTED_STATUS
case "$HOSTED_STATUS" in 200|404) ;; *) printf 'Unexpected GitHub HTTP status: %s\n' "$HOSTED_STATUS" >&2; exit 1 ;; esac
printf 'PyPI status: %s; hosted release status: %s\n' "$PYPI_STATUS" "$HOSTED_STATUS"
```

## 10. Safe recovery

If both public-state queries returned 404, no package file was accepted, no
hosted release exists, and an incorrect remote tag is the only external
mutation, deletion still requires explicit destructive approval:

```text
set -euo pipefail
test -n "${RELEASE_COMMIT:-}"
test "${DESTRUCTIVE_TAG_DELETE_APPROVED:-}" = "yes"
test -n "${WRONG_TAG_COMMIT:-}"
GITHUB_TOKEN="$(gh auth token)"
test -n "${GITHUB_TOKEN:-}"
DELETE_STATE_DIR="$(mktemp -d)"
REPO_STATUS="$(curl -sS -L -o "$DELETE_STATE_DIR/repository.json" -w '%{http_code}' -H 'Accept: application/vnd.github+json' -H "Authorization: Bearer $GITHUB_TOKEN" https://api.github.com/repos/nutrik/vercor)"
test "$REPO_STATUS" = "200"
RELEASE_STATUS="$(curl -sS -L -o "$DELETE_STATE_DIR/release.json" -w '%{http_code}' -H 'Accept: application/vnd.github+json' -H "Authorization: Bearer $GITHUB_TOKEN" https://api.github.com/repos/nutrik/vercor/releases/tags/v0.4.0)"
test "$RELEASE_STATUS" = "404"
PYPI_STATUS="$(curl -sS -L -o "$DELETE_STATE_DIR/pypi.json" -w '%{http_code}' https://pypi.org/pypi/vercor/0.4.0/json)"
test "$PYPI_STATUS" = "404"
REMOTE_TAG_COMMIT="$(git ls-remote origin 'refs/tags/v0.4.0^{}' | awk '{print $1}')"
test "$REMOTE_TAG_COMMIT" = "$WRONG_TAG_COMMIT"
test "$REMOTE_TAG_COMMIT" != "$RELEASE_COMMIT"
test "$(git rev-parse 'v0.4.0^{commit}')" = "$WRONG_TAG_COMMIT"
IMMEDIATE_REPO_STATUS="$(curl -sS -L -o "$DELETE_STATE_DIR/immediate-repository.json" -w '%{http_code}' -H 'Accept: application/vnd.github+json' -H "Authorization: Bearer $GITHUB_TOKEN" https://api.github.com/repos/nutrik/vercor)"
test "$IMMEDIATE_REPO_STATUS" = "200"
IMMEDIATE_RELEASE_STATUS="$(curl -sS -L -o "$DELETE_STATE_DIR/immediate-release.json" -w '%{http_code}' -H 'Accept: application/vnd.github+json' -H "Authorization: Bearer $GITHUB_TOKEN" https://api.github.com/repos/nutrik/vercor/releases/tags/v0.4.0)"
test "$IMMEDIATE_RELEASE_STATUS" = "404"
IMMEDIATE_PYPI_STATUS="$(curl -sS -L -o "$DELETE_STATE_DIR/immediate-pypi.json" -w '%{http_code}' https://pypi.org/pypi/vercor/0.4.0/json)"
test "$IMMEDIATE_PYPI_STATUS" = "404"
git push --delete origin v0.4.0
git tag --delete v0.4.0
```

If PyPI returned 200 with only one verified file, run exactly one separately
labeled missing-file alternative after confirming which filename is absent.

### Missing PyPI wheel only

```text
set -euo pipefail
test -n "${RELEASE_COMMIT:-}"
(cd dist && shasum -a 256 -c SHA256SUMS)
PYPI_RECOVERY_DIR="$(mktemp -d)"
PYPI_STATUS="$(curl -sS -L -o "$PYPI_RECOVERY_DIR/pypi.json" -w '%{http_code}' https://pypi.org/pypi/vercor/0.4.0/json)"
test "$PYPI_STATUS" = "200"
python tools/validate_release_state.py pypi --json "$PYPI_RECOVERY_DIR/pypi.json" --manifest dist/SHA256SUMS --expect vercor-0.4.0.tar.gz
REMOTE_TAG_COMMIT="$(git ls-remote origin 'refs/tags/v0.4.0^{}' | awk '{print $1}')"
test "$REMOTE_TAG_COMMIT" = "$RELEASE_COMMIT"
IMMEDIATE_PYPI_STATUS="$(curl -sS -L -o "$PYPI_RECOVERY_DIR/immediate-pypi.json" -w '%{http_code}' https://pypi.org/pypi/vercor/0.4.0/json)"
test "$IMMEDIATE_PYPI_STATUS" = "200"
python tools/validate_release_state.py pypi --json "$PYPI_RECOVERY_DIR/immediate-pypi.json" --manifest dist/SHA256SUMS --expect vercor-0.4.0.tar.gz
python -m twine upload --repository-url https://upload.pypi.org/legacy/ dist/vercor-0.4.0-py3-none-any.whl
```

### Missing PyPI sdist only

```text
set -euo pipefail
test -n "${RELEASE_COMMIT:-}"
(cd dist && shasum -a 256 -c SHA256SUMS)
PYPI_RECOVERY_DIR="$(mktemp -d)"
PYPI_STATUS="$(curl -sS -L -o "$PYPI_RECOVERY_DIR/pypi.json" -w '%{http_code}' https://pypi.org/pypi/vercor/0.4.0/json)"
test "$PYPI_STATUS" = "200"
python tools/validate_release_state.py pypi --json "$PYPI_RECOVERY_DIR/pypi.json" --manifest dist/SHA256SUMS --expect vercor-0.4.0-py3-none-any.whl
REMOTE_TAG_COMMIT="$(git ls-remote origin 'refs/tags/v0.4.0^{}' | awk '{print $1}')"
test "$REMOTE_TAG_COMMIT" = "$RELEASE_COMMIT"
IMMEDIATE_PYPI_STATUS="$(curl -sS -L -o "$PYPI_RECOVERY_DIR/immediate-pypi.json" -w '%{http_code}' https://pypi.org/pypi/vercor/0.4.0/json)"
test "$IMMEDIATE_PYPI_STATUS" = "200"
python tools/validate_release_state.py pypi --json "$PYPI_RECOVERY_DIR/immediate-pypi.json" --manifest dist/SHA256SUMS --expect vercor-0.4.0-py3-none-any.whl
python -m twine upload --repository-url https://upload.pypi.org/legacy/ dist/vercor-0.4.0.tar.gz
```

If an incorrect package file was accepted, yank 0.4.0 through package-index
administration, preserve the tag and evidence, and prepare a new patch release.
Published files cannot be replaced and a released version must not be reused
for different bytes.

### Missing hosted release

Use only when the hosted-release API query proved no release exists:

```text
set -euo pipefail
test -n "${RELEASE_COMMIT:-}"
(cd dist && shasum -a 256 -c SHA256SUMS)
GITHUB_TOKEN="$(gh auth token)"
test -n "${GITHUB_TOKEN:-}"
HOSTED_RECOVERY_DIR="$(mktemp -d)"
REPO_STATUS="$(curl -sS -L -o "$HOSTED_RECOVERY_DIR/repository.json" -w '%{http_code}' -H 'Accept: application/vnd.github+json' -H "Authorization: Bearer $GITHUB_TOKEN" https://api.github.com/repos/nutrik/vercor)"
test "$REPO_STATUS" = "200"
RELEASE_STATUS="$(curl -sS -L -o "$HOSTED_RECOVERY_DIR/release.json" -w '%{http_code}' -H 'Accept: application/vnd.github+json' -H "Authorization: Bearer $GITHUB_TOKEN" https://api.github.com/repos/nutrik/vercor/releases/tags/v0.4.0)"
test "$RELEASE_STATUS" = "404"
REMOTE_TAG_COMMIT="$(git ls-remote origin 'refs/tags/v0.4.0^{}' | awk '{print $1}')"
test "$REMOTE_TAG_COMMIT" = "$RELEASE_COMMIT"
IMMEDIATE_REPO_STATUS="$(curl -sS -L -o "$HOSTED_RECOVERY_DIR/immediate-repository.json" -w '%{http_code}' -H 'Accept: application/vnd.github+json' -H "Authorization: Bearer $GITHUB_TOKEN" https://api.github.com/repos/nutrik/vercor)"
test "$IMMEDIATE_REPO_STATUS" = "200"
IMMEDIATE_RELEASE_STATUS="$(curl -sS -L -o "$HOSTED_RECOVERY_DIR/immediate-release.json" -w '%{http_code}' -H 'Accept: application/vnd.github+json' -H "Authorization: Bearer $GITHUB_TOKEN" https://api.github.com/repos/nutrik/vercor/releases/tags/v0.4.0)"
test "$IMMEDIATE_RELEASE_STATUS" = "404"
gh release create v0.4.0 --repo nutrik/vercor --title "VerCOR 0.4.0" --notes-file docs/release-notes-0.4.0.md dist/vercor-0.4.0-py3-none-any.whl dist/vercor-0.4.0.tar.gz
```

### Missing hosted wheel asset only

```text
set -euo pipefail
test -n "${RELEASE_COMMIT:-}"
(cd dist && shasum -a 256 -c SHA256SUMS)
GITHUB_TOKEN="$(gh auth token)"
test -n "${GITHUB_TOKEN:-}"
HOSTED_RECOVERY_DIR="$(mktemp -d)"
REPO_STATUS="$(curl -sS -L -o "$HOSTED_RECOVERY_DIR/repository.json" -w '%{http_code}' -H 'Accept: application/vnd.github+json' -H "Authorization: Bearer $GITHUB_TOKEN" https://api.github.com/repos/nutrik/vercor)"
test "$REPO_STATUS" = "200"
RELEASE_STATUS="$(curl -sS -L -o "$HOSTED_RECOVERY_DIR/release.json" -w '%{http_code}' -H 'Accept: application/vnd.github+json' -H "Authorization: Bearer $GITHUB_TOKEN" https://api.github.com/repos/nutrik/vercor/releases/tags/v0.4.0)"
test "$RELEASE_STATUS" = "200"
python tools/validate_release_state.py assets --json "$HOSTED_RECOVERY_DIR/release.json" --expect vercor-0.4.0.tar.gz
UNAFFECTED_ASSET_DIR="$(mktemp -d)"
gh release download v0.4.0 --repo nutrik/vercor --pattern vercor-0.4.0.tar.gz --dir "$UNAFFECTED_ASSET_DIR"
python tools/validate_release_state.py files --directory "$UNAFFECTED_ASSET_DIR" --manifest dist/SHA256SUMS --expect vercor-0.4.0.tar.gz
REMOTE_TAG_COMMIT="$(git ls-remote origin 'refs/tags/v0.4.0^{}' | awk '{print $1}')"
test "$REMOTE_TAG_COMMIT" = "$RELEASE_COMMIT"
IMMEDIATE_RELEASE_STATUS="$(curl -sS -L -o "$HOSTED_RECOVERY_DIR/immediate-release.json" -w '%{http_code}' -H 'Accept: application/vnd.github+json' -H "Authorization: Bearer $GITHUB_TOKEN" https://api.github.com/repos/nutrik/vercor/releases/tags/v0.4.0)"
test "$IMMEDIATE_RELEASE_STATUS" = "200"
python tools/validate_release_state.py assets --json "$HOSTED_RECOVERY_DIR/immediate-release.json" --expect vercor-0.4.0.tar.gz
IMMEDIATE_UNAFFECTED_ASSET_DIR="$(mktemp -d)"
gh release download v0.4.0 --repo nutrik/vercor --pattern vercor-0.4.0.tar.gz --dir "$IMMEDIATE_UNAFFECTED_ASSET_DIR"
python tools/validate_release_state.py files --directory "$IMMEDIATE_UNAFFECTED_ASSET_DIR" --manifest dist/SHA256SUMS --expect vercor-0.4.0.tar.gz
gh release upload v0.4.0 --repo nutrik/vercor dist/vercor-0.4.0-py3-none-any.whl
```

### Missing hosted sdist asset only

```text
set -euo pipefail
test -n "${RELEASE_COMMIT:-}"
(cd dist && shasum -a 256 -c SHA256SUMS)
GITHUB_TOKEN="$(gh auth token)"
test -n "${GITHUB_TOKEN:-}"
HOSTED_RECOVERY_DIR="$(mktemp -d)"
REPO_STATUS="$(curl -sS -L -o "$HOSTED_RECOVERY_DIR/repository.json" -w '%{http_code}' -H 'Accept: application/vnd.github+json' -H "Authorization: Bearer $GITHUB_TOKEN" https://api.github.com/repos/nutrik/vercor)"
test "$REPO_STATUS" = "200"
RELEASE_STATUS="$(curl -sS -L -o "$HOSTED_RECOVERY_DIR/release.json" -w '%{http_code}' -H 'Accept: application/vnd.github+json' -H "Authorization: Bearer $GITHUB_TOKEN" https://api.github.com/repos/nutrik/vercor/releases/tags/v0.4.0)"
test "$RELEASE_STATUS" = "200"
python tools/validate_release_state.py assets --json "$HOSTED_RECOVERY_DIR/release.json" --expect vercor-0.4.0-py3-none-any.whl
UNAFFECTED_ASSET_DIR="$(mktemp -d)"
gh release download v0.4.0 --repo nutrik/vercor --pattern vercor-0.4.0-py3-none-any.whl --dir "$UNAFFECTED_ASSET_DIR"
python tools/validate_release_state.py files --directory "$UNAFFECTED_ASSET_DIR" --manifest dist/SHA256SUMS --expect vercor-0.4.0-py3-none-any.whl
REMOTE_TAG_COMMIT="$(git ls-remote origin 'refs/tags/v0.4.0^{}' | awk '{print $1}')"
test "$REMOTE_TAG_COMMIT" = "$RELEASE_COMMIT"
IMMEDIATE_RELEASE_STATUS="$(curl -sS -L -o "$HOSTED_RECOVERY_DIR/immediate-release.json" -w '%{http_code}' -H 'Accept: application/vnd.github+json' -H "Authorization: Bearer $GITHUB_TOKEN" https://api.github.com/repos/nutrik/vercor/releases/tags/v0.4.0)"
test "$IMMEDIATE_RELEASE_STATUS" = "200"
python tools/validate_release_state.py assets --json "$HOSTED_RECOVERY_DIR/immediate-release.json" --expect vercor-0.4.0-py3-none-any.whl
IMMEDIATE_UNAFFECTED_ASSET_DIR="$(mktemp -d)"
gh release download v0.4.0 --repo nutrik/vercor --pattern vercor-0.4.0-py3-none-any.whl --dir "$IMMEDIATE_UNAFFECTED_ASSET_DIR"
python tools/validate_release_state.py files --directory "$IMMEDIATE_UNAFFECTED_ASSET_DIR" --manifest dist/SHA256SUMS --expect vercor-0.4.0-py3-none-any.whl
gh release upload v0.4.0 --repo nutrik/vercor dist/vercor-0.4.0.tar.gz
```

### Hosted release metadata correction

Use only when the tag and assets are correct and only title or notes differ:

```text
set -euo pipefail
test -n "${RELEASE_COMMIT:-}"
(cd dist && shasum -a 256 -c SHA256SUMS)
GITHUB_TOKEN="$(gh auth token)"
test -n "${GITHUB_TOKEN:-}"
HOSTED_RECOVERY_DIR="$(mktemp -d)"
REPO_STATUS="$(curl -sS -L -o "$HOSTED_RECOVERY_DIR/repository.json" -w '%{http_code}' -H 'Accept: application/vnd.github+json' -H "Authorization: Bearer $GITHUB_TOKEN" https://api.github.com/repos/nutrik/vercor)"
test "$REPO_STATUS" = "200"
RELEASE_STATUS="$(curl -sS -L -o "$HOSTED_RECOVERY_DIR/release.json" -w '%{http_code}' -H 'Accept: application/vnd.github+json' -H "Authorization: Bearer $GITHUB_TOKEN" https://api.github.com/repos/nutrik/vercor/releases/tags/v0.4.0)"
test "$RELEASE_STATUS" = "200"
python tools/validate_release_state.py assets --json "$HOSTED_RECOVERY_DIR/release.json" --expect vercor-0.4.0-py3-none-any.whl vercor-0.4.0.tar.gz
VERIFIED_ASSET_DIR="$(mktemp -d)"
gh release download v0.4.0 --repo nutrik/vercor --pattern vercor-0.4.0-py3-none-any.whl --dir "$VERIFIED_ASSET_DIR"
gh release download v0.4.0 --repo nutrik/vercor --pattern vercor-0.4.0.tar.gz --dir "$VERIFIED_ASSET_DIR"
python tools/validate_release_state.py files --directory "$VERIFIED_ASSET_DIR" --manifest dist/SHA256SUMS --expect vercor-0.4.0-py3-none-any.whl vercor-0.4.0.tar.gz
REMOTE_TAG_COMMIT="$(git ls-remote origin 'refs/tags/v0.4.0^{}' | awk '{print $1}')"
test "$REMOTE_TAG_COMMIT" = "$RELEASE_COMMIT"
IMMEDIATE_RELEASE_STATUS="$(curl -sS -L -o "$HOSTED_RECOVERY_DIR/immediate-release.json" -w '%{http_code}' -H 'Accept: application/vnd.github+json' -H "Authorization: Bearer $GITHUB_TOKEN" https://api.github.com/repos/nutrik/vercor/releases/tags/v0.4.0)"
test "$IMMEDIATE_RELEASE_STATUS" = "200"
python tools/validate_release_state.py assets --json "$HOSTED_RECOVERY_DIR/immediate-release.json" --expect vercor-0.4.0-py3-none-any.whl vercor-0.4.0.tar.gz
IMMEDIATE_VERIFIED_ASSET_DIR="$(mktemp -d)"
gh release download v0.4.0 --repo nutrik/vercor --pattern vercor-0.4.0-py3-none-any.whl --dir "$IMMEDIATE_VERIFIED_ASSET_DIR"
gh release download v0.4.0 --repo nutrik/vercor --pattern vercor-0.4.0.tar.gz --dir "$IMMEDIATE_VERIFIED_ASSET_DIR"
python tools/validate_release_state.py files --directory "$IMMEDIATE_VERIFIED_ASSET_DIR" --manifest dist/SHA256SUMS --expect vercor-0.4.0-py3-none-any.whl vercor-0.4.0.tar.gz
gh release edit v0.4.0 --repo nutrik/vercor --title "VerCOR 0.4.0" --notes-file docs/release-notes-0.4.0.md
```

For an incorrect hosted asset, verify the local replacement bundle first,
download only the selected remote asset, prove its digest differs from the
manifest-verified local replacement, and clobber only that selected asset.

### Replace differing hosted wheel asset

```text
set -euo pipefail
test -n "${RELEASE_COMMIT:-}"
test "${DESTRUCTIVE_ASSET_CLOBBER_APPROVED:-}" = "yes"
(cd dist && shasum -a 256 -c SHA256SUMS)
GITHUB_TOKEN="$(gh auth token)"
test -n "${GITHUB_TOKEN:-}"
HOSTED_RECOVERY_DIR="$(mktemp -d)"
REPO_STATUS="$(curl -sS -L -o "$HOSTED_RECOVERY_DIR/repository.json" -w '%{http_code}' -H 'Accept: application/vnd.github+json' -H "Authorization: Bearer $GITHUB_TOKEN" https://api.github.com/repos/nutrik/vercor)"
test "$REPO_STATUS" = "200"
RELEASE_STATUS="$(curl -sS -L -o "$HOSTED_RECOVERY_DIR/release.json" -w '%{http_code}' -H 'Accept: application/vnd.github+json' -H "Authorization: Bearer $GITHUB_TOKEN" https://api.github.com/repos/nutrik/vercor/releases/tags/v0.4.0)"
test "$RELEASE_STATUS" = "200"
python tools/validate_release_state.py assets --json "$HOSTED_RECOVERY_DIR/release.json" --expect vercor-0.4.0-py3-none-any.whl vercor-0.4.0.tar.gz
DOWNLOADED_SELECTED_DIR="$(mktemp -d)"
DOWNLOADED_UNAFFECTED_DIR="$(mktemp -d)"
gh release download v0.4.0 --repo nutrik/vercor --pattern vercor-0.4.0-py3-none-any.whl --dir "$DOWNLOADED_SELECTED_DIR"
gh release download v0.4.0 --repo nutrik/vercor --pattern vercor-0.4.0.tar.gz --dir "$DOWNLOADED_UNAFFECTED_DIR"
python tools/validate_release_state.py differs --file "$DOWNLOADED_SELECTED_DIR/vercor-0.4.0-py3-none-any.whl" --manifest dist/SHA256SUMS --name vercor-0.4.0-py3-none-any.whl
python tools/validate_release_state.py files --directory "$DOWNLOADED_UNAFFECTED_DIR" --manifest dist/SHA256SUMS --expect vercor-0.4.0.tar.gz
REMOTE_TAG_COMMIT="$(git ls-remote origin 'refs/tags/v0.4.0^{}' | awk '{print $1}')"
test "$REMOTE_TAG_COMMIT" = "$RELEASE_COMMIT"
IMMEDIATE_RELEASE_STATUS="$(curl -sS -L -o "$HOSTED_RECOVERY_DIR/immediate-release.json" -w '%{http_code}' -H 'Accept: application/vnd.github+json' -H "Authorization: Bearer $GITHUB_TOKEN" https://api.github.com/repos/nutrik/vercor/releases/tags/v0.4.0)"
test "$IMMEDIATE_RELEASE_STATUS" = "200"
python tools/validate_release_state.py assets --json "$HOSTED_RECOVERY_DIR/immediate-release.json" --expect vercor-0.4.0-py3-none-any.whl vercor-0.4.0.tar.gz
IMMEDIATE_SELECTED_DIR="$(mktemp -d)"
IMMEDIATE_UNAFFECTED_DIR="$(mktemp -d)"
gh release download v0.4.0 --repo nutrik/vercor --pattern vercor-0.4.0-py3-none-any.whl --dir "$IMMEDIATE_SELECTED_DIR"
gh release download v0.4.0 --repo nutrik/vercor --pattern vercor-0.4.0.tar.gz --dir "$IMMEDIATE_UNAFFECTED_DIR"
python tools/validate_release_state.py differs --file "$IMMEDIATE_SELECTED_DIR/vercor-0.4.0-py3-none-any.whl" --manifest dist/SHA256SUMS --name vercor-0.4.0-py3-none-any.whl
python tools/validate_release_state.py files --directory "$IMMEDIATE_UNAFFECTED_DIR" --manifest dist/SHA256SUMS --expect vercor-0.4.0.tar.gz
gh release upload v0.4.0 --repo nutrik/vercor --clobber dist/vercor-0.4.0-py3-none-any.whl
```

### Replace differing hosted sdist asset

```text
set -euo pipefail
test -n "${RELEASE_COMMIT:-}"
test "${DESTRUCTIVE_ASSET_CLOBBER_APPROVED:-}" = "yes"
(cd dist && shasum -a 256 -c SHA256SUMS)
GITHUB_TOKEN="$(gh auth token)"
test -n "${GITHUB_TOKEN:-}"
HOSTED_RECOVERY_DIR="$(mktemp -d)"
REPO_STATUS="$(curl -sS -L -o "$HOSTED_RECOVERY_DIR/repository.json" -w '%{http_code}' -H 'Accept: application/vnd.github+json' -H "Authorization: Bearer $GITHUB_TOKEN" https://api.github.com/repos/nutrik/vercor)"
test "$REPO_STATUS" = "200"
RELEASE_STATUS="$(curl -sS -L -o "$HOSTED_RECOVERY_DIR/release.json" -w '%{http_code}' -H 'Accept: application/vnd.github+json' -H "Authorization: Bearer $GITHUB_TOKEN" https://api.github.com/repos/nutrik/vercor/releases/tags/v0.4.0)"
test "$RELEASE_STATUS" = "200"
python tools/validate_release_state.py assets --json "$HOSTED_RECOVERY_DIR/release.json" --expect vercor-0.4.0-py3-none-any.whl vercor-0.4.0.tar.gz
DOWNLOADED_SELECTED_DIR="$(mktemp -d)"
DOWNLOADED_UNAFFECTED_DIR="$(mktemp -d)"
gh release download v0.4.0 --repo nutrik/vercor --pattern vercor-0.4.0.tar.gz --dir "$DOWNLOADED_SELECTED_DIR"
gh release download v0.4.0 --repo nutrik/vercor --pattern vercor-0.4.0-py3-none-any.whl --dir "$DOWNLOADED_UNAFFECTED_DIR"
python tools/validate_release_state.py differs --file "$DOWNLOADED_SELECTED_DIR/vercor-0.4.0.tar.gz" --manifest dist/SHA256SUMS --name vercor-0.4.0.tar.gz
python tools/validate_release_state.py files --directory "$DOWNLOADED_UNAFFECTED_DIR" --manifest dist/SHA256SUMS --expect vercor-0.4.0-py3-none-any.whl
REMOTE_TAG_COMMIT="$(git ls-remote origin 'refs/tags/v0.4.0^{}' | awk '{print $1}')"
test "$REMOTE_TAG_COMMIT" = "$RELEASE_COMMIT"
IMMEDIATE_RELEASE_STATUS="$(curl -sS -L -o "$HOSTED_RECOVERY_DIR/immediate-release.json" -w '%{http_code}' -H 'Accept: application/vnd.github+json' -H "Authorization: Bearer $GITHUB_TOKEN" https://api.github.com/repos/nutrik/vercor/releases/tags/v0.4.0)"
test "$IMMEDIATE_RELEASE_STATUS" = "200"
python tools/validate_release_state.py assets --json "$HOSTED_RECOVERY_DIR/immediate-release.json" --expect vercor-0.4.0-py3-none-any.whl vercor-0.4.0.tar.gz
IMMEDIATE_SELECTED_DIR="$(mktemp -d)"
IMMEDIATE_UNAFFECTED_DIR="$(mktemp -d)"
gh release download v0.4.0 --repo nutrik/vercor --pattern vercor-0.4.0.tar.gz --dir "$IMMEDIATE_SELECTED_DIR"
gh release download v0.4.0 --repo nutrik/vercor --pattern vercor-0.4.0-py3-none-any.whl --dir "$IMMEDIATE_UNAFFECTED_DIR"
python tools/validate_release_state.py differs --file "$IMMEDIATE_SELECTED_DIR/vercor-0.4.0.tar.gz" --manifest dist/SHA256SUMS --name vercor-0.4.0.tar.gz
python tools/validate_release_state.py files --directory "$IMMEDIATE_UNAFFECTED_DIR" --manifest dist/SHA256SUMS --expect vercor-0.4.0-py3-none-any.whl
gh release upload v0.4.0 --repo nutrik/vercor --clobber dist/vercor-0.4.0.tar.gz
```

Once any 0.4.0 package artifact is public, preserve the published tag and
evidence. Correct substantive mistakes in a new patch release.
