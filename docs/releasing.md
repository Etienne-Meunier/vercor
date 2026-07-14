# Releasing VerCOR

This is the verification procedure for a release candidate. Preparing a
candidate does not authorize a tag, push, upload, or publication.

## 1. Confirm the candidate

- Work from a clean release branch with complete history.
- Confirm `pyproject.toml` and `CHANGELOG.md` use the intended version.
- Confirm the package root and canonical owner manifests match live signatures.
- Confirm optional JCM and Veros versions in the verification environment.
- Leave CAMulator uninstalled and unpinned until an exact compatible release is
  verified.

## 2. Run source gates

Use the supported environment binaries directly if the Conda launcher is
unavailable:

```bash
python -m black --check vercor examples tests
python -m flake8 . --count --max-line-length=120 --statistics
python -m mypy vercor examples tests
python -m compileall -q vercor examples tests
python -m pytest tests/ -q --fast --tb=short
python -m pytest tests/ -q --tb=short
python -m pytest tests/ -q --cov=vercor --cov-branch --cov-report=term-missing --cov-fail-under=90
git diff --check
```

## 3. Build once

Build VerCOR plus both plugin evidence artifacts from the verified source:

```bash
python -m build --outdir dist
python -m build --wheel --outdir dist tests/fixtures/public_plugin
python -m build --wheel --outdir dist tests/fixtures/public_plugin_3_0
```

Inspect wheel and source-distribution metadata, filenames, `vercor/py.typed`,
and canonical owner manifests. Install the wheel and source distribution in
separate clean targets outside the checkout.

Run the executable artifact boundary against exactly that bundle. It inspects
both archives for metadata, manifests, signatures, PEP 561 markers, and
forbidden cache/platform files; installs the wheel and sdist in separate clean
targets; runs the slab and native plugin; and runs strict installed-plugin
mypy. The frozen v3 wheel is inspected as historical metadata only.

```bash
VERCOR_ARTIFACT_DIR="$(pwd)/dist" python -m pytest tests/test_distribution_boundaries.py -q --tb=short
```

Record candidate hashes and the locally tested optional-model versions:

```bash
shasum -a 256 dist/vercor-4.0.0a1-py3-none-any.whl dist/vercor-4.0.0a1.tar.gz dist/vercor_public_plugin-0.1.0-py3-none-any.whl dist/vercor_compat_plugin_3_0-0.1.0-py3-none-any.whl
python -c 'import importlib.metadata as m; print("JCM", m.version("jcm")); print("Veros", m.version("veros"))'
```

## 4. Run installed-artifact gates

- Run base, JCM, and Veros lanes on Python 3.12 and 3.13.
- Run the installed v4 public plugin and strict mypy use site on both Python
  versions.
- Inspect the frozen v3 plugin wheel as historical metadata/source evidence;
  do not execute it against v4.
- Run the dependency-free slab and public-plugin smoke on macOS.
- Confirm JVP and reverse gradients with `output=None` and confirm that it
  creates no files.

Run the same bounded JCM/Veros nodes used by CI and the explicit output-free
gradient acceptance:

```bash
python -m pytest tests/test_setup_lifecycle_helpers.py::test_make_jcm_land_atmosphere_replaces_only_missing_forcing tests/test_external_components_coverage.py::test_jax_gcm_initialize_uses_provided_forcing_and_can_spin_up tests/test_external_components_coverage.py::test_jax_gcm_initialize_builds_default_forcing_when_missing tests/test_setup_boundaries.py::test_veros_implementation_import_does_not_configure_runtime tests/test_setup_boundaries.py::test_veros_factory_configures_once_before_implementation_import tests/test_external_components_coverage.py::test_veros_initialize_spinup_follows_enabled_only -q --tb=short
python -m pytest tests/test_v4_workflow_execution.py::test_output_free_workflow_preserves_jvp_and_reverse_mode_gradients tests/test_v4_workflow_execution.py::test_payload_dependent_multi_step_scan_preserves_treedef_jvp_and_grad tests/test_v4_output_providers.py::test_all_disabled_target_remains_jit_and_gradient_compatible -q --tb=short
```

On the supported macOS release runner, verify the installed wheel and native
plugin outside the checkout:

```bash
python -m pip install "dist/vercor-4.0.0a1-py3-none-any.whl"
python -m pip install --no-deps "dist/vercor_public_plugin-0.1.0-py3-none-any.whl"
cd "$(mktemp -d)"
python -m vercor_public_plugin.smoke --output-dir "$(pwd)/plugin-output"
```

CI consumes the single uploaded artifact bundle; matrix cells must not rebuild
the checkout.

## 5. Handoff

Record exact commands, counts, warnings, artifact hashes, and environment
versions in `PROGRESS.md` or the task report. Stop after candidate preparation.
Tagging, pushing, signing, release creation, and package-index upload require a
separate explicit authorization.
