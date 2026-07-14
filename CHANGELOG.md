# Changelog

All notable changes to VerCOR are recorded here. The project follows semantic
versioning; alpha releases may still refine the new v4 contracts.

## [4.0.0a1] - 2026-07-14

### Added

- Structural component authoring with immutable declarations and setup results.
- Stable exchange route IDs, route-keyed topology, and scalar/vector regridder
  capabilities.
- Validated workflow plans, chunk-oriented execution backends, and a public
  runtime driver.
- Unified provider, period, target, and snapshot output contracts.
- Frozen traced `PhysicalConstants` and setup-owned frozen configuration.
- Installed wheel, source-distribution, public-plugin, optional-model, and
  macOS release gates.

### Changed

- The package root now exports exactly six primary conveniences.
- Coupler assembly is constructor-only and prepared configuration is private.
- `RunState` is opaque and exposes immutable field replacement.
- Runtime precision is owned only by `RuntimeOptions.dtype`.
- Bundled slab, JCM, Veros, and CAMulator factories use the v4 component and
  output contracts.

### Removed

- Primary v3 aliases, settings, authoring mixins, coupler recipes/mutators,
  callable-derived route identity, backend-owned output, and public preparation
  internals.
- Duplicate native/generic output accumulators and hidden output markers.

### Compatibility

This alpha does not ship legacy adapters. Follow
`docs/migration-3-to-4.md`. The frozen v3 manifest and v3 plugin fixture are
historical evidence only.

[4.0.0a1]: https://github.com/Roman-N/VerCOR/compare/v3.1.1...v4.0.0a1
