# Changelog

All notable changes to VerCOR are recorded here. The project follows semantic
versioning; alpha releases may still refine the new 0.4 contracts.

## [0.4.0a1] - 2026-07-14

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
- Bundled slab, JCM, Veros, and CAMulator factories use the 0.4 component and
  output contracts.
- Bundled slab and data factories accept per-component `OutputSpec` overrides
  while retaining step-period output by default.

### Removed

- Primary 0.3 aliases, settings, authoring mixins, coupler recipes/mutators,
  callable-derived route identity, backend-owned output, and public preparation
  internals.
- Duplicate native/generic output accumulators and hidden output markers.

### Compatibility

This alpha does not ship legacy adapters. Follow
`docs/migration-0.3-to-0.4.md` to migrate 0.3-only workflows directly.

[0.4.0a1]: https://github.com/Roman-N/VerCOR/compare/v0.3.2...v0.4.0a1
