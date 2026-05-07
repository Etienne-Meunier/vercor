1. `vercor/dtypes.py` - canonical JAX/NumPy dtype policy and array-construction helpers
2. `vercor/settings.py` - unified metadata-backed `VercorSettings` container with dynamic attribute access, typed known-setting annotations, default settings records, physical constants, runtime/component settings, and settings-bound dtype policy consumed by translated kernels
3. `vercor/fluxes/utilities.py` - JAX-native scalar/array thermodynamic helpers built on (1, 2)
4. `vercor/fluxes/bulk_formula_cesm.py` - JAX-native atmosphere-ocean / atmosphere-ice bulk flux kernels built on (1, 2, 3)
5. `vercor/host_arrays.py` - explicit JAX/NumPy host-transfer boundary for non-differentiable adapters and output
6. `vercor/components/external/jax_gcm_tools.py` - existing JAX helper layer used by the JCM adapter; validated for `jax.jit` and built on (1)
7. `vercor/components/external/jax_gcm.py` - JCM adapter boundary that stores translated kernel outputs built on (1, 6)
8. `vercor/components/external/veros_runtime_settings.py` and `vercor/components/external/veros_gcm.py` - explicit Veros host-runtime configuration plus the Veros adapter boundary that converts translated flux outputs back to NumPy for Veros state updates built on (1, 4, 5)
9. `vercor/components/external/camulator.py` - CAMulator adapter boundary with JAX-backed runtime-field helpers and explicit Torch / xarray output boundaries built on (1, 5)
10. `vercor/grid.py` - JAX-friendly `RectilinearGrid` holder with eager validation and PyTree registration
11. `vercor/field_layout.py` - canonical component data-field layout validation and time-last forcing normalization helpers built on (10)
12. `vercor/regridders/helpers.py` - JAX-native rectilinear helper kernels built on (1, 10)
13. `vercor/interpolators/bilinear_rectilinear.py` - JAX-native bilinear scalar/vector interpolation and extrapolation built on (1, 10, 12)
14. `vercor/interpolators/conservative_remap_rectilinear.py` - JAX-native conservative scalar remapping runtime with eager overlap preprocessing built on (1, 10, 12)
15. `vercor/regridders/conservative.py` - conservative regridder wrapper over (14)
16. `vercor/grid_masks.py` - grid identity, component lookup, land/ocean mask construction, and remap-conservation checks built on (1, 10, 15)
17. `vercor/assets.py` - forcing asset cache/download/checksum boundary for data components
18. `vercor/forcing_data.py` - NetCDF forcing-file read boundary for data components
19. `vercor/time_selection.py` - calendar, day-slice, and periodic interpolation index helpers
20. `vercor/components/slab/atmosphere.py` - slab atmosphere wrapper over pure JAX bulk-flux, default-SST, and wind kernels built on (1, 10)
21. `vercor/components/slab/ocean.py` - slab ocean wrapper over pure JAX SST tendency kernel built on (1, 10)
22. `vercor/components/slab/land.py` - slab land wrapper over pure JAX soil-moisture update kernel built on (1, 10)
23. `vercor/components/slab/seaice.py` - slab sea-ice wrapper over pure JAX ice-fraction diagnostic kernel built on (1, 10)
24. `vercor/components/data/era5_atmosphere.py` - pure ERA5 atmospheric data component with canonical data-field layout and JAX-backed pressure/model-level diagnostic initialization built on (3, 10, 11, 17, 18)
25. `vercor/components/data/era5_ocean.py` - ERA5 ocean forcing adapter with canonical data-field layout, JAX-backed mask, and SST application built on (10, 11, 17, 18)
26. `vercor/components/data/era5_land.py` - ERA5 land forcing adapter with canonical data-field layout, JAX-backed mask preparation, and runtime temperature storage built on (10, 11, 17, 18)
27. `vercor/components/data/erainterim_ocean.py` - ERA-Interim ocean forcing adapter with canonical data-field layout and JAX-backed global field assembly built on (10, 11, 17, 18)
28. `vercor/components/data/jcm_land.py` - JCM land forcing adapter with canonical data-field layout, JAX-backed coordinate conversion, and runtime storage built on (10, 11, 16)
29. `vercor/components/data/camulator_land.py` - CAMulator land forcing adapter with JAX-backed runtime temperature storage and forcing-only CAMulator config loading built on (1, 9, 10, 16)
30. `vercor/jax_logging.py` - callback-backed logger protocol and setup helper for Python and traced JAX runtime diagnostics
31. `vercor/runtime/interrupts.py` - internal terminal-signal runtime cancellation controller with host and JAX callback checkpoints, plus wakeup-fd polling for compiled runtime signals, built on JAX callback errors and Python signal handling
32. `vercor/runtime/contexts.py` - immutable component initialization and runtime step context payloads built on clock, run-sequence, settings helpers, and (30)
33. `vercor/runtime/views.py` - explicit runtime component metadata/field view used by diagnostics and output built on (10)
34. `vercor/diagnostics.py` - runtime-view means tables, plotting helpers, and plotting-only derived field helpers built on (5, 33)
35. `vercor/components/base.py` - slim component-author interfaces for active differentiable components, data-only components, host-runtime adapters, `from_fields()` / `from_model()` authoring facade, `ComponentFieldSpec` declarations, backward-compatible callable/data wrappers, scalar-to-grid field defaults, seed/runtime field helpers, setup validation, explicit runtime contexts, canonical data-field validation, and finalized runtime boundary hooks built on (1, 10, 11, 32)
36. `vercor/runtime/state.py` and `vercor/runtime/__init__.py` - immutable runtime state, coupler-owned component-name metadata, runtime field stores, import/export contract construction, pure exchange dispatch, and the internal `vercor.runtime` re-export surface built on (1, 10, 13, 14, 35)
37. `vercor/runtime/time.py` - host-precomputed daily/monthly runtime step metadata built on clock/settings helpers and (19, 36)
38. `vercor/runtime/components.py` - generic component runtime state creation, explicit contract prefill/validation, canonical data-field validation, receive/send, and time-selection helpers built on (1, 11, 35, 36)
39. `vercor/runtime/driver.py` - bundled runtime dispatch context, single per-component runtime step helper with explicit host-runtime allowance, outgoing priming, host-adapter detection, and callback-safe runtime logging built on (32, 35, 36, 38)
40. `vercor/output.py` - runtime-view NetCDF output boundary built on (5, 10, 33, 36)
41. `vercor/components/external/jax_gcm.py` runtime payload path - immutable JCM state/forcing runtime integration built on (7, 32, 35, 36, 38)
42. `vercor/coupler.py` unified runtime facade - canonical `run()` / `create_runtime_state()` path, component registration through the base component contract, exchange-mask/regridder setup, runtime component views, final output, callback-backed logging setup, unified host/scanned runtime interruption, and scanned-runtime rejection for host adapters built on (1, 15, 16, 30, 31, 32, 33, 36, 37, 38, 39, 40, 41)
