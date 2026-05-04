1. `vercor/settings.py` - physical constants and runtime settings consumed by translated kernels
2. `vercor/fluxes/utilities.py` - JAX-native scalar/array thermodynamic helpers built on (1)
3. `vercor/fluxes/bulk_formula_cesm.py` - JAX-native atmosphere-ocean / atmosphere-ice bulk flux kernels built on (1, 2)
4. `vercor/host_arrays.py` - explicit JAX/NumPy host-transfer boundary for non-differentiable adapters and output
5. `vercor/components/external/jax_gcm_tools.py` - existing JAX helper layer used by the JCM adapter; validated for `jax.jit`
6. `vercor/components/external/jax_gcm.py` - JCM adapter boundary that stores translated kernel outputs
7. `vercor/components/external/veros_runtime_settings.py` and `vercor/components/external/veros_gcm.py` - explicit Veros host-runtime configuration plus the Veros adapter boundary that converts translated flux outputs back to NumPy for Veros state updates
8. `vercor/components/external/camulator.py` - CAMulator adapter boundary with JAX-backed runtime-field helpers and explicit Torch / xarray output boundaries
9. `vercor/grid.py` - JAX-friendly `RectilinearGrid` holder with eager validation and PyTree registration
10. `vercor/field_layout.py` - canonical component data-field layout validation and time-last forcing normalization helpers built on (9)
11. `vercor/regridders/helpers.py` - JAX-native rectilinear helper kernels built on (9)
12. `vercor/interpolators/bilinear_rectilinear.py` - JAX-native bilinear scalar/vector interpolation and extrapolation built on (9, 11)
13. `vercor/interpolators/conservative_remap_rectilinear.py` - JAX-native conservative scalar remapping runtime with eager overlap preprocessing built on (9, 11)
14. `vercor/regridders/conservative.py` - conservative regridder wrapper over (13)
15. `vercor/grid_masks.py` - grid identity, component lookup, land/ocean mask construction, and remap-conservation checks built on (9, 14)
16. `vercor/assets.py` - forcing asset cache/download/checksum boundary for data components
17. `vercor/forcing_data.py` - NetCDF forcing-file read boundary for data components
18. `vercor/time_selection.py` - calendar, day-slice, and periodic interpolation index helpers
19. `vercor/components/slab/atmosphere.py` - slab atmosphere wrapper over pure JAX bulk-flux, default-SST, and wind kernels built on (9)
20. `vercor/components/slab/ocean.py` - slab ocean wrapper over pure JAX SST tendency kernel built on (9)
21. `vercor/components/slab/land.py` - slab land wrapper over pure JAX soil-moisture update kernel built on (9)
22. `vercor/components/slab/seaice.py` - slab sea-ice wrapper over pure JAX ice-fraction diagnostic kernel built on (9)
23. `vercor/components/data/era5_atmosphere.py` - pure ERA5 atmospheric data component with canonical data-field layout and JAX-backed pressure/model-level diagnostic initialization built on (2, 9, 10, 16, 17)
24. `vercor/components/data/era5_ocean.py` - ERA5 ocean forcing adapter with canonical data-field layout, JAX-backed mask, and SST application built on (9, 10, 16, 17)
25. `vercor/components/data/era5_land.py` - ERA5 land forcing adapter with canonical data-field layout, JAX-backed mask preparation, and runtime temperature storage built on (9, 10, 16, 17)
26. `vercor/components/data/erainterim_ocean.py` - ERA-Interim ocean forcing adapter with canonical data-field layout and JAX-backed global field assembly built on (9, 10, 16, 17)
27. `vercor/components/data/jcm_land.py` - JCM land forcing adapter with canonical data-field layout, JAX-backed coordinate conversion, and runtime storage built on (9, 10, 15)
28. `vercor/components/data/camulator_land.py` - CAMulator land forcing adapter with JAX-backed runtime temperature storage and forcing-only CAMulator config loading built on (8, 9, 15)
29. `vercor/jax_logging.py` - callback-backed logger protocol and setup helper for Python and traced JAX runtime diagnostics
30. `vercor/runtime/contexts.py` - immutable component initialization and runtime step context payloads built on clock, run-sequence, settings helpers, and (29)
31. `vercor/runtime/views.py` - explicit runtime component metadata/field view used by diagnostics and output built on (9)
32. `vercor/diagnostics.py` - runtime-view means tables, plotting helpers, and plotting-only derived field helpers built on (4, 31)
33. `vercor/components/base.py` - slim component-author interfaces for active differentiable components, data-only components, host-runtime adapters, seed data, setup validation, explicit runtime contexts, canonical data-field validation, and finalized runtime boundary hooks built on (9, 10, 30)
34. `vercor/runtime/state.py` and `vercor/runtime/__init__.py` - immutable runtime state, coupler-owned component-name metadata, runtime field stores, import/export contract construction, pure exchange dispatch, and the internal `vercor.runtime` re-export surface built on (9, 12, 13, 33)
35. `vercor/runtime/time.py` - host-precomputed daily/monthly runtime step metadata built on clock/settings helpers and (18, 34)
36. `vercor/runtime/components.py` - generic component runtime state creation, explicit contract prefill/validation, canonical data-field validation, receive/send, and time-selection helpers built on (10, 33, 34)
37. `vercor/runtime/driver.py` - bundled runtime dispatch context, single per-component runtime step helper with explicit host-runtime allowance, outgoing priming, host-adapter detection, and callback-safe runtime logging built on (30, 33, 34, 36)
38. `vercor/output.py` - runtime-view NetCDF output boundary built on (4, 9, 31, 34)
39. `vercor/components/external/jax_gcm.py` runtime payload path - immutable JCM state/forcing runtime integration built on (6, 30, 33, 34, 36)
40. `vercor/coupler.py` unified runtime facade - canonical `run()` / `create_runtime_state()` path, component registration through the base component contract, exchange-mask/regridder setup, runtime component views, final output, callback-backed logging setup, and scanned-runtime rejection for host adapters built on (14, 15, 29, 30, 31, 34, 35, 36, 37, 38, 39)
