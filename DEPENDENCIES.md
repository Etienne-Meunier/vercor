1. `vercor/settings.py` - physical constants and runtime settings consumed by translated kernels
2. `vercor/fluxes/utilities.py` - JAX-native scalar/array thermodynamic helpers built on (1)
3. `vercor/fluxes/bulk_formula_cesm.py` - JAX-native atmosphere-ocean / atmosphere-ice bulk flux kernels built on (1, 2)
4. `vercor/host_arrays.py` - explicit JAX/NumPy host-transfer boundary for non-differentiable adapters and output
5. `vercor/components/external/jax_gcm_tools.py` - existing JAX helper layer used by the JCM adapter; validated for `jax.jit`
6. `vercor/components/external/jax_gcm.py` - JCM adapter boundary that stores translated kernel outputs
7. `vercor/components/external/veros_gcm.py` - Veros adapter boundary that converts translated flux outputs back to NumPy for Veros state updates
8. `vercor/components/external/camulator.py` - CAMulator adapter boundary with JAX-backed runtime-field helpers and explicit Torch / xarray output boundaries
9. `vercor/grid.py` - JAX-friendly `RectilinearGrid` holder with eager validation and PyTree registration
10. `vercor/regridders/helpers.py` - JAX-native rectilinear helper kernels built on (9)
11. `vercor/interpolators/bilinear_rectilinear.py` - JAX-native bilinear scalar/vector interpolation and extrapolation built on (9, 10)
12. `vercor/interpolators/conservative_remap_rectilinear.py` - JAX-native conservative scalar remapping runtime with eager overlap preprocessing built on (9, 10)
13. `vercor/regridders/conservative.py` - conservative regridder wrapper over (12)
14. `vercor/grid_masks.py` - grid identity, component lookup, land/ocean mask construction, and remap-conservation checks built on (9, 13)
15. `vercor/assets.py` - forcing asset cache/download/checksum boundary for data components
16. `vercor/time_selection.py` - calendar, day-slice, and periodic interpolation index helpers
17. `vercor/components/slab/atmosphere.py` - slab atmosphere wrapper over pure JAX bulk-flux, default-SST, and wind kernels built on (9)
18. `vercor/components/slab/ocean.py` - slab ocean wrapper over pure JAX SST tendency kernel built on (9)
19. `vercor/components/slab/land.py` - slab land wrapper over pure JAX soil-moisture update kernel built on (9)
20. `vercor/components/slab/seaice.py` - slab sea-ice wrapper over pure JAX ice-fraction diagnostic kernel built on (9)
21. `vercor/components/data/era5_atmosphere.py` - ERA5 atmospheric forcing adapter with JAX-backed pressure/diagnostic helpers built on (2, 9, 15)
22. `vercor/components/data/era5_ocean.py` - ERA5 ocean forcing adapter with JAX-backed mask and SST application built on (9, 15)
23. `vercor/components/data/era5_land.py` - ERA5 land forcing adapter with JAX-backed mask preparation and runtime temperature storage built on (9, 15)
24. `vercor/components/data/erainterim_ocean.py` - ERA-Interim ocean forcing adapter with JAX-backed global field assembly built on (9, 15)
25. `vercor/components/data/jcm_land.py` - JCM land forcing adapter with JAX-backed coordinate conversion and runtime storage built on (9, 14)
26. `vercor/components/data/camulator_land.py` - CAMulator land forcing adapter with JAX-backed runtime temperature storage built on (8, 9, 14)
27. `vercor/runtime_views.py` - explicit runtime component metadata/field view used by diagnostics and output built on (9)
28. `vercor/diagnostics.py` - runtime-view means tables and plotting helpers built on (4, 27)
29. `vercor/components/base.py` - slim component interface for seed data, component-specific runtime hooks, explicit component init/runtime step contexts, public host-runtime adapter contract, and pure stepping built on (9)
30. `vercor/runtime.py` - immutable runtime state, coupler-owned component-name metadata, runtime field stores, and pure exchange dispatch built on (9, 11, 12, 29)
31. `vercor/runtime_contracts.py` - exchange-field flattening and coupler-owned runtime import/export contract construction from component-name sequences built on (30)
32. `vercor/runtime_time.py` - host-precomputed daily/monthly runtime step metadata built on clock/settings helpers and (16, 30)
33. `vercor/runtime_components.py` - generic component runtime state creation, explicit contract prefill/validation, receive/send, and time-selection helpers built on (29, 30, 31)
34. `vercor/runtime_driver.py` - bundled runtime dispatch context, per-component dispatch, receive, pure/host step selection, outgoing priming, and host-adapter detection built on (29, 30, 33)
35. `vercor/output.py` - runtime-view NetCDF output boundary built on (4, 9, 27, 30)
36. `vercor/components/external/jax_gcm.py` runtime payload path - immutable JCM state/forcing runtime integration built on (6, 29, 30, 33)
37. `vercor/coupler.py` unified runtime facade - canonical `run()` / `create_runtime_state()` path, component registration, exchange-mask/regridder setup, runtime component views, final output, and scanned-runtime rejection for host adapters built on (13, 14, 27, 30, 31, 32, 33, 34, 35, 36)
