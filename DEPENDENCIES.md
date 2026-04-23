1. `vercor/settings.py` - physical constants and runtime settings consumed by translated kernels
2. `vercor/fluxes/utilities.py` - JAX-native scalar/array thermodynamic helpers built on (1)
3. `vercor/fluxes/bulk_formula_cesm.py` - JAX-native atmosphere-ocean / atmosphere-ice bulk flux kernels built on (1, 2)
4. `vercor/components/external/jax_gcm_tools.py` - existing JAX helper layer used by the JCM adapter; validated for `jax.jit`
5. `vercor/components/external/jax_gcm.py` - JCM adapter boundary that stores translated kernel outputs
6. `vercor/components/external/veros_gcm.py` - Veros adapter boundary that converts translated flux outputs back to NumPy for Veros state updates
7. `vercor/components/external/camulator.py` - CAMulator adapter boundary that converts translated thermodynamic outputs back to NumPy / Torch-facing storage
8. `vercor/grid.py` - JAX-friendly `RectilinearGrid` holder with eager validation and PyTree registration
9. `vercor/regridders/helpers.py` - JAX-native rectilinear helper kernels built on (8)
10. `vercor/interpolators/bilinear_rectilinear.py` - JAX-native bilinear scalar/vector interpolation and extrapolation built on (8, 9)
11. `vercor/interpolators/conservative_remap_rectilinear.py` - JAX-native conservative scalar remapping runtime with eager overlap preprocessing built on (8, 9)
12. `vercor/regridders/conservative.py` - conservative regridder wrapper over (11)
13. `vercor/components/slab/atmosphere.py` - slab atmosphere wrapper over pure JAX bulk-flux, default-SST, and wind kernels built on (8)
14. `vercor/components/slab/ocean.py` - slab ocean wrapper over pure JAX SST tendency kernel built on (8)
15. `vercor/components/slab/land.py` - slab land wrapper over pure JAX soil-moisture update kernel built on (8)
16. `vercor/components/slab/seaice.py` - slab sea-ice wrapper over pure JAX ice-fraction diagnostic kernel built on (8)
