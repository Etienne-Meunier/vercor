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
11. Pending next slice: `vercor/interpolators/conservative_remap_rectilinear.py`
12. Pending third slice: `vercor/components/slab/atmosphere.py`
13. Pending third slice: `vercor/components/slab/ocean.py`
14. Pending third slice: `vercor/components/slab/land.py`
15. Pending third slice: `vercor/components/slab/seaice.py`
