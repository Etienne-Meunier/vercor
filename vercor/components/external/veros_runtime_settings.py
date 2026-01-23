from veros import runtime_settings  # type: ignore[import]

"""
Hardcoded peace of code setting Veros runtime settings
"""

setattr(runtime_settings, "backend", "numpy")
setattr(runtime_settings, "force_overwrite", True)
# setattr(runtime_settings, 'linear_solver', 'scipy_jax')
# setattr(runtime_settings, 'device', 'cpu')
