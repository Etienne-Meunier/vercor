from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from vercor.host_arrays import runtime_array_to_host
from vercor.runtime_views import RuntimeComponentView
from vercor.types import RuntimeArray


def safe_component_nanmean(component: RuntimeComponentView, field_name: str) -> float:
    """Return a robust NaN-aware mean for a runtime component view field."""

    try:
        return float(jnp.nanmean(jnp.asarray(_view_field(component, field_name))))
    except Exception:
        return float("nan")


def _view_field_candidates(
    component: RuntimeComponentView, field_name: str
) -> list[RuntimeArray]:
    """Return matching fields from an explicit runtime component view."""

    candidates: list[RuntimeArray] = []
    for store in (component.data, component.incoming, component.outgoing):
        if field_name in store.field_names:
            candidates.append(store.get(field_name))
    return candidates


def _view_field(component: RuntimeComponentView, field_name: str) -> RuntimeArray:
    """Return a field from an explicit runtime component view."""

    candidates = _view_field_candidates(component, field_name)
    if candidates:
        return candidates[0]
    raise KeyError(f"Field {field_name!r} not found")


def _component_plot_field(
    component: RuntimeComponentView, field_name: str
) -> RuntimeArray:
    """Return a 2D field suitable for plotting when one is available."""

    candidates = _view_field_candidates(component, field_name)
    for candidate in candidates:
        if jnp.asarray(candidate).ndim == 2:
            return candidate
    if candidates:
        return candidates[0]
    raise KeyError(f"Field {field_name!r} not found")


def _safe_component_metric_mean(
    component: RuntimeComponentView,
    metric: str | Callable[[RuntimeComponentView], RuntimeArray | float],
) -> float:
    """Resolve a metric and return a robust mean value as float."""

    if isinstance(metric, str):
        return safe_component_nanmean(component, metric)

    try:
        return float(jnp.nanmean(jnp.asarray(metric(component))))
    except Exception:
        return float("nan")


def print_component_field_means_table(
    components: Mapping[str, RuntimeComponentView],
    fields: Sequence[
        tuple[str | Callable[[RuntimeComponentView], RuntimeArray | float], str]
    ],
    component_order: Sequence[str] | None = None,
) -> None:
    """Print a means table for component fields with configurable column order."""

    ordered_names = list(component_order or components.keys())
    ordered_names = [name for name in ordered_names if name in components]

    first_col_width = max(10, max((len(label) for _, label in fields), default=10))
    value_col_width = 15

    header = f"{'Variable':<{first_col_width}} " + " ".join(
        f"{name:>{value_col_width}}" for name in ordered_names
    )
    print(header)
    print("-" * len(header))

    for field_name, label in fields:
        values = [
            _safe_component_metric_mean(components[name], field_name)
            for name in ordered_names
        ]
        value_text = " ".join(f"{value:>{value_col_width}.4f}" for value in values)
        print(f"{label:<{first_col_width}} {value_text}")


def _get_component_plot_data(
    component: RuntimeComponentView,
    scalar_field_name: str,
    u_field_name: str,
    v_field_name: str,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Return lon/lat grids and scalar/vector fields for one component."""

    grid = component.grid
    lon = runtime_array_to_host(grid.longitude)
    lat = runtime_array_to_host(grid.latitude)
    lon_2d, lat_2d = np.meshgrid(lon, lat, indexing="ij")
    scalar_field = runtime_array_to_host(
        jnp.asarray(_component_plot_field(component, scalar_field_name)).T
    )
    u_field = runtime_array_to_host(
        jnp.asarray(_component_plot_field(component, u_field_name)).T
    )
    v_field = runtime_array_to_host(
        jnp.asarray(_component_plot_field(component, v_field_name)).T
    )
    return lon_2d, lat_2d, scalar_field, u_field, v_field


def plot_component_scalar_vector_comparison(
    rows: Sequence[tuple[str, RuntimeComponentView, str, str, str]],
    *,
    figsize: tuple[float, float] = (15.0, 10.0),
    quiver_scale: float = 100.0,
    cmap: str = "coolwarm",
) -> tuple[Any, NDArray[Any], Any]:
    """Create aligned scalar/vector plots for multiple components."""

    import matplotlib.pyplot as plt

    if not rows:
        raise ValueError("rows must contain at least one component")

    n_rows = len(rows)
    fig, axs = plt.subplots(n_rows, 2, figsize=figsize, layout="constrained")

    if n_rows == 1:
        axs = np.asarray([axs])
    else:
        axs = np.asarray(axs)

    plot_data = [
        (label, *_get_component_plot_data(component, scalar_name, u_name, v_name))
        for label, component, scalar_name, u_name, v_name in rows
    ]

    scalar_min = float(min(np.nanmin(item[3]) for item in plot_data))
    scalar_max = float(max(np.nanmax(item[3]) for item in plot_data))

    lon_min = float(min(np.nanmin(item[1]) for item in plot_data))
    lon_max = float(max(np.nanmax(item[1]) for item in plot_data))
    lat_min = float(min(np.nanmin(item[2]) for item in plot_data))
    lat_max = float(max(np.nanmax(item[2]) for item in plot_data))

    scalar_mappable = None
    for i, (label, lon_2d, lat_2d, scalar_field, u_field, v_field) in enumerate(
        plot_data
    ):
        scalar_plot = axs[i, 0].pcolormesh(
            lon_2d,
            lat_2d,
            scalar_field,
            shading="auto",
            cmap=cmap,
            vmin=scalar_min,
            vmax=scalar_max,
        )
        if scalar_mappable is None:
            scalar_mappable = scalar_plot

        axs[i, 0].set_title(f"{label} Scalar Field")
        axs[i, 0].set_xlabel("Longitude")
        axs[i, 0].set_ylabel("Latitude")

        axs[i, 1].quiver(
            lon_2d,
            lat_2d,
            u_field,
            v_field,
            scale=quiver_scale,
        )
        axs[i, 1].set_title(f"{label} Vector Field")
        axs[i, 1].set_xlabel("Longitude")
        axs[i, 1].set_ylabel("Latitude")

    for ax in axs.flat:
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)

    if scalar_mappable is None:
        raise ValueError("No scalar field was plotted")

    return fig, axs, scalar_mappable
