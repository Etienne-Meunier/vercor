"""Helpers for the VerCOR + Veros-AD gradient demonstration notebook.

Builds the same ATM (ERA5) + OCN (Veros) + LND (ERA5) coupled system as
``examples/run_verosad_global4deg_grad.py`` and adds small utilities for
capturing a rollout's field evolution, reading/writing Veros ocean
variables, and plotting. Keeps the notebook itself focused on the jax calls.

Field evolution (section 1) is captured through vercor's built-in
period-output pipeline rather than by chaining several `Coupler.run()` calls:
building a fresh `Coupler` re-runs the Veros setup routine and produces a
structurally distinct payload (fresh ``settings``/``var_meta`` objects), so a
`RunState` carried forward from one `Coupler` fails schema validation on
another. One `Coupler`/`Clock` per notebook section keeps this simple.
"""

from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, "veros"))  # veros-ad submodule instead of pip veros
sys.path.insert(0, _REPO_ROOT)  # repo root, for `import vercor`

import numpy as np

from vercor import Clock, Coupler, Exchange, RuntimeOptions
from vercor.setups import VerosConfig, make_era5_atmosphere, make_era5_land, make_veros_gcm
from vercor.setups._external.veros_state import copy_state, set_veros_variable
from vercor.recipes import (
    ATMOSPHERE_TO_LAND_BASIC_FIELDS,
    ATMOSPHERE_TO_VEROS_FORCING_FIELDS,
    LAND_TO_ATMOSPHERE_SURFACE_FIELDS,
    OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS,
)
from vercor.output import OutputSpec, OutputTarget, PeriodOutput
from vercor.regridding import bilinear
from vercor.topology import SurfaceMaskPolicy

RUN_ORDER = ["OCN", "LND", "ATM"]

EXCHANGES = (
    Exchange(
        source="ATM",
        target="OCN",
        fields=ATMOSPHERE_TO_VEROS_FORCING_FIELDS,
        regridder_factory=bilinear,
    ),
    Exchange(
        source="OCN",
        target="ATM",
        fields=OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS,
        regridder_factory=bilinear,
    ),
    Exchange(
        source="ATM",
        target="LND",
        fields=ATMOSPHERE_TO_LAND_BASIC_FIELDS,
        regridder_factory=bilinear,
    ),
    Exchange(
        source="LND",
        target="ATM",
        fields=LAND_TO_ATMOSPHERE_SURFACE_FIELDS,
        regridder_factory=bilinear,
    ),
)


def build_coupler(
    start: datetime,
    steps: int,
    dt_seconds: float = 86400.0,
    *,
    ocean_output_variables: tuple[str, ...] | None = None,
) -> Coupler:
    """Return a fresh ATM+OCN(Veros)+LND coupler for `steps` days from `start`.

    `ocean_output_variables`, if given, makes the OCN component write one
    NetCDF snapshot per day (see `rollout_and_capture`).
    """

    atm = make_era5_atmosphere()
    output = (
        OutputSpec(period=PeriodOutput(frequency="step", variables=ocean_output_variables))
        if ocean_output_variables
        else OutputSpec()
    )
    ocn = make_veros_gcm(
        config=VerosConfig(
            setup="global_4deg_learning",
            uses_atmosphere_forcing=True,
            restore_to_climatology=True,
            output=output,
        ),
    )
    lnd = make_era5_land()
    clock = Clock(start=start, dt_seconds=dt_seconds, steps=steps, calendar="noleap")
    return Coupler(
        clock=clock,
        components=[ocn, lnd, atm],
        exchanges=EXCHANGES,
        run_order=RUN_ORDER,
        runtime=RuntimeOptions(topology=SurfaceMaskPolicy()),
    )


def ocean_payload(state):
    """Return the raw Veros ``VerosState`` behind the "OCN" component."""

    return state._component_state("OCN").payload


def with_ocean_payload(state, payload):
    """Return `state` with the "OCN" component's Veros payload replaced."""

    component_state = state._component_state("OCN")
    return state._with_component_state("OCN", component_state.with_payload(payload))


def set_ocean_variable(state, name: str, value):
    """Return `state` with one Veros ``variables.<name>`` replaced by `value`."""

    return with_ocean_payload(state, set_veros_variable(ocean_payload(state), name, value))


def disable_eke(state):
    """Return `state` with ``enable_eke`` off, so ``K_gm_0`` drives ``K_gm`` directly.

    Veros only reads the fixed ``K_gm_0`` diffusivity when its EKE
    parameterization is switched off (otherwise ``K_gm`` is diagnosed from
    the EKE budget and ``K_gm_0`` has no effect).
    """

    payload = copy_state(ocean_payload(state), jitted=True)
    with payload.settings.unlock():
        payload.settings.enable_eke = False
    return with_ocean_payload(state, payload)


def rollout_and_capture(start, total_steps, output_dir=None, *, variables=("temp", "salt"), dt_seconds=86400.0):
    """Integrate one coupled rollout, writing a per-day NetCDF snapshot of `variables`.

    Returns ``(initial_state, output_dir)``. `output_dir` defaults to a fresh
    temp directory; read the snapshots back with `load_daily_field`.
    """

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="verosad_grad_demo_")

    cpl = build_coupler(start, total_steps, dt_seconds, ocean_output_variables=variables)
    initial_state = cpl.initial_state()
    output = OutputTarget(output_dir, write_period=True, write_final_fields=False, write_snapshots=False)
    cpl.run(initial_state, output=output)
    return initial_state, output_dir


def load_daily_field(output_dir, variable: str, level: int = -1):
    """Return ``[(day, 2D array), ...]`` for one OCN NetCDF variable, day 1, 2, ...

    Arrays come back in ``(yt, xt)`` order -- already what `matplotlib`'s
    `pcolormesh(lon, lat, data)` expects and, unlike the raw Veros payload,
    with no ghost cells or leftover leap-frog time axis to trim.
    """

    import h5netcdf

    paths = sorted(Path(output_dir).glob("ocn.averages.*.nc"))
    days = []
    for day, path in enumerate(paths, start=1):
        with h5netcdf.File(path, "r") as f:
            data = np.asarray(f.variables[variable][0, level, :, :])
        days.append((day, data))
    return days


def ocean_grid(payload):
    """Return trimmed (lon, lat) 1D T-grid coordinates and the land mask.

    Veros carries a 2-cell ghost/halo ring on each side (for cyclic BCs);
    that ring is trimmed here so shapes match the physical domain.
    """

    sl = slice(2, -2)
    lon = np.asarray(payload.variables.xt[sl])
    lat = np.asarray(payload.variables.yt[sl])
    mask = np.asarray(payload.variables.maskT[sl, sl, :], dtype=bool)
    return lon, lat, mask


def surface_field(payload, name: str, level: int = -1):
    """Return one trimmed, land-masked Veros T-grid field, ``(xt, yt)`` order."""

    sl = slice(2, -2)
    tau = int(payload.variables.tau)
    field = np.asarray(getattr(payload.variables, name)[sl, sl, level, tau])
    mask = np.asarray(payload.variables.maskT[sl, sl, level], dtype=bool)
    return np.where(mask, field, np.nan)


def plot_field_evolution(initial_state, output_dir, field_name, *, level=-1, every=1, cmap="coolwarm", label=None):
    """Plot one Veros T-grid field across a captured rollout, land masked.

    Combines the in-memory initial state (day 0) with NetCDF snapshots
    written by `rollout_and_capture` (day 1, 2, ...), keeping every `every`-th
    snapshot.
    """

    import matplotlib.pyplot as plt

    payload = ocean_payload(initial_state)
    lon, lat, mask = ocean_grid(payload)
    mask_level = mask[:, :, level].T  # -> (yt, xt), matching the NetCDF snapshots

    panels = [(0, surface_field(payload, field_name, level=level).T)]
    for day, data in load_daily_field(output_dir, field_name, level=level)[::every]:
        panels.append((day, np.where(mask_level, data, np.nan)))

    vmin = min(np.nanmin(data) for _, data in panels)
    vmax = max(np.nanmax(data) for _, data in panels)

    fig, axs = plt.subplots(
        1, len(panels), figsize=(4.5 * len(panels), 4),
        constrained_layout=True, sharex=True, sharey=True,
    )
    axs = np.atleast_1d(axs)
    im = None
    for ax, (day, data) in zip(axs, panels):
        im = ax.pcolormesh(lon, lat, data, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
        ax.set_facecolor("0.8")  # land (nan) shows through as gray
        ax.set_title("initial" if day == 0 else f"day {day}")
        ax.set_xlabel("longitude (deg)")
    axs[0].set_ylabel("latitude (deg)")
    fig.colorbar(im, ax=list(axs), shrink=0.85, label=label or field_name)
    fig.suptitle(f"{label or field_name} over integration")
    return fig, axs


def plot_gradient_map(payload, grad_field, *, level=-1, title=""):
    """Plot a land-masked spatial gradient field at one vertical level.

    Uses a diverging colormap centered on zero, since gradients are signed.
    """

    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    sl = slice(2, -2)
    lon, lat, mask = ocean_grid(payload)
    tau = int(payload.variables.tau)
    data = np.where(mask[:, :, level], np.asarray(grad_field[sl, sl, level, tau]), np.nan)

    vmax = np.nanmax(np.abs(data))
    vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1e-12

    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    im = ax.pcolormesh(
        lon, lat, data.T, cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax), shading="auto",
    )
    ax.set_facecolor("0.8")
    ax.set_xlabel("longitude (deg)")
    ax.set_ylabel("latitude (deg)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.85)
    return fig, ax


__all__ = [
    "build_coupler",
    "disable_eke",
    "load_daily_field",
    "ocean_grid",
    "ocean_payload",
    "plot_field_evolution",
    "plot_gradient_map",
    "rollout_and_capture",
    "set_ocean_variable",
    "surface_field",
    "with_ocean_payload",
]
