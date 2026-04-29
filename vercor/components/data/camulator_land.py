from datetime import timedelta
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from vercor.components.external.camulator_state import (
    load_camulator_forcing_context,
    parse_datetime_from_config,
)

from vercor.grid import RectilinearGrid
from vercor.components.base import (
    ComponentInitContext,
    HostRuntimeComponent,
    RuntimeStepContext,
)
from vercor.grid_masks import create_lnd_mask_from_ocn

if TYPE_CHECKING:
    from vercor.runtime import RuntimeComponentState


def _prepare_camulator_land_surface_temperature(
    land_surface_temperature: ArrayLike,
) -> jax.Array:
    """Normalize CAMulator land temperature fields for JAX-backed runtime storage."""
    return jnp.asarray(land_surface_temperature)


class CAMulatorLand(HostRuntimeComponent):
    def __init__(
        self,
        config_path: str,
        camulator_grid: RectilinearGrid,
        ocn_grid: RectilinearGrid,
        name: str = "LND",
    ) -> None:
        """
        Read all necessary fields from the provided forcing files.

        Arguments:
            name (str): component name
            config_path (str): path to CAMulator configuration file (e.g., camulator_config.yml)
            camulator_grid (RectilinearGrid): CAMulator grid object
            ocn_grid (RectilinearGrid): Ocean component grid object

        Attributes of parent classes to be initialized:
            Component
                name: str
                grid: RectilinearGrid
        """

        self.config_path = config_path

        longitude = camulator_grid.longitude
        latitude = camulator_grid.latitude
        lnd_bmask, _ = create_lnd_mask_from_ocn(
            atm_lat=latitude,
            atm_lon=longitude,
            ocn_grid=ocn_grid,
        )

        context = load_camulator_forcing_context(config_path=self.config_path)

        self.conf = context["conf"]
        self.forcing_ds = context["forcing_dataset_raw"]
        self.lead_time_periods = self.conf["data"]["lead_time_periods"]

        grid = RectilinearGrid(
            name=f"{name.lower()}-grid",
            longitude=longitude,
            latitude=latitude,
            binary_mask=lnd_bmask,
        )

        super().__init__(name, grid=grid)

    def initialize(self, context: ComponentInitContext) -> None:
        logger = context.logger
        self.coupler_start_datetime = context.start
        self.coupling_timestep = timedelta(seconds=context.dt_seconds)

        self.model_timestep = timedelta(hours=self.lead_time_periods)
        self.model_substeps = int(
            self.coupling_timestep.total_seconds()
            // self.model_timestep.total_seconds()
        )

        if self.coupling_timestep % self.model_timestep != timedelta(days=0):
            raise ValueError(
                f"model_timestep ({self.model_timestep}) must be a "
                f"multiple of coupling_timestep ({self.coupling_timestep})"
            )

        self.dynamic_ds = self.forcing_ds[
            [
                "TS",
            ]
        ]

        # IMPORTANT: Use the config's datetime object directly for xarray lookup
        # It might be cftime.DatetimeNoLeap, which xarray expects
        start_datetime_raw = self.conf["predict"]["start_datetime"]
        loc = self.dynamic_ds.indexes["time"].get_loc(start_datetime_raw)
        self.start_ix = loc.start if isinstance(loc, slice) else loc
        logger.info(f"Starting integration at time index: {self.start_ix}")

        # Now convert to Python datetime for output formatting (if it's a string or cftime)
        init_dt = parse_datetime_from_config(self.conf)
        self.init_str = init_dt.strftime("%Y-%m-%dT%HZ")

        if self.coupler_start_datetime != init_dt:
            logger.warning(
                f"Coupler start datetime ({self.coupler_start_datetime}) does not match "
                f"CAMulator forcing start datetime ({start_datetime_raw}). "
                f"Using CAMulator start datetime for indexing."
            )

        self.timestep_counter = 0

        # Units: [K]
        self.data["land_surface_temperature"] = jnp.full(
            self.grid.shape, 283.0, dtype=jnp.float32
        )

    def step_host_runtime_state(
        self,
        component_state: "RuntimeComponentState",
        context: RuntimeStepContext,
    ) -> "RuntimeComponentState":
        """Advance the private host-backed CAMulator land forcing boundary."""

        time = context.time
        if time is None:
            return component_state

        data = component_state.data

        idx = self.start_ix + self.timestep_counter * self.model_substeps
        ts = self.dynamic_ds.isel(time=idx).load()

        data = data.set(
            "land_surface_temperature",
            _prepare_camulator_land_surface_temperature(ts["TS"].values),
        )

        self.timestep_counter += 1

        return component_state.with_data(data)
