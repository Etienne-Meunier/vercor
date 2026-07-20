"""CAMulator atmosphere setup-state ownership and lifecycle callbacks."""

from __future__ import annotations

from datetime import timedelta
from typing import Optional

import torch
import xarray as xr

from vercor.components import (
    Component,
    SetupContext,
    SetupResult,
)
from vercor.dtypes import jax_ones
from vercor.grids import RectilinearGrid
from vercor.jax_logging import LoggerLike, get_default_logger
from vercor.setups._time_helpers import (
    assign_model_timestep_alignment,
    grid_field_defaults,
)
import vercor.setups._external.camulator_contracts as _camulator_contracts
from vercor.setups._external.camulator_forcing import CamulatorRuntimeCursor
import vercor.setups._external.camulator_init as _camulator_init
import vercor.setups._external.camulator_tensors as _camulator_tensors


class CAMulatorGCMSetupState:
    """Mutable setup-time owner for a host-backed CAMulator atmosphere adapter."""

    coupling_timestep: timedelta
    model_timestep: timedelta
    model_substeps: int

    def __init__(
        self,
        config_path: str,
        name: str = "ATM",
        model_weights_path: str = "checkpoint.pt00091.pt",
        init_noise: Optional[float] = None,
        device: str = "cuda",
        logger: LoggerLike | None = None,
    ) -> None:
        """Build CAMulator model resources and the VerCOR atmosphere grid."""

        self.logger = logger if logger is not None else get_default_logger()
        self.config_path = config_path
        self.model_weights_path = model_weights_path
        self.device = device
        self.init_noise = init_noise
        self.runtime_cursor = CamulatorRuntimeCursor()
        self._output_prediction: torch.Tensor | None = None
        self._output_prediction_samples: torch.Tensor | None = None

        context = _camulator_init.initialize_camulator(
            config_path=self.config_path,
            model_name=self.model_weights_path,
            device=self.device,
            logger=self.logger,
        )

        self.conf = context["conf"]
        self.stepper = context["stepper"]
        self.forcing_ds_norm = context["forcing_dataset"]
        self.static_forcing = context["static_forcing"]
        self.state = context["initial_state"]
        self.latlons = context["latlons"]
        self.metadata = context["metadata"]
        self.device = context["device"]
        self.state_transformer = context["state_transformer"]

        self.df_vars = self.conf["data"]["dynamic_forcing_variables"]
        self.lead_time_periods = self.conf["data"]["lead_time_periods"]

        self.grid = RectilinearGrid(
            name=name,
            longitude=self.latlons.longitude.values,
            latitude=self.latlons.latitude.values,
            binary_mask=jax_ones(
                (
                    self.latlons.latitude.values.shape[0],
                    self.latlons.longitude.values.shape[0],
                )
            ),
        )

    def setup(
        self,
        component: Component,
        context: SetupContext,
    ) -> SetupResult:
        """Align timestep, initialize runtime forcing, and seed output fields."""

        logger = context.logger
        self.coupler_start_datetime = context.start
        assign_model_timestep_alignment(
            self,
            context.dt_seconds,
            timedelta(hours=self.lead_time_periods),
        )

        if self.init_noise is not None:
            self.state = _camulator_init.add_init_noise(
                self.state,
                noise_std=self.init_noise,
                logger=logger,
            )

        logger.info("Tracing model with torch.jit...")
        dummy_input = torch.zeros_like(self.state)
        traced_model = torch.jit.trace(self.stepper.model, dummy_input)
        self.stepper.model = traced_model
        logger.info(f"Model traced with input shape: {dummy_input.shape}")

        ds_physics = xr.open_dataset(self.conf["data"]["save_loc_physics"])

        self.P0 = 100000.0
        self.hyai = torch.tensor(ds_physics["hyai"].values / self.P0).to(self.device)[
            None, :, None, None
        ]
        self.hyam = torch.tensor(ds_physics["hyam"].values).to(self.device)[
            None, :, None, None
        ]
        self.hybi = torch.tensor(ds_physics["hybi"].values).to(self.device)[
            None, :, None, None
        ]
        self.hybm = torch.tensor(ds_physics["hybm"].values).to(self.device)[
            None, :, None, None
        ]
        self.LANDM_COSLAT = ds_physics["LANDM_COSLAT"].values

        self.dynamic_ds = self.forcing_ds_norm[self.df_vars]

        self.runtime_cursor.initialize(
            conf=self.conf,
            dynamic_ds=self.dynamic_ds,
            coupler_start_datetime=self.coupler_start_datetime,
            model_substeps=self.model_substeps,
            logger=logger,
        )

        self.accessor_input = _camulator_tensors.StateVariableAccessor(
            self.conf, tensor_type="input"
        )
        self.accessor_output = _camulator_tensors.StateVariableAccessor(
            self.conf, tensor_type="output"
        )
        self._output_prediction = None
        self._output_prediction_samples = None

        self.forecast_hour = 1
        _ = component
        return SetupResult(
            fields=grid_field_defaults(
                _camulator_contracts.CAMULATOR_RUNTIME_FIELD_NAMES,
            )
        )


__all__ = [
    "CAMulatorGCMSetupState",
]
