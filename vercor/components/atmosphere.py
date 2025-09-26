from typing import Dict
import numpy as np
from vercor.components.base import Component
from vercor.components.forcing import ERA5Forcing
from vercor.grid import RectilinearGrid
from vercor.fluxes.utilities import get_press_levs, compute_z_level, potential_temperature, air_density


class Atmosphere(Component):
    """Toy atmosphere: produces surface fluxes and 2m temperature from SST.
    Inputs: SST [K]
    Outputs: SHF [W/m2], LHF [W/m2], TA2M [K]
    """

    def __init__(self, name: str, grid: RectilinearGrid) -> None:
        super().__init__(name, grid)

    def initialize(self, coupler) -> None:
        ny, nx = self.grid.shape
        self.state["TA2M"] = 273.15 + 15.0 * np.ones((ny, nx))
        self.state["SHF"] = np.zeros((ny, nx))
        self.state["LHF"] = np.zeros((ny, nx))
        self.state["u10m"] = np.zeros((ny, nx))
        self.state["v10m"] = np.zeros((ny, nx))

    def step(self, dt, time, coupler) -> None:
        # Bulk formula toy: flux proportional to (TA2M - SST)
        SST = self.state.get("SST")
        if SST is None:
            ny, nx = self.grid.shape
            SST = 273.15 + 15.0 * np.ones((ny, nx))

        TA = self.state["TA2M"]
        dT = TA - SST
        C = 10.0  # W m-2 K-1, toy exchange coefficient
        SHF = -C * dT  # ocean heat gain positive when SST < TA
        LHF = -0.5 * SHF

        # Update wind (toy)
        lat = np.array(self.grid.latitude)
        lon = np.array(self.grid.longitude) - 180.0
        latitudes, longitudes = np.meshgrid(lat, lon, indexing="ij")
        u10m = np.cos(np.deg2rad(latitudes))  # zonal flow varying with latitude
        v10m = 0.5 * np.sin(np.deg2rad(longitudes))  # small meridional perturbation

        self.state["SHF"] = SHF
        self.state["LHF"] = LHF

        self.state["u10m"] = u10m
        self.state["v10m"] = v10m

        # Relax TA2M toward SST weakly (toy boundary layer)
        self.state["TA2M"] = TA - 0.01 * dT


class DataAtmosphere(Component):
    """Data atmosphere: reads and iterates atmospheric data from provided dataset."""

    def __init__(self, name: str, dataset: ERA5Forcing) -> None:
        super().__init__(name, dataset.grid)
        self.dataset = dataset

    def initialize(self, coupler) -> None:
        ny, nx = self.grid.shape
        settings = coupler.settings
        ds = self.dataset

        self.state["u10m"] = np.zeros((ny, nx))
        self.state["v10m"] = np.zeros((ny, nx))

        for m in range(12):
            ph = get_press_levs(ds.spres[..., m], ds.hyai, ds.hybi)
            pf = get_press_levs(ds.spres[..., m], ds.hyam, ds.hybm)

            zbot = compute_z_level(settings, ds.temperature[..., m], ds.specific_humidity[..., m], ph[:, :]) # L136
            rbot = air_density(settings, ds.tbot[:, :, m], pf[:, :, 0])
            thbot = potential_temperature(settings, ds.tbot[:, :, m], pf[:, :, 0])

    def step(self, dt, time, coupler) -> None:
        """Advance to the next time step in the dataset
        using time interpolation from one month to another.
        """
        pass
