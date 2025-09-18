import numpy as np
from verec.components.base import Component
from verec.fields import Field


class Atmosphere(Component):
    """Toy atmosphere: produces surface fluxes and 2m temperature from SST.
    Inputs: SST [K]
    Outputs: SHF [W/m2], LHF [W/m2], TA2M [K]
    """

    def __init__(self, name, grid) -> None:
        # super().__init__(name, grid, inputs=["SST"], outputs=["SHF", "LHF", "TA2M", "u10m", "v10m"])
        super().__init__(name, grid)

    def initialize(self, coupler) -> None:
        ny, nx = self.grid.shape
        self.state["TA2M"] = Field(
            "TA2M", 273.15 + 15.0 * np.ones((ny, nx)), self.grid, units="K"
        )
        self.state["SHF"] = Field("SHF", np.zeros((ny, nx)), self.grid, units="W m-2")
        self.state["LHF"] = Field("LHF", np.zeros((ny, nx)), self.grid, units="W m-2")
        self.state["u10m"] = Field("u10m", np.zeros((ny, nx)), self.grid, units="m s-1")
        self.state["v10m"] = Field("v10m", np.zeros((ny, nx)), self.grid, units="m s-1")

    def step(self, dt, time, coupler) -> None:
        # Bulk formula toy: flux proportional to (TA2M - SST)
        SST = self.state.get("SST")
        if SST is None:
            ny, nx = self.grid.shape
            SST = Field("SST", 273.15 + 15.0 * np.ones((ny, nx)), self.grid, units="K")

        TA = self.state["TA2M"].data
        dT = TA - SST.data
        C = 10.0  # W m-2 K-1, toy exchange coefficient
        SHF = -C * dT  # ocean heat gain positive when SST < TA
        LHF = -0.5 * SHF

        # Update wind (toy)
        lat = np.array(self.grid.latitude)
        lon = np.array(self.grid.longitude) - 180.0
        latitudes, longitudes = np.meshgrid(lat, lon, indexing="ij")
        u10m = np.cos(np.deg2rad(latitudes))  # zonal flow varying with latitude
        v10m = 0.5 * np.sin(np.deg2rad(longitudes))  # small meridional perturbation

        self.state["SHF"] = Field("SHF", SHF, self.grid, units="W m-2")
        self.state["LHF"] = Field("LHF", LHF, self.grid, units="W m-2")

        self.state["u10m"] = Field("u10m", u10m, self.grid, units="m s-1")
        self.state["v10m"] = Field("v10m", v10m, self.grid, units="m s-1")

        # Relax TA2M toward SST weakly (toy boundary layer)
        self.state["TA2M"].data = TA - 0.01 * dT
