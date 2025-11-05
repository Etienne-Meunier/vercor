from dataclasses import dataclass, field
from typing import Any
from numpy.typing import NDArray
import numpy as np
import h5netcdf
from vercor.grid import RectilinearGrid


@dataclass
class ForcingData:
    DATA_FILES: dict[str, str] = field(default_factory=dict)

    def _read_forcing(self, var: str, forcing: str, flip_y: bool = False):
        with h5netcdf.File(self.DATA_FILES[forcing], "r") as infile:
            var_obj = np.array(infile.variables[var]).T
            if flip_y:
                return np.flip(var_obj, axis=1)
            else:
                return var_obj


@dataclass
class ERA5Forcing(ForcingData):
    """
    model_level_file (str): path to netCDF file with data at model levels
    surface_file (str): path to netCDF file with data at surface level
    """

    model_level_file: str = field(init=False)
    surface_file: str = field(init=False)

    hyai: NDArray[np.floating[Any]] = field(init=False)
    hybi: NDArray[np.floating[Any]] = field(init=False)
    hyam: NDArray[np.floating[Any]] = field(init=False)
    hybm: NDArray[np.floating[Any]] = field(init=False)
    spres: NDArray[np.floating[Any]] = field(init=False)
    specific_humidity: NDArray[np.floating[Any]] = field(init=False)
    temperature: NDArray[np.floating[Any]] = field(init=False)
    qbot: NDArray[np.floating[Any]] = field(init=False)
    tbot: NDArray[np.floating[Any]] = field(init=False)
    ubot: NDArray[np.floating[Any]] = field(init=False)
    vbot: NDArray[np.floating[Any]] = field(init=False)
    swr_net: NDArray[np.floating[Any]] = field(init=False)
    lwr_dw: NDArray[np.floating[Any]] = field(init=False)

    def __post_init__(self):
        if not hasattr(self, "model_level_file") or not hasattr(self, "surface_file"):
            raise ValueError(
                "Both 'model_level_file' and 'surface_file' must be provided."
            )
        self.DATA_FILES = {
            "ml": self.model_level_file,
            "sfc": self.surface_file,
        }

    def read(self) -> None:
        """Read all necessary fields from the provided forcing files.
        Logic:
        ------
        - only the lowest to the ground model levels are read (L136, L137)
        """
        longitude = self._read_forcing("longitude", forcing="ml")
        latitude = self._read_forcing("latitude", forcing="ml")[::-1]

        self.grid = RectilinearGrid(
            name="atm-grid",
            longitude=longitude,
            latitude=latitude,
        )

        self.hyai = self._read_forcing("hyai", forcing="ml")[-3:]  # L135-L137
        self.hybi = self._read_forcing("hybi", forcing="ml")[-3:]  # L135-L137
        self.hyam = self._read_forcing("hyam", forcing="ml")[-2:]  # L136-L137
        self.hybm = self._read_forcing("hybm", forcing="ml")[-2:]  # L136-L137

        lnsp = self._read_forcing("lnsp", forcing="ml", flip_y=True)[..., 0, :]
        self.spres = np.exp(lnsp)
        self.specific_humidity = self._read_forcing("q", forcing="ml", flip_y=True)[
            ..., 1:, :
        ]  # L136-L137
        self.temperature = self._read_forcing("t", forcing="ml", flip_y=True)[
            ..., 1:, :
        ]  # L136-L137

        self.ubot = self._read_forcing("u", forcing="ml", flip_y=True)[
            :, :, 1, :
        ]  # L136
        self.vbot = self._read_forcing("v", forcing="ml", flip_y=True)[
            :, :, 1, :
        ]  # L136

        # tcc = self._read_forcing("tcc", forcing="sfc", flip_y=True)
        self.swr_net = self._read_forcing("msnswrf", forcing="sfc", flip_y=True)
        self.lwr_dw = self._read_forcing("msdwlwrf", forcing="sfc", flip_y=True)

        self.qbot = self.specific_humidity[..., 0, :]  # L136
        self.tbot = self.temperature[..., 0, :]  # L136
