from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING
import numpy as np
from vercor.components.base import Component, ForcingData, write_shared_to_netcdf
from vercor.components.base import TimedNamedArray as TNA
from vercor.grid import RectilinearGrid
from vercor.tools import get_field_at_specific_time, get_forcing_data


if TYPE_CHECKING:
    from vercor.coupler import Coupler


class ERA5Ocean(Component, ForcingData):
    def __init__(
        self,
        name: str = "ERA5-OCN",
        surface_file: Path = get_forcing_data("surface"),
    ) -> None:
        """
        Read all necessary fields from the provided forcing files.

        Arguments:
            name (str): component name
            surface_file (Path): path to netCDF file with data at surface level

        Attributes of parent classes to be initialized:
            ForcingData
                DATA_FILES: dict [str, str]
            Component
                name: str
                grid: RectilinearGrid
                shared_fields: Dict[str, NDArray] = field(default_factory=dict)
        """

        self.DATA_FILES = {
            "surface": str(surface_file),
        }

        self.fields2share = ("sst",)

        self._state = {}

        longitude = self._read_forcing("longitude", where="surface")
        latitude = self._read_forcing("latitude", where="surface")[::-1]
        fraction_mask = self._read_forcing("lsm", where="surface", flip_y=True).T[0, ::]
        fraction_mask = 1 - fraction_mask
        binary_mask = np.where(fraction_mask > 0.0, 1.0, 0.0)

        self.grid = RectilinearGrid(
            name=f"{name.lower()}-grid",
            longitude=longitude,
            latitude=latitude,
            binary_mask=binary_mask,
            fraction_mask=fraction_mask,
        )

        super().__init__(name, grid=self.grid)

        self._state["sst"] = self._read_forcing("sst", where="surface", flip_y=True)

    def initialize(self, coupler: "Coupler") -> None:

        for field in self.fields2share:
            setattr(
                self.outgoing_fields,
                field,
                TNA(
                    get_field_at_specific_time(field, self._state, coupler),
                    coupler.clock.start,
                    self.name,
                ),
            )

    def step(self, dt: timedelta, time: datetime, coupler: "Coupler") -> None:
        """Advance to the next time step in the dataset
        using time interpolation from one month to another.
        """

        for field in self.fields2share:
            setattr(
                self.outgoing_fields,
                field,
                TNA(
                    get_field_at_specific_time(field, self._state, coupler),
                    time,
                    self.name,
                ),
            )
