from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


if TYPE_CHECKING:
    from vercor.coupler import Coupler


def get_periodic_interval(current_time, cycle_length, rec_spacing, n_rec):
    """
    Ported from Veros: https://github.com/team-ocean/veros/blob/main/veros/tools/setup.py#L88

    Used for linear interpolation between periodic time intervals.

    One common application is the interpolation of external forcings that are defined
    at discrete times (e.g. one value per month of a standard year) to the current
    time step.

    Arguments:
       current_time (float): Time to interpolate to.
       cycle_length (float): Total length of one periodic cycle.
       rec_spacing (float): Time spacing between each data record.
       n_rec (int): Total number of records available.

    Returns:
       :obj:`tuple` containing (n1, f1), (n2, f2): Indices and weights for the interpolated
       record array.

    Example:
       The following interpolates a record array ``data`` containing 12 monthly values
       to the current time step:

       >>> year_in_seconds = 60. * 60. * 24. * 365.
       >>> current_time = 60. * 60. * 24. * 45. # mid-february
       >>> print(data.shape)
       (360, 180, 12)
       >>> (n1, f1), (n2, f2) = get_periodic_interval(current_time, year_in_seconds, year_in_seconds / 12, 12)
       >>> data_at_current_time = f1 * data[..., n1] + f2 * data[..., n2]

    """
    current_time = current_time % cycle_length
    # using npx.array works with both NumPy and JAX
    t_idx_1 = np.array(current_time // rec_spacing, dtype="int")
    t_idx_2 = np.array((1 + t_idx_1) % n_rec, dtype="int")
    weight_2 = (current_time - rec_spacing * t_idx_1) / rec_spacing
    weight_1 = 1.0 - weight_2
    return (t_idx_1, weight_1), (t_idx_2, weight_2)


def datetime_to_seconds_in_year(dt: datetime) -> float:
    year_start = datetime(dt.year, 1, 1)
    seconds_since_year_start = (dt - year_start).total_seconds()
    return seconds_since_year_start


def get_forcing_data(file_type: str) -> Path:
    """Return the absolute Paths to the ./forcing directory relative to this file."""

    output = {
        "model_level": (
            Path(__file__).parent
            / ".."
            / "forcing"
            / "era5_198x_ml_4x4deg_monthly_mean.nc"
        ).resolve(),
        "surface": (
            Path(__file__).parent
            / ".."
            / "forcing"
            / "era5_198x_sfc_4x4deg_monthly_mean.nc"
        ).resolve(),
    }

    return output[file_type]


def get_field_at_specific_time(
    field_name: str,
    state: Dict,
    coupler: "Coupler",
    current_time: Optional[datetime] = None,
) -> NDArray:

    total_seconds = datetime_to_seconds_in_year(
        coupler.clock.start if current_time is None else current_time
    )

    (n1, f1), (n2, f2) = get_periodic_interval(
        current_time=total_seconds,
        cycle_length=coupler.settings.year_in_seconds,
        rec_spacing=coupler.settings.year_in_seconds / 12.0,
        n_rec=12,
    )

    # Use transpose to have (lat, lon) ordering
    out: NDArray = (
        f1 * state[f"{field_name}"][..., n1].T + f2 * state[f"{field_name}"][..., n2].T
    )

    return out
