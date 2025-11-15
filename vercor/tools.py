import numpy as np
from datetime import datetime


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
