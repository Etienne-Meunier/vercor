import numpy as np


def qsat(tk: np.ndarray) -> np.ndarray:
    """The saturation humidity of air (kg/m^3)

    Argument:
        tk (:obj:`ndarray`): temperature (K)
    """
    return 640380.0 / np.exp(5107.4 / tk)


def qsat_august_eqn(ps: np.ndarray, tk: np.ndarray) -> np.ndarray:
    """Saturated specific humidity (kg/kg)

    Arguments:
        ps (:obj:`ndarray`): atm sfc pressure (Pa)
        tk (:obj:`ndarray`): atm temperature (K)

    Returns:
        :obj:`ndarray`

    Reference:
        Barnier B., L. Siefridt, P. Marchesiello, (1995):
        Thermal forcing for a global ocean circulation model
        using a three-year climatology of ECMWF analyses,
        Journal of Marine Systems, 6, p. 363-380.
    """
    return 0.622 / ps * 10 ** (9.4051 - 2353.0 / tk) * 133.322


def get_press_levs(sp: np.ndarray, hya: np.ndarray, hyb: np.ndarray) -> np.ndarray:
    """Compute pressure levels

    Arguments:
        sp (:obj:`ndarray`): Atmospheric surface pressure
        hya (:obj:`ndarray`): Hybrid sigma level A coefficient for vertical grid
        hyb (:obj:`ndarray`): Hybrid sigma level B coefficient for vertical grid

    Return:
        :obj:`ndarray`
    """

    return (
        hya[np.newaxis, np.newaxis, :]
        + hyb[np.newaxis, np.newaxis, :] * sp[:, :, np.newaxis]
    )


def compute_z_level(
    settings, t: np.ndarray, q: np.ndarray, ph: np.ndarray
) -> np.ndarray:
    """Computes the altitudes at ECMWF Integrated Forecasting System
    (ECMWF-IFS) model half- and full-levels (for 137 levels model reanalysis: L137)

    Arguments:
        t (:obj:`ndarray`): Atmospheric temperture [K]
        q (:obj:`ndarray`): Atmospheric specific humidity [kg/kg]
        ph (:obj:`ndarray`): Pressure at half model levels

    Note:
        The top level of the atmosphere is excluded

    Reference:
        - https://www.ecmwf.int/sites/default/files/elibrary/2015/
        9210-part-iii-dynamics-and-numerical-procedures.pdf
        - https://confluence.ecmwf.int/display/CKB/
        ERA5%3A+compute+pressure+and+geopotential+on+model+levels%2C+geopotential+height+and+geometric+height

    Returns:
        :obj:`ndarray`: Altitude of the atmospheric near surface layer (second IFS level)
    """

    # virtual temperature (K)
    tv = t[...] * (1.0 + settings.zvir * q[...])

    # compute geopotential for 2 lowermost (near-surface) model levels
    dlog_p = np.log(ph[:, :, 1:] / ph[:, :, :-1])
    alpha = 1.0 - ((ph[:, :, :-1] / (ph[:, :, 1:] - ph[:, :, :-1])) * dlog_p)
    tv = tv * settings.rdair

    # zh is the geopotential of 'half-levels'
    # integrate zh to next half level
    increment = np.flip(tv * dlog_p, axis=2)
    zh = np.cumsum(increment, axis=2)

    # zf is the geopotential of this full level
    # integrate from previous (lower) half-level zh to the
    # full level
    increment_zh = np.insert(zh, 0, 0, axis=2)
    zf = np.flip(tv * alpha, axis=2) + increment_zh[:, :, :-1]

    alt = settings.radius * zf / settings.grav / (settings.radius - zf / settings.grav)

    return alt[:, :, -1]


def cdn(umps: np.ndarray) -> np.ndarray:
    """Neutral drag coeff at 10m

    Argument:
        umps (:obj:`ndarray`): wind speed (m/s)
    """
    return 0.0027 / umps + 0.000142 + 0.0000764 * umps


def psimhu(xd: np.ndarray) -> np.ndarray:
    """Unstable part of psimh

    Argument:
        xd (:obj:`ndarray`): model level height devided by Obukhov length
    """
    return (
        np.log((1.0 + xd * (2.0 + xd)) * (1.0 + xd * xd) / 8.0)
        - 2.0 * np.arctan(xd)
        + 1.571
    )


def psixhu(xd: np.ndarray) -> np.ndarray:
    """Unstable part of psimx

    Argument:
        xd (:obj:`ndarray`): model level height devided by Obukhov length
    """
    return 2.0 * np.log((1.0 + xd * xd) / 2.0)
