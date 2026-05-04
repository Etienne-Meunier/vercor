import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from vercor.dtypes import as_jax_real_array
from vercor.settings import VercorSettings


def _as_jax_array(value: ArrayLike) -> jax.Array:
    return as_jax_real_array(value)


def qsat(tk: ArrayLike) -> jax.Array:
    """The saturation humidity of air (kg/m^3)

    Argument:
        tk (:obj:`ndarray`): temperature (K)
    """
    tk_array = _as_jax_array(tk)
    return 640380.0 / jnp.exp(5107.4 / tk_array)


def qsat_august_eqn(ps: ArrayLike, tk: ArrayLike) -> jax.Array:
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
    ps_array = _as_jax_array(ps)
    tk_array = _as_jax_array(tk)
    return 0.622 / ps_array * 10 ** (9.4051 - 2353.0 / tk_array) * 133.322


def compute_pressure_levels(
    sp: ArrayLike,
    hya: ArrayLike,
    hyb: ArrayLike,
) -> jax.Array:
    """Compute pressure levels

    Arguments:
        sp (:obj:`ndarray`): Atmospheric surface pressure
        hya (:obj:`ndarray`): Hybrid sigma level A coefficient for vertical grid
        hyb (:obj:`ndarray`): Hybrid sigma level B coefficient for vertical grid

    Return:
        :obj:`ndarray`
    """
    sp_array = _as_jax_array(sp)
    hya_array = _as_jax_array(hya)
    hyb_array = _as_jax_array(hyb)
    return (
        hya_array[jnp.newaxis, jnp.newaxis, :]
        + hyb_array[jnp.newaxis, jnp.newaxis, :] * sp_array[:, :, jnp.newaxis]
    )


def get_altitudes_hybrid_sigma_levels(
    settings: VercorSettings,
    t: ArrayLike,
    q: ArrayLike,
    ph: ArrayLike,
) -> jax.Array:
    """Computes the altitudes at ECMWF Integrated Forecasting System
    (ECMWF-IFS) model half- and full-levels (for 137 levels model reanalysis: L137)

    Arguments:
        t (:obj:`ndarray`): Atmospheric temperture [K]
        q (:obj:`ndarray`): Atmospheric specific humidity [kg/kg]
        ph (:obj:`ndarray`): Pressure at half model levels [Pa]

    Note:
        The top level of the atmosphere is excluded

    Reference:
        - https://www.ecmwf.int/sites/default/files/elibrary/2015/9210-part-iii-dynamics-and-numerical-procedures.pdf
        - https://confluence.ecmwf.int/display/ECC/compute_geopotential_on_ml.py

    Returns:
        :obj:`ndarray`: Altitudes of the atmospheric full model levels [m]
    """

    # virtual temperature (K)
    t_array = _as_jax_array(t)
    q_array = _as_jax_array(q)
    ph_array = _as_jax_array(ph)

    tv = t_array * (1.0 + settings.zvir * q_array)

    # dlog_p[0] = np.log(ph[:, :, 1:] / 0.1)
    # alpha[0] = np.log(2)
    dlog_p = jnp.log(ph_array[:, :, 1:] / ph_array[:, :, :-1])
    alpha = 1.0 - (
        (ph_array[:, :, :-1] / (ph_array[:, :, 1:] - ph_array[:, :, :-1])) * dlog_p
    )
    tv *= settings.rdair

    # zh is the geopotential of 'half-levels'
    # integrate zh to next half level
    increment = jnp.flip(tv * dlog_p, axis=2)
    zh = jnp.cumsum(increment, axis=2)

    # zf is the geopotential of this full level
    # integrate from previous (lower) half-level zh to the
    # full level
    increment_zh = jnp.pad(zh, ((0, 0), (0, 0), (1, 0)))
    zf = jnp.flip(tv * alpha, axis=2) + increment_zh[:, :, :-1]

    alt = (
        settings.earth_radius
        * zf
        / settings.gravity
        / (settings.earth_radius - zf / settings.gravity)
    )

    return alt[:, :, :]


def cdn(umps: ArrayLike) -> jax.Array:
    """Neutral drag coeff at 10m

    Argument:
        umps (:obj:`ndarray`): wind speed (m/s)
    """
    umps_array = _as_jax_array(umps)
    return 0.0027 / umps_array + 0.000142 + 0.0000764 * umps_array


def psimhu(xd: ArrayLike) -> jax.Array:
    """Unstable part of psimh

    Argument:
        xd (:obj:`ndarray`): model level height devided by Obukhov length
    """

    xd_array = _as_jax_array(xd)
    return (
        jnp.log((1.0 + xd_array * (2.0 + xd_array)) * (1.0 + xd_array * xd_array) / 8.0)
        - 2.0 * jnp.arctan(xd_array)
        + 1.571
    )


def psixhu(xd: ArrayLike) -> jax.Array:
    """Unstable part of psimx

    Argument:
        xd (:obj:`ndarray`): model level height devided by Obukhov length
    """
    xd_array = _as_jax_array(xd)
    return 2.0 * jnp.log((1.0 + xd_array * xd_array) / 2.0)


def compute_air_density(
    settings: VercorSettings,
    pf: ArrayLike,
    t: ArrayLike,
) -> jax.Array:
    """Air density (kg/m^3)"""
    pf_array = _as_jax_array(pf)
    t_array = _as_jax_array(t)
    return settings.mwdair / settings.rgas * pf_array / t_array


def compute_potential_temperature(
    settings: VercorSettings,
    tbot: ArrayLike,
    pf: ArrayLike,
) -> jax.Array:
    """Potential temperature (K)"""
    tbot_array = _as_jax_array(tbot)
    pf_array = _as_jax_array(pf)
    return tbot_array * (settings.p0 / pf_array) ** settings.cappa
