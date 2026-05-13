import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from vercor.dtypes import as_jax_real_array
from vercor.settings import VercorSettings


def _as_jax_array(value: ArrayLike) -> jax.Array:
    return as_jax_real_array(value)


def _virtual_temperature_from_specific_humidity(
    temperature: ArrayLike,
    specific_humidity: ArrayLike,
    virtual_temperature_correction: float,
) -> jax.Array:
    """Return virtual temperature for specific humidity in kg/kg."""

    temperature_array = _as_jax_array(temperature)
    specific_humidity_array = _as_jax_array(specific_humidity)
    return temperature_array * (
        1.0 + virtual_temperature_correction * specific_humidity_array
    )


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
    """Compute geometric altitudes at ECMWF-IFS hybrid-sigma full levels.

    Arguments:
        t (:obj:`ndarray`): Atmospheric temperature [K], ordered top-to-bottom
            along the final axis.
        q (:obj:`ndarray`): Atmospheric specific humidity [kg/kg], ordered
            top-to-bottom along the final axis.
        ph (:obj:`ndarray`): Pressure at half model levels [Pa], ordered
            top-to-bottom along the final axis.

    Note:
        The top level of the atmosphere is excluded. Returned full-level
        altitudes are ordered bottom-to-top along the final axis to preserve
        the existing consumer contract.

    Reference:
        - https://www.ecmwf.int/sites/default/files/elibrary/2015/9210-part-iii-dynamics-and-numerical-procedures.pdf
        - https://confluence.ecmwf.int/display/ECC/compute_geopotential_on_ml.py

    Returns:
        :obj:`ndarray`: Altitudes of the atmospheric full model levels [m]
    """

    return _compute_hybrid_sigma_full_level_altitudes(
        t,
        q,
        ph,
        earth_radius=settings.earth_radius,
        gravity=settings.gravity,
        rdair=settings.rdair,
        zvir=settings.zvir,
    )


def _compute_hybrid_sigma_full_level_altitudes(
    t: ArrayLike,
    q: ArrayLike,
    ph: ArrayLike,
    *,
    earth_radius: float,
    gravity: float,
    rdair: float,
    zvir: float,
) -> jax.Array:
    """Return bottom-to-top hybrid-sigma full-level geometric altitudes."""

    ph_array = _as_jax_array(ph)
    virtual_temperature = _virtual_temperature_from_specific_humidity(t, q, zvir)

    lower_half_pressure = ph_array[:, :, :-1]
    upper_half_pressure = ph_array[:, :, 1:]
    zero_lower_half_pressure = lower_half_pressure == 0.0
    safe_lower_half_pressure = jnp.where(
        zero_lower_half_pressure,
        0.1,
        lower_half_pressure,
    )

    dlog_p = jnp.log(upper_half_pressure / safe_lower_half_pressure)
    alpha_general = 1.0 - (
        safe_lower_half_pressure
        / (upper_half_pressure - safe_lower_half_pressure)
        * dlog_p
    )
    alpha = jnp.where(zero_lower_half_pressure, jnp.log(2.0), alpha_general)

    moist_temperature_rd = virtual_temperature * rdair
    half_level_geopotential_increment = jnp.flip(
        moist_temperature_rd * dlog_p,
        axis=2,
    )
    half_level_geopotential = jnp.cumsum(half_level_geopotential_increment, axis=2)

    padded_half_level_geopotential = jnp.pad(
        half_level_geopotential,
        ((0, 0), (0, 0), (1, 0)),
    )
    full_level_geopotential = (
        jnp.flip(moist_temperature_rd * alpha, axis=2)
        + padded_half_level_geopotential[:, :, :-1]
    )
    geopotential_height = full_level_geopotential / gravity
    return earth_radius * geopotential_height / (earth_radius - geopotential_height)


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
