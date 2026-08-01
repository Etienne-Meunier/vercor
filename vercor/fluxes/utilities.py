import jax
import jax.numpy as jnp
from jax import custom_jvp
from jax.typing import ArrayLike
from typing import cast

from vercor.dtypes import as_jax_real_array
from vercor.physics import PhysicalConstants

_SAFE_SQRT_GRAD_FLOOR = 1e-8


@custom_jvp
def safe_sqrt(x: ArrayLike) -> jax.Array:
    """sqrt with a bounded derivative at ``x == 0``.

    Plain ``jnp.sqrt`` has an infinite local derivative at 0
    (``0.5 / sqrt(0)``). At masked/land or becalmed grid cells the argument
    legitimately hits exactly 0; reverse-mode autodiff still evaluates that
    infinite local derivative even when an outer ``jnp.where``/``jnp.maximum``
    discards the branch, and ``0 * inf == nan`` survives the multiply -- the
    NaN then contaminates unrelated cells through the coupled physics. The
    primal value is unchanged; only the gradient is floored.
    """

    return jnp.sqrt(as_jax_real_array(x))


@safe_sqrt.defjvp
def _safe_sqrt_jvp(
    primals: tuple[ArrayLike],
    tangents: tuple[ArrayLike],
) -> tuple[jax.Array, jax.Array]:
    (x,) = primals
    (x_dot,) = tangents
    primal_out = jnp.sqrt(as_jax_real_array(x))
    grad = 0.5 / jnp.maximum(primal_out, _SAFE_SQRT_GRAD_FLOOR)
    return primal_out, grad * x_dot


def qsat(tk: ArrayLike) -> jax.Array:
    """The saturation humidity of air (kg/m^3)

    Argument:
        tk (:obj:`ndarray`): temperature (K)
    """
    tk_array = as_jax_real_array(tk)
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
    ps_array = as_jax_real_array(ps)
    tk_array = as_jax_real_array(tk)
    return 0.622 / ps_array * 10 ** (9.4051 - 2353.0 / tk_array) * 133.322


def cdn(umps: ArrayLike) -> jax.Array:
    """Neutral drag coeff at 10m

    Argument:
        umps (:obj:`ndarray`): wind speed (m/s)
    """
    umps_array = jnp.maximum(as_jax_real_array(umps), _SAFE_SQRT_GRAD_FLOOR)
    return 0.0027 / umps_array + 0.000142 + 0.0000764 * umps_array


def psimhu(xd: ArrayLike) -> jax.Array:
    """Unstable part of psimh

    Argument:
        xd (:obj:`ndarray`): model level height devided by Obukhov length
    """

    xd_array = as_jax_real_array(xd)
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
    xd_array = as_jax_real_array(xd)
    return 2.0 * jnp.log((1.0 + xd_array * xd_array) / 2.0)


def compute_air_density(
    constants: PhysicalConstants,
    pf: ArrayLike,
    t: ArrayLike,
) -> jax.Array:
    """Air density (kg/m^3)"""
    pf_array = as_jax_real_array(pf)
    t_array = as_jax_real_array(t)
    return cast(
        jax.Array,
        constants.dry_air_molecular_weight
        / constants.universal_gas_constant
        * pf_array
        / t_array,
    )


def compute_potential_temperature(
    constants: PhysicalConstants,
    tbot: ArrayLike,
    pf: ArrayLike,
) -> jax.Array:
    """Potential temperature (K)"""
    tbot_array = as_jax_real_array(tbot)
    pf_array = as_jax_real_array(pf)
    return (
        tbot_array
        * (constants.reference_pressure / pf_array) ** constants.dry_air_kappa
    )
