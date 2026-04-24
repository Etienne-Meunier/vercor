from typing import Any

import jax
import jax.numpy as jnp

from vercor.types import RuntimeArray


def to_host_array(array: RuntimeArray) -> Any:
    """Transfer a runtime array to a host array for external runtime boundaries."""

    return jax.device_get(jnp.asarray(array))


def transposed_host_array(array: RuntimeArray) -> Any:
    """Transfer a transposed runtime array to host memory."""

    return to_host_array(jnp.asarray(array).T)


def component_vector_speed(
    component: Any, u_field: str = "u_velocity", v_field: str = "v_velocity"
) -> jax.Array:
    """Return vector speed from component fields using JAX array math."""

    u = jnp.asarray(component.get(u_field))
    v = jnp.asarray(component.get(v_field))
    return jnp.sqrt(u**2 + v**2)
