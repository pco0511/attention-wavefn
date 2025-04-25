from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Float, Complex, Array

G = jnp.array([
    [jnp.cos(2 * jnp.pi * 0 / 3), jnp.sin(2 * jnp.pi * 0 / 3)],
    [jnp.cos(2 * jnp.pi * 1 / 3), jnp.sin(2 * jnp.pi * 1 / 3)],
    [jnp.cos(2 * jnp.pi * 2 / 3), jnp.sin(2 * jnp.pi * 2 / 3)],
])

def moire_potential(
    x: Float[Array, "2"], 
    V0: float,
    a_M: float,
    phi: float
) -> Float[Array, ""]:
    g = (4 * jnp.pi) / (jnp.sqrt(3) * a_M)
    
    g0 = g * jnp.array([jnp.cos(2 * jnp.pi * 0 / 3), jnp.sin(2 * jnp.pi * 0 / 3)])
    g1 = g * jnp.array([jnp.cos(2 * jnp.pi * 1 / 3), jnp.sin(2 * jnp.pi * 1 / 3)])
    g2 = g * jnp.array([jnp.cos(2 * jnp.pi * 2 / 3), jnp.sin(2 * jnp.pi * 2 / 3)])
    
    cos0 = jnp.cos(jnp.dot(g0, x) + phi)
    cos1 = jnp.cos(jnp.dot(g1, x) + phi)
    cos2 = jnp.cos(jnp.dot(g2, x) + phi)
    
    return -2 * V0 * (cos0 + cos1 + cos2)