from functools import partial

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Complex, Array

G = jnp.array([
    [jnp.cos(2 * jnp.pi * 0 / 3), jnp.sin(2 * jnp.pi * 0 / 3)],
    [jnp.cos(2 * jnp.pi * 1 / 3), jnp.sin(2 * jnp.pi * 1 / 3)],
    [jnp.cos(2 * jnp.pi * 2 / 3), jnp.sin(2 * jnp.pi * 2 / 3)],
])

def potential(
    x: Float[Array, "2"], 
    V_0: float,
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
    
    return -2 * V_0 * (cos0 + cos1 + cos2)

def coulomb_repulsive(
    x: Float[Array, "2"],
    y: Float[Array, "2"],
    epsilon: float = 1e-6
) -> Float[Array, ""]:
    return 1 / (jnp.linalg.norm(x - y) + epsilon)

def wfn_laplacian(
    wavefn: eqx.Module,
    point: Float[Array, "n_par 2"]
) -> Float[Array, ""]:
    def laplacian(fn_real):
        grad_fn = jax.grad(fn_real)
        
        def hvp(v):
            _, hv = jax.jvp(grad_fn, (point,), (v,))
            return hv
        
        d = point.size # n_par * spc_dim
        
        def body(i, acc):
            v_flat = jnp.zeros(d, point.dtype).at[i].set(1.)
            v = v_flat.reshape(point.shape)
            hv = hvp(v)
            diag_i = hv.reshape(-1)[i]
            return acc + diag_i
        
        zero = jnp.asarray(0., point.real.dtype)
        return jax.lax.fori_loop(0, d, body, zero)
    
    lap_re = laplacian(lambda r: jnp.real(wavefn(r)))
    lap_im = laplacian(lambda r: jnp.imag(wavefn(r)))

    return jax.lax.complex(lap_re, lap_im)

def local_energy(
    wavefn: eqx.Module,
    point: Float[Array, "n_par 2"],
    V_0: float,
    a_M: float,
    phi: float,
    epsilon: float = 1e-6
) -> Float[Array, ""]:
    v = jnp.sum(jax.vmap(potential, (0, None, None, None))(point, V_0, a_M, phi))
    idx_i, idx_j = jnp.triu_indices(point.shape[0], k=1) 
    pair_energy = jax.vmap(
        lambda i, j: coulomb_repulsive(point[i], point[j], epsilon),
        in_axes=(0, 0),
        out_axes=0
    )
    interactions = pair_energy(idx_i, idx_j)
    u = jnp.sum(interactions)
    t = (-1/2) * wfn_laplacian(wavefn, point) / (wavefn(point) + epsilon)
    return t + v + u