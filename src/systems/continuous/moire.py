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

def moire_potential(
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
    def laplacian(fn, x):
        grad_fn = jax.grad(fn)
        
        def hvp(v):
            _, hv = jax.jvp(grad_fn, (x,), (v,))
            return hv
        
        d = x.size # n_par * spc_dim
        
        def body(i, acc):
            v_flat = jnp.zeros(d, x.dtype).at[i].set(1.)
            v = v_flat.reshape(x.shape)
            hv = hvp(v)
            diag_i = hv.reshape(-1)[i]
            return acc + diag_i
        
        zero = jnp.asarray(0., point.real.dtype)
        return jax.lax.fori_loop(0, d, body, zero)
    
    lap_re = laplacian(lambda r: jnp.real(wavefn(r)), point)
    lap_im = laplacian(lambda r: jnp.imag(wavefn(r)), point)

    return jax.lax.complex(lap_re, lap_im)

import time, functools


def potential_energy(
    point: Float[Array, "n_par 2"],
    V_0: float,
    a_M: float,
    phi: float,
) -> Float[Array, ""]:
    potentials = jax.vmap(moire_potential, (0, None, None, None))(point, V_0, a_M, phi)
    return jnp.sum(potentials)

def interaction_energy(
    point: Float[Array, "n_par 2"],
    epsilon: float = 1e-6
) -> Float[Array, ""]:
    idx_i, idx_j = jnp.triu_indices(point.shape[0], k=1) 
    pair_energy = jax.vmap(
        lambda i, j: coulomb_repulsive(point[i], point[j], epsilon),
        in_axes=(0, 0),
        out_axes=0
    )
    interactions = pair_energy(idx_i, idx_j)
    return jnp.sum(interactions)

def kinetic_energy(
    wavefn: eqx.Module,
    point: Float[Array, "n_par 2"]
) -> Float[Array, ""]:
    start = time.time()
    lap = wfn_laplacian(wavefn, point)
    lap.block_until_ready()
    elapsed = time.time() - start
    print(f"    - laplacian: {elapsed:.4f} seconds")
    
    start = time.time()
    psi = wavefn(point)
    psi.block_until_ready()
    elapsed = time.time() - start
    print(f"    - wavefn: {elapsed:.4f} seconds")
    
    return (-1/2) * (lap / psi)

def local_energy(
    wavefn: eqx.Module,
    point: Float[Array, "n_par 2"],
    V_0: float,
    a_M: float,
    phi: float,
    epsilon: float = 1e-6
) -> Float[Array, ""]:
    
    start = time.time()
    t = kinetic_energy(wavefn, point)
    t.block_until_ready()
    elapsed = time.time() - start
    print(f"kinetic term: {elapsed:.4f} seconds")

    start = time.time()
    v = potential_energy(point, V_0, a_M, phi)
    v.block_until_ready()
    elapsed = time.time() - start
    print(f"potential term: {elapsed:.4f} seconds")

    start = time.time()
    u = interaction_energy(point, epsilon)
    u.block_until_ready()
    elapsed = time.time() - start
    print(f"interaction term: {elapsed:.4f} seconds")

    start = time.time()
    E_loc = t + v + u
    E_loc.block_until_ready()
    elapsed = time.time() - start
    print(f"summation: {elapsed:.4f} seconds")

    
    
    return E_loc

def batched_local_energy(
    wavefn: eqx.Module,
    points: Float[Array, "batch n_par 2"],
    V_0: float,
    a_M: float,
    phi: float,
    epsilon: float = 1e-6
) -> Float[Array, "batch"]:
    # with jax.profiler.TraceAnnotation("kienetic"):
    start = time.time()
    t = jax.vmap(kinetic_energy, (None, 0))(wavefn, points)
    t.block_until_ready()
    elapsed = time.time() - start
    print(f"kinetic term: {elapsed:.4f} seconds")
    # with jax.profiler.TraceAnnotation("potential"):
    start = time.time()
    v = jax.vmap(potential_energy, (0, None, None, None))(points, V_0, a_M, phi)
    v.block_until_ready()
    elapsed = time.time() - start
    print(f"potential term: {elapsed:.4f} seconds")
    # with jax.profiler.TraceAnnotation("interaction"):
    start = time.time()
    u = jax.vmap(interaction_energy, (0, None))(points, epsilon)
    u.block_until_ready()
    elapsed = time.time() - start
    print(f"interaction term: {elapsed:.4f} seconds")
    # with jax.profiler.TraceAnnotation("sum"):
    start = time.time()
    E_locs = t + v + u
    E_locs.block_until_ready()
    elapsed = time.time() - start
    print(f"sum: {elapsed:.4f} seconds")
    return E_locs