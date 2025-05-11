from functools import partial

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Complex, Array

import netket as nk


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


def complex_laplacian(fn, x: Float[Array, "n_par 2"]) -> Complex[Array, ""]:
    original_shape = x.shape
    x_flat = x.flatten()
    d = x_flat.size
    
    def wrapped_fn_flat(x_flat_arg: Float[Array, "dim"]):
        return fn(x_flat_arg.reshape(original_shape))
    
    grad_fn = nk.jax.grad(wrapped_fn_flat)
    
    def get_diag_hessian_element(i):
        return nk.jax.grad(lambda y: grad_fn(y)[i])(x_flat)[i]
    
    diag_elements = jax.vmap(get_diag_hessian_element)(jnp.arange(d))
    return jnp.sum(diag_elements)

# def kinetic(
#     log_psi_fn: eqx.Module,
#     point: Float[Array, "n_par 2"]
# ):
#     original_shape = point.shape
#     point_flat = point.flatten()
    
#     def theta_flat(x_flat):
#         return log_psi_fn(x_flat.reshape(original_shape))
    
#     grad_theta_flat = nk.jax.grad(theta_flat)(point_flat)
#     grad_sq = jnp.vdot(grad_theta_flat, grad_theta_flat)
#     lapl_theta = complex_laplacian(log_psi_fn, point)
#     t = (-0.5) * (lapl_theta + grad_sq)
    
#     return t

def kinetic(
    log_psi_fn: eqx.Module,
    point: Float[Array, "n_par 2"]
) -> Float[Array, ""]:
    point_flat = point.reshape(-1)

    def theta(x_flat):
        return log_psi_fn(x_flat.reshape(point.shape))

    grad_theta  = nk.jax.grad(theta)
    g_flat      = grad_theta(point_flat)
    grad_sq     = jnp.dot(g_flat, g_flat)
    
    def hvp_diag(i, acc):
        e_i = jnp.zeros_like(point_flat).at[i].set(1.0)
        _, tang = jax.jvp(grad_theta, (point_flat,), (e_i,))
        return acc + tang[i]

    lapl_theta = jax.lax.fori_loop(
        0, point_flat.size, hvp_diag, 0.0
    )

    return -0.5 * (lapl_theta + grad_sq)


def local_energy(
    log_psi_fn: eqx.Module,
    point: Float[Array, "n_par 2"],
    V_0: float,
    a_M: float,
    phi: float,
    epsilon: float = 1e-6
) -> Float[Array, ""]:
    t = kinetic(log_psi_fn, point)
    
    potentials = jax.vmap(moire_potential, (0, None, None, None))(point, V_0, a_M, phi)
    v = jnp.sum(potentials)
    
    idx_i, idx_j = jnp.triu_indices(point.shape[0], k=1) 
    pair_energy = jax.vmap(
        lambda i, j: coulomb_repulsive(point[i], point[j], epsilon),
        in_axes=(0, 0),
        out_axes=0
    )
    interactions = pair_energy(idx_i, idx_j)
    u = jnp.sum(interactions)
    
    E_loc = t + v + u
    return E_loc





# def laplacian_hessian_trace(fn, x: Float[Array, "n_par 2"]) -> Float[Array, ""]:
#     original_shape = x.shape
#     x_flat = x.flatten()

#     def wrapped_fn_flat(x_flat_arg):
#         return fn(x_flat_arg.reshape(original_shape))

#     H = jax.hessian(wrapped_fn_flat)(x_flat)
#     return jnp.trace(H)
    
# def laplacian_vmap_grad(fn, x: Float[Array, "n_par 2"]) -> Float[Array, ""]:
#     original_shape = x.shape
#     x_flat = x.flatten()
#     d = x_flat.size

#     def wrapped_fn_flat(x_flat_arg):
#         return fn(x_flat_arg.reshape(original_shape))

#     grad_fn = jax.grad(wrapped_fn_flat)

#     def get_diag_hessian_element(i):
#         return jax.grad(lambda y: grad_fn(y)[i])(x_flat)[i]
    
#     diag_elements = jax.vmap(get_diag_hessian_element)(jnp.arange(d))
#     return jnp.sum(diag_elements)

# def wfn_laplacian(
#     wavefn: eqx.Module,
#     point: Float[Array, "n_par 2"]
# ) -> Float[Array, ""]:
    
#     lap_re = laplacian_hessian_trace(lambda r: jnp.real(wavefn(r)), point)
#     lap_im = laplacian_hessian_trace(lambda r: jnp.imag(wavefn(r)), point)
    
#     return jax.lax.complex(lap_re, lap_im)