from collections.abc import Callable
from typing import Literal

import equinox as eqx
import jax 
import jax.numpy as jnp
from jaxtyping import Float, Complex, Array, PRNGKeyArray

from nn import ResidualMLP

class SlaterNet(eqx.Module):
    mlp: ResidualMLP
    activation: Callable
    recip_latt_vecs: Float[Array, "dim dim"] = eqx.field(static=True)
    space_dim: int = eqx.field(static=True)
    num_particle: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)
    mlp_depth: int = eqx.field(static=True)
    
    def __init__(
        self,
        num_particle: int,
        recip_latt_vecs: Float[Array, "dim dim"],
        hidden_dim: int,
        mlp_depth: int,
        mlp_activation: Callable=jax.nn.tanh,
        *,
        key: PRNGKeyArray
    ):
        self.mlp_activation = mlp_activation
        self.recip_latt_vecs = recip_latt_vecs
        self.space_dim = recip_latt_vecs.shape[0]
        self.num_particle = num_particle
        self.hidden_dim = hidden_dim
        self.mlp_depth = mlp_depth
        
        self.mlp = ResidualMLP(
            2 * self.space_dim, 
            2 * num_particle,
            self.hidden_dim, 
            self.mlp_depth, 
            self.mlp_activation,
            use_final_bias=False,
            key=key
        )
            
    def __call__(
        self, x: Float[Array, "n_par dim"]
    ) -> Complex:
        y = jnp.einsum("ik,jk->ij", x, self.recip_latt_vecs) # shape=(n_particle, space_dim)
        sines = jnp.sin(y)                                   # shape=(n_particle, space_dim)
        cosines = jnp.cos(y)                                 # shape=(n_particle, space_dim)
        features = jnp.concat([sines, cosines], axis=1)      # shape=(n_particle, 2 * space_dim)
        mlp_outputs = jax.vmap(self.mlp)(features)           # shape=(n_particle, 2 * n_particle)
        real, imag = jnp.split(mlp_outputs, 2, axis=1)
        single_wave_fns = jax.lax.complex(real, imag)        # shape=(n_particle, n_particle), dtype=complex
        return jnp.linalg.det(single_wave_fns)
        
        