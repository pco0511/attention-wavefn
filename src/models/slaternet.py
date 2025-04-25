from typing import Callable

import equinox as eqx
import jax 
import jax.numpy as jnp
from jaxtyping import Float, Complex, Array, PRNGKeyArray

from .nn import PeriodicEmbedding, ResidualMLP

class SlaterNet(eqx.Module):
    periodic_embedding: PeriodicEmbedding
    mlp: ResidualMLP
    activation: Callable
    recip_latt_vecs: Float[Array, "num dim"] = eqx.field(static=True)
    num_recip_vecs: int = eqx.field(static=True)
    space_dim: int = eqx.field(static=True)
    num_particle: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)
    mlp_depth: int = eqx.field(static=True)
    
    def __init__(
        self,
        num_particle: int,
        recip_latt_vecs: Float[Array, "num dim"],
        hidden_dim: int,
        mlp_depth: int,
        mlp_activation: Callable=jax.nn.tanh,
        *,
        key: PRNGKeyArray
    ):
        self.activation = mlp_activation
        self.recip_latt_vecs = recip_latt_vecs
        self.num_recip_vecs = recip_latt_vecs.shape[0]
        self.space_dim = recip_latt_vecs.shape[1]
        self.num_particle = num_particle
        self.hidden_dim = hidden_dim
        self.mlp_depth = mlp_depth
        
        emb_key, mlp_key = jax.random.split(key, 2)
        
        self.periodic_embedding = PeriodicEmbedding(
            self.recip_latt_vecs, self.hidden_dim,
            key=emb_key
        )
        
        self.mlp = ResidualMLP(
            self.hidden_dim, 
            2 * num_particle,
            self.hidden_dim, 
            self.mlp_depth, 
            self.activation,
            use_final_bias=False,
            key=mlp_key
        )
            
    def __call__(
        self, x: Float[Array, "n_par dim"]
    ) -> Complex[Array, ""]:
        embedded = jax.vmap(self.periodic_embedding)(x) # shape=(n_particle, hidden_dim)
        mlp_outs = jax.vmap(self.mlp)(embedded)         # shape=(n_particle, 2 * n_particle)
        real, imag = jnp.split(mlp_outs, 2, axis=1)
        single_wave_fns = jax.lax.complex(real, imag)   # shape=(n_particle, n_particle), dtype=complex
        return jnp.linalg.det(single_wave_fns)
        
        