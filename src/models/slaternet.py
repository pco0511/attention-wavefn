from collections.abc import Callable

import equinox as eqx
import jax 
import jax.numpy as jnp
from jaxtyping import Float, Complex, Array, PRNGKeyArray

from nn import R2toC


class SlaterNet(eqx.Module):
    embedding: eqx.nn.Linear
    mlps: list[eqx.nn.MLP]
    recip_latt_vecs: Float[Array, "dim dim"]  
    
    def __init__(
        self,
        num_particle: int,
        recip_latt_vecs: Float[Array, "dim dim"],
        embedding_dim: int,
        hidden_dim: int,
        mlp_depth: int,
        mlp_activation: Callable=jax.nn.tanh,
        *,
        key: PRNGKeyArray
    ):
        
        self.num_particle = num_particle
        self.space_dim = recip_latt_vecs.shape[0]
        self.recip_latt_vecs = recip_latt_vecs
        
        embedding_key, key = jax.random.split(key, 2)
        self.embedding = eqx.nn.Linear(2 * self.space_dim, embedding_dim, key=embedding_key)
        
        mlp_keys = jax.random.split(key, num_particle)
        self.mlps = [
            eqx.nn.MLP(embedding_dim, 2, hidden_dim, mlp_depth, mlp_activation, key=subkey) for subkey in mlp_keys
        ]
        
            
    def __call__(
        self, x: Float[Array, "n_particle dim"]
    ) -> Complex:
        y = jnp.einsum("ik,jk->ij", x, self.recip_latt_vecs) # shape=(n_particle, dim)
        sines = jnp.sin(y)      # shape=(n_particle, dim)
        cosines = jnp.cos(y)    # shape=(n_particle, dim)
        features = jnp.concat([sines, cosines], axis=1)      # shape=(n_particle, 2 * dim)
        embedded = jax.vmap(self.embedding)(features)        # shape=(n_particle, embedding_dim)
        single_wave_fn = jnp.concat(
            [
                jax.vmap(R2toC)(jax.vmap(self.mlps[idx])(embedded)) # shape=(n_partocle, 1), dtype=complex
                for idx in range(self.num_particle)
            ],
            axis=1
        ) # shape=(n_partocle, n_partocle), dtype=complex
        return jnp.linalg.det(single_wave_fn)
        
        