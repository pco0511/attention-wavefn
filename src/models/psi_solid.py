from collections.abc import Callable

import equinox as eqx
import jax 
import jax.numpy as jnp
from jaxtyping import Float, Complex, Array, PRNGKeyArray

from nn import PeriodicEmbedding

class AttentionBlock(eqx.Module):
    attention: eqx.nn.MultiheadAttention
    num_heads: int = eqx.field(static=True)
    ff_layer: eqx.nn.Linear
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        activation: Callable=jax.nn.tanh,
        *,
        key: PRNGKeyArray
    ):
        embedder_key, attention_key, ff_key = jax.random.split(key, 2)
        self.num_heads = num_heads
        self.attention = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=hidden_dim,
            use_query_bias=True,
            use_key_bias=True,
            use_value_bias=True,
            use_output_bias=True,
            key=attention_key
        )
        self.activation = activation
        self.ff_layer = eqx.nn.Linear(hidden_dim, hidden_dim, key=ff_key)
    
    def __call__(
        self,
        inputs: Float[Array, "n_par, h_dim"]
    ) -> Float[Array, "n_par, h_dim"]:
        attention_output = self.attention(
            query=inputs,
            key=inputs,
            value=inputs
        )
        ff_input = attention_output + inputs
        ff_output = self.activation(jax.vmap(self.ff_layer)(ff_input))
        return ff_output + ff_input
    
class ProjectorBlock(eqx.Module):
    linear: eqx.nn.Linear
    hidden_dim: int = eqx.field(statis=True)
    num_particle: int = eqx.field(statis=True)
    num_det: int = eqx.field(statis=True)
    
    def __init__(
        self,
        hidden_dim: int,
        num_particle: int,
        num_det: int,
        *,
        key: PRNGKeyArray
    ):
        self.hidden_dim = hidden_dim
        self.num_particle = num_particle
        self.num_det = num_det
        self.linear = eqx.nn.Linear(
            hidden_dim,
            2 * num_particle * num_det,
            use_bias=False,
            key=key
        )
    
    def __call__(
        self,
        input: Float[Array, "h_dim"],
    ) -> Complex[Array, "n_det n_par"]:
        real, imag = self.linear(input).reshape(2, self.num_det, self.num_particle)
        return jax.lax.complex(real, imag)
    
    
class PsiSolid(eqx.Module):
    periodic_embedding: PeriodicEmbedding
    attention_blocks: list[AttentionBlock]
    activation: Callable
    projector: ProjectorBlock
    recip_latt_vecs: Float[Array, "num dim"] = eqx.field(static=True)
    num_recip_vecs: int = eqx.field(static=True)
    space_dim: int = eqx.field(static=True)
    num_particle: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    num_blocks: int = eqx.field(static=True)
    num_det: int = eqx.field(static=True)
    
    def __init__(
        self,
        num_particle: int,
        recip_latt_vecs: Float[Array, "num dim"],
        hidden_dim: int,
        num_heads: int,
        num_blocks: int,
        num_det: int,
        activation: Callable=jax.nn.tanh,
        *
        key: PRNGKeyArray
    ):
        self.activation = activation
        self.recip_latt_vecs = recip_latt_vecs
        self.num_recip_vecs = recip_latt_vecs.shape[0]
        self.space_dim = recip_latt_vecs.shape[1]
        self.num_particle = num_particle
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.num_det = num_det
        
        emb_key, atten_key, proj_key = jax.random.split(key, 3)
        
        self.periodic_embedding = PeriodicEmbedding(
            self.recip_latt_vecs, self.hidden_dim,
            key=emb_key
        )
        
        attention_block_keys = jax.random.split(atten_key, self.num_blocks)
        self.attention_blocks = [
            AttentionBlock(
                self.hidden_dim,
                self.num_heads,
                self.activation,
                key=subkey
            ) for subkey in attention_block_keys
        ]
        
        self.projectors = ProjectorBlock(
            self.hidden_dim,
            self.num_particle,
            self.num_det,
            key=proj_key
        )
    
    def __call__(
        self, x: Float[Array, "n_particle dim"]
    ) -> Complex:
        z = jax.vmap(self.periodic_embedding)(x)         # shape=(n_particle, hidden_dim)
        
        for attention_block in self.attention_blocks:
            z = attention_block(z)                       # shape=(n_particle, hidden_dim)
            
        single_wave_fns = jax.vmap(self.projector)(z)    # shape=(n_particle, n_det, n_particle)
        single_wave_fns = jnp.permute_dims(single_wave_fns, (1, 0, 2)) # shape=(n_det, n_particle, n_particle)
        multi_wave_fns = jnp.linalg.det(single_wave_fns) # shape=(n_det, )
        return jnp.sum(multi_wave_fns)
        