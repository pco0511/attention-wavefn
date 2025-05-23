from typing import Callable

import equinox as eqx
import jax 
import jax.numpy as jnp
from jaxtyping import Float, Complex, Array, PRNGKeyArray
from netket.jax import logdet_cmplx
from jax.scipy.special import logsumexp

from .nn import PeriodicEmbedding, ResidualMLP

class AttentionBlock(eqx.Module):
    attention: eqx.nn.MultiheadAttention
    activation: Callable
    ff_layer: ResidualMLP
    
    def __init__(
        self,
        hidden_dim: int,
        attention_dim: int,
        num_heads: int,
        num_mlp_layers: int,
        intermediate_dim: int,
        activation: Callable=jax.nn.tanh,
        *,
        key: PRNGKeyArray
    ):
        attention_key, ff_key = jax.random.split(key, 2)
        self.attention = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=hidden_dim,
            qk_size=attention_dim,
            use_query_bias=True,
            use_key_bias=True,
            use_value_bias=True,
            use_output_bias=True,
            key=attention_key
        )
        self.activation = activation
        self.ff_layer = ResidualMLP(
            hidden_dim,
            hidden_dim,
            intermediate_dim,
            num_mlp_layers,
            activation,
            final_activation=activation,
            use_final_residual=True,
            key=ff_key
        )
        
    def __call__(
        self,
        inputs: Float[Array, "n_par h_dim"]
    ) -> Float[Array, "n_par h_dim"]:
        attention_output = self.attention(
            query=inputs,
            key_=inputs,
            value=inputs
        )
        ff_input = attention_output + inputs
        ff_output = jax.vmap(self.ff_layer)(ff_input)
        return ff_output
    
class ProjectorBlock(eqx.Module):
    linear: eqx.nn.Linear
    num_particle: int = eqx.field(static=True)
    num_det: int = eqx.field(static=True)
    
    def __init__(
        self,
        hidden_dim: int,
        num_particle: int,
        num_det: int,
        *,
        key: PRNGKeyArray
    ):
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
    num_recip_vecs: int = eqx.field(static=True)
    space_dim: int = eqx.field(static=True)
    num_particle: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)
    attention_dim: int = eqx.field(static=True)
    intermediate_dim: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    num_blocks: int = eqx.field(static=True)
    num_mlp_layers: int = eqx.field(static=True)
    num_det: int = eqx.field(static=True)
    
    def __init__(
        self,
        num_particle: int,
        recip_latt_vecs: Float[Array, "num dim"],
        hidden_dim: int,
        intermediate_dim: int,
        attention_dim: int,
        num_heads: int,
        num_blocks: int,
        num_mlp_layers: int,
        num_det: int,
        activation: Callable=jax.nn.tanh,
        *,
        key: PRNGKeyArray
    ):
        self.activation = activation
        self.num_recip_vecs = recip_latt_vecs.shape[0]
        self.space_dim = recip_latt_vecs.shape[1]
        self.num_particle = num_particle
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.num_mlp_layers = num_mlp_layers
        self.num_det = num_det
        
        emb_key, atten_key, proj_key = jax.random.split(key, 3)
        
        self.periodic_embedding = PeriodicEmbedding(
            recip_latt_vecs, self.hidden_dim,
            key=emb_key
        )
        
        attention_block_keys = jax.random.split(atten_key, self.num_blocks)
        self.attention_blocks = [
            AttentionBlock(
                self.hidden_dim,
                self.attention_dim,
                self.num_heads,
                self.num_mlp_layers,
                self.intermediate_dim,
                self.activation,
                key=subkey
            ) for subkey in attention_block_keys
        ]
        
        self.projector = ProjectorBlock(
            self.hidden_dim,
            self.num_particle,
            self.num_det,
            key=proj_key
        )
        
    def __call__(
        self, x: Float[Array, "n_particle dim"]
    ) -> Complex[Array, ""]:
        z = jax.vmap(self.periodic_embedding)(x)         # shape=(n_particle, hidden_dim)
        for attention_block in self.attention_blocks:
            z = attention_block(z)                       # shape=(n_particle, hidden_dim)
        single_wave_fns = jax.vmap(self.projector)(z)    # shape=(n_particle, n_det, n_particle)
        single_wave_fns = jnp.permute_dims(single_wave_fns, (1, 0, 2)) # shape=(n_det, n_particle, n_particle)
        
        logdets = jax.vmap(logdet_cmplx)(single_wave_fns)
        log_psi = logsumexp(logdets)
        
        return log_psi