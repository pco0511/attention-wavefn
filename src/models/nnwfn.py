from typing import Callable

import equinox as eqx
import jax 
import jax.numpy as jnp
from jaxtyping import Float, Complex, Array, PRNGKeyArray
from netket.jax import logdet_cmplx

from .nn import PeriodicEmbedding, ResidualMLP


class DiscreteEmbedderBlock(eqx.Module):
    embedder: eqx.nn.Embedding
    layernorm: eqx.nn.LayerNorm

    def __init__(
        self,
        local_dof: int,
        embedding_dim: int,
        *,
        key: PRNGKeyArray
    ):
        self.embedder = eqx.nn.Embedding(
            num_embeddings=local_dof,
            embedding_size=embedding_dim,
            key=key
        )
        self.layernorm = eqx.nn.LayerNorm(
            shape=(embedding_dim,)
        )
    
    def __call__(
        self,
        indexseq: Float[Array, " n_par"]
    ) -> Float[Array, "n_par h_dim"]:
        embedded = jax.vmap(self.embedder)(indexseq)
        embedded = jax.vmap(self.layernorm)(embedded)
        return embedded
    
    
class PeriodicEmbedderBlock(eqx.Module):
    embedder: PeriodicEmbedding
    layernorm: eqx.nn.LayerNorm
    
    def __init__(
        self,
        recip_latt_vecs: Float[Array, "num dim"],
        embedding_dim: int,
        *,
        key: PRNGKeyArray
    ):
        self.embedder = PeriodicEmbedding(
            recip_latt_vecs=recip_latt_vecs,
            embedding_dim=embedding_dim,
            key=key
        )
        self.layernorm = eqx.nn.LayerNorm(
            shape=(embedding_dim,)
        )
    
    def __call__(
        self,
        x: Float[Array, "n_particle dim"]
    ):
        embedded = jax.vmap(self.embedder)(x)
        embedded = jax.vmap(self.layernorm)(x)
        return embedded
       
class FeedForwardBlock(eqx.Module):
    up_projection: eqx.nn.Linear
    down_projection: eqx.nn.Linear
    activation: Callable
    layernorm: eqx.nn.LayerNorm
    
    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        activation: Callable,
        *,
        key: PRNGKeyArray
    ):
        up_key, down_key = jax.random.split(key, 2)
        self.up_projection = eqx.nn.Linear(
            in_features=hidden_dim, out_features=intermediate_dim, key=up_key
        )
        self.down_projection = eqx.nn.Linear(
            in_features=intermediate_dim, out_features=hidden_dim, key=down_key
        )
        self.activation = activation
        self.layernorm = eqx.nn.LayerNorm(
            shape=(hidden_dim,)
        )
    
    def __call__(
        self,
        inputs: Float[Array, " h_dim"]
    ) -> Float[Array, " h_dim"]:
        intermidiate = self.up_projection(inputs)
        intermidiate = self.activation(intermidiate)
        outputs = self.down_projection(intermidiate) + inputs
        normed = self.layernorm(outputs)
        return normed
    
class AttentionBlock(eqx.Module):
    attention: eqx.nn.MultiheadAttention
    layernorm: eqx.nn.LayerNorm
    
    def __init__(
        self,
        hidden_dim: int,
        attention_dim: int,
        num_heads: int,
        *,
        key: PRNGKeyArray
    ):
        self.attention = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=hidden_dim,
            qk_size=attention_dim,
            use_query_bias=True,
            use_key_bias=True,
            use_value_bias=True,
            use_output_bias=True,
            key=key
        )
        self.layernorm = eqx.nn.LayerNorm(
            shape=(hidden_dim,)
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
        outputs = attention_output + inputs
        normed = jax.vmap(self.layernorm)(outputs)
        return normed


class TransformerLayer(eqx.Module):
    attention_block: AttentionBlock
    feedforward_block: FeedForwardBlock
    
    def __init__(
        self,
        hidden_dim: int,
        attention_dim: int,
        num_heads: int,
        intermediate_dim: int,
        activation: Callable,
        *,
        key: PRNGKeyArray
    ):
        attention_key, ff_key = jax.random.split(key, 2)
        self.attention_block = AttentionBlock(
            hidden_dim=hidden_dim,
            attention_dim=attention_dim,
            num_heads=num_heads,
            key=attention_key
        )
        self.feedforward_block = FeedForwardBlock(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            activation=activation,
            key=ff_key
        )
        
    
    def __call__(
        self,
        inputs: Float[Array, "n_par h_dim"]
    ) -> Float[Array, "n_par h_dim"]:
        attention_output = self.attention_block(inputs)
        output = jax.vmap(self.feedforward_block)(attention_output)
        return output

class HFBlock(eqx.Module):
    ff_layer: ResidualMLP
    n_particles: int = eqx.field(static=True)
    
    def __init__(
        self,
        hidden_dim: int,
        mlp_layers: int,
        intermediate_dim: int,
        n_particles: int,
        activation: Callable,
        *,
        key: PRNGKeyArray
    ):
        self.ff_layer = ResidualMLP(
            in_size=hidden_dim,
            out_size=2 * n_particles,
            width_size=intermediate_dim,
            depth=mlp_layers,
            activation=activation,
            final_activation=activation,
            key=key
        )
        self.n_particles = n_particles
    
    
    def __call__(
        self,
        inputs: Float[Array, "n_par h_dim"]
    ) -> Complex[Array, "n_par n_par"]:
        outputs = jax.vmap(self.ff_layer)(inputs)
        outputs_r, outputs_i = jnp.split(outputs, 2, 1)
        outputs_complex = jax.lax.complex(outputs_r, outputs_i)
        return outputs_complex

class AWFN(eqx.Module):
    embedder_block: DiscreteEmbedderBlock
    layers: list[TransformerLayer]
    pre_hf_atten_block: AttentionBlock
    hf_block: HFBlock
    
    def __init__(
        self,
        local_dof: int,
        n_particles: int,
        embedding_dim: int,
        attention_dim: int,
        intermidiate_dim: int,
        num_heads: int,
        num_attention_blocks: int,
        num_mlp_layers: int,
        mlp_hidden_dim: int,
        activation: Callable=jax.nn.gelu,
        *,
        key: PRNGKeyArray
    ):
        embedder_key, attention_key, pre_hf_key, hf_key = jax.random.split(key, 4)
        self.embedder_block = DiscreteEmbedderBlock(
            local_dof,
            embedding_dim,
            key=embedder_key
        )
        
        if num_attention_blocks > 0:
            layer_keys = jax.random.split(attention_key, num_attention_blocks)
            self.layers = [
                TransformerLayer(
                    hidden_dim=embedding_dim,
                    attention_dim=attention_dim,
                    num_heads=num_heads,
                    intermediate_dim=intermidiate_dim,
                    activation=activation,
                    key=subkey
                ) for subkey in layer_keys
            ]
        else:
            self.layers = []
        
        self.pre_hf_atten_block = AttentionBlock(
            hidden_dim=embedding_dim,
            attention_dim=attention_dim,
            num_heads=num_heads,
            key=pre_hf_key
        )
        
        self.hf_block = HFBlock(
            hidden_dim=embedding_dim,
            mlp_layers=num_mlp_layers,
            intermediate_dim=mlp_hidden_dim,
            n_particles=n_particles,
            activation=activation,
            key=hf_key
        )
        
        
    def __call__(
        self,
        indexseq: Float[Array, "n_par"]
    ) -> Complex[Array, ""]:
        embedded = self.embedder_block(indexseq)
        for layer in self.layers:
            embedded = layer(embedded)
        embedded = self.pre_hf_atten_block(embedded)
        output = self.hf_block(embedded)
        return logdet_cmplx(output)
    
class AWFNContinuous(eqx.Module):
    embedder_block: PeriodicEmbedderBlock
    layers: list[TransformerLayer]
    pre_hf_atten_block: AttentionBlock
    hf_block: HFBlock
    
    def __init__(
        self,
        recip_latt_vecs: Float[Array, "num dim"],
        n_particles: int,
        embedding_dim: int,
        attention_dim: int,
        intermidiate_dim: int,
        num_heads: int,
        num_attention_blocks: int,
        num_mlp_layers: int,
        mlp_hidden_dim: int,
        activation: Callable=jax.nn.gelu,
        *,
        key: PRNGKeyArray
    ):
        embedder_key, attention_key, pre_hf_key, hf_key = jax.random.split(key, 4)
        self.embedder_block = PeriodicEmbedderBlock(
            recip_latt_vecs,
            embedding_dim,
            key=embedder_key
        )
        
        if num_attention_blocks > 0:
            layer_keys = jax.random.split(attention_key, num_attention_blocks)
            self.layers = [
                TransformerLayer(
                    hidden_dim=embedding_dim,
                    attention_dim=attention_dim,
                    num_heads=num_heads,
                    intermediate_dim=intermidiate_dim,
                    activation=activation,
                    key=subkey
                ) for subkey in layer_keys
            ]
        else:
            self.layers = []
        
        self.pre_hf_atten_block = AttentionBlock(
            hidden_dim=embedding_dim,
            attention_dim=attention_dim,
            num_heads=num_heads,
            key=pre_hf_key
        )
        
        self.hf_block = HFBlock(
            hidden_dim=embedding_dim,
            mlp_layers=num_mlp_layers,
            intermediate_dim=mlp_hidden_dim,
            n_particles=n_particles,
            activation=activation,
            key=hf_key
        )
        
        
    def __call__(
        self,
        indexseq: Float[Array, "n_par"]
    ) -> Complex[Array, ""]:
        embedded = self.embedder_block(indexseq)
        for layer in self.layers:
            embedded = layer(embedded)
        embedded = self.pre_hf_atten_block(embedded)
        output = self.hf_block(embedded)
        return logdet_cmplx(output)