from typing import Callable
from typing import Any, Literal

import numpy as np

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Complex, Array, PRNGKeyArray
import numpy as np

def _identity(x):
    return x

def _default_floating_dtype():
    if jax.config.jax_enable_x64:
        return jnp.float64
    else:
        return jnp.float32

def _is_array(element: Any) -> bool:
    """Returns `True` if `element` is a JAX array or NumPy array."""
    return isinstance(element, (np.ndarray, np.generic, jax.Array))

class ResidualMLP(eqx.Module, strict=True):
    layers: tuple[eqx.nn.Linear, ...]
    activation: Callable
    final_activation: Callable
    use_bias: bool = eqx.field(static=True)
    use_final_bias: bool = eqx.field(static=True)
    use_final_residual: bool = eqx.field(static=True)
    in_size: int | Literal["scalar"] = eqx.field(static=True)
    out_size: int | Literal["scalar"] = eqx.field(static=True)
    width_size: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)

    def __init__(
        self,
        in_size: int | Literal["scalar"],
        out_size: int | Literal["scalar"],
        width_size: int,
        depth: int,
        activation: Callable = jax.nn.tanh,
        final_activation: Callable = _identity,
        use_bias: bool = True,
        use_final_bias: bool = True,
        use_final_residual: bool = False,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        dtype = _default_floating_dtype() if dtype is None else dtype
        keys = jax.random.split(key, depth + 1)
        layers = []
        if depth == 0:
            layers.append(
                eqx.nn.Linear(in_size, out_size, use_final_bias, dtype=dtype, key=keys[0])
            )
        else:
            layers.append(
                eqx.nn.Linear(in_size, width_size, use_bias, dtype=dtype, key=keys[0])
            )
            for i in range(depth - 1):
                layers.append(
                    eqx.nn.Linear(
                        width_size, width_size, use_bias, dtype=dtype, key=keys[i + 1]
                    )
                )
            layers.append(
                eqx.nn.Linear(width_size, out_size, use_final_bias, dtype=dtype, key=keys[-1])
            )
        self.layers = tuple(layers)
        self.in_size = in_size
        self.out_size = out_size
        self.width_size = width_size
        self.depth = depth
        # In case `activation` or `final_activation` are learnt, then make a separate
        # copy of their weights for every neuron.
        self.activation = eqx.filter_vmap(
            eqx.filter_vmap(lambda: activation, axis_size=width_size), axis_size=depth
        )()
        if out_size == "scalar":
            self.final_activation = final_activation
        else:
            self.final_activation = eqx.filter_vmap(
                lambda: final_activation, axis_size=out_size
            )()
        self.use_bias = use_bias
        self.use_final_bias = use_final_bias
        self.use_final_residual = use_final_residual
        
        if use_final_residual and self.depth > 0:
            assert out_size == width_size
        
    def __call__(
        self, 
        x: Float[Array, "indim"], 
        *,
        key: PRNGKeyArray | None = None
    ) -> Float[Array, "outdim"]:
        for i, layer in enumerate(self.layers[:-1]):
            y = layer(x)
            layer_activation = jax.tree_util.tree_map(
                lambda y: y[i] if _is_array(y) else y, self.activation
            )
            y = eqx.filter_vmap(lambda a, b: a(b))(layer_activation, y)
            if i == 0 and self.in_size != self.width_size:
                x = y
            else:
                x = jax.tree_util.tree_map(
                    lambda a, b: a + b,
                    x, y
                )
        y = self.layers[-1](x)
        if self.out_size == "scalar":
            y = self.final_activation(y)
        else:
            y = eqx.filter_vmap(lambda a, b: a(b))(self.final_activation, y)

        if self.use_final_residual:
            x = jax.tree_util.tree_map(
                lambda a, b: a + b, 
                x, y
            )
        else:
            x = y
        return x

    
class PeriodicEmbedding(eqx.Module):
    linear: eqx.nn.Linear
    recip_latt_vecs: np.ndarray = eqx.field(static=True)
    num_recip_vecs: int = eqx.field(static=True)
    space_dim: int = eqx.field(static=True)
    embedding_dim: int = eqx.field(static=True)
    
    def __init__(
        self,
        recip_latt_vecs: np.ndarray,
        embedding_dim: int,
        *,
        key: PRNGKeyArray,
    ):
        self.recip_latt_vecs = recip_latt_vecs
        self.num_recip_vecs = recip_latt_vecs.shape[0]
        self.space_dim = recip_latt_vecs.shape[1]
        self.embedding_dim = embedding_dim
        self.linear = eqx.nn.Linear(
            2 * self.num_recip_vecs,
            self.embedding_dim,
            use_bias=False,
            key=key
        )
    
    def __call__(
        self, x: Float[Array, "dim"]
    ) -> Float[Array, "e_dim"]:
        phi = jnp.einsum("j,ij->i", x, self.recip_latt_vecs)
        sines = jnp.sin(phi)
        cosines = jnp.cos(phi)
        embedded = self.linear(jnp.concat([sines, cosines]))
        return embedded
        