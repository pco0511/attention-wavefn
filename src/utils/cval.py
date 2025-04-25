import jax
import jax.numpy as jnp
from jaxtyping import Float, Array, Complex, PyTree

def tree_real(tree: PyTree[Complex[Array, "..."], "T"]) -> PyTree[Float[Array, "..."], "T"]:
    return jax.tree_util.tree_map(jnp.real, tree)

def tree_imag(tree: PyTree[Complex[Array, "..."], "T"]) -> PyTree[Float[Array, "..."], "T"]:
    return jax.tree_util.tree_map(jnp.imag, tree)

def tree_complex(
    real_tree: PyTree[Float[Array, "..."], "T"], 
    imag_tree: PyTree[Float[Array, "..."], "T"]
) -> PyTree[Complex[Array, "..."], "T"]:
    return jax.tree_util.tree_map(jax.lax.complex, real_tree, imag_tree)

def cval_diff_wrapper(
    transformation
):
    def cval_transform(
        cval_fun,
        argnums=0,
        has_aux=False
    ):
        def real(x):
            return tree_real(cval_fun(x))
        
        def imag(x):
            return tree_imag(cval_fun(x))
        
        transformed_real = transformation(real, argnums=argnums, has_aux=has_aux)
        transformed_imag = transformation(imag, argnums=argnums, has_aux=has_aux)
        return lambda x: tree_complex(transformed_real(x), transformed_imag(x))
    return cval_transform

cval_grad = cval_diff_wrapper(jax.grad)
cval_value_and_grad = cval_diff_wrapper(jax.value_and_grad)
cval_jacobian = cval_diff_wrapper(jax.jacobian)
cval_jacfwd = cval_diff_wrapper(jax.jacfwd)
cval_jacrev = cval_diff_wrapper(jax.jacrev)
cval_hessian = cval_diff_wrapper(jax.hessian)


# def cval_grad(
#     cval_fun, 
#     argnums=0, 
#     has_aux=False,
#     allow_int=False, 
#     reduce_axes=()
# ):
#     real = lambda x: tree_real(cval_fun(x))
#     imag = lambda x: tree_imag(cval_fun(x))
#     real_grad = jax.grad(real, argnums=argnums, has_aux=has_aux, allow_int=allow_int, reduce_axes=reduce_axes)
#     imag_grad = jax.grad(imag, argnums=argnums, has_aux=has_aux, allow_int=allow_int, reduce_axes=reduce_axes)
#     complex_grad = lambda x: tree_complex(real_grad(x), imag_grad(x))
#     return complex_grad

