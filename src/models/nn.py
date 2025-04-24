import jax
import equinox as eqx
from jaxtyping import Float, Complex, Array, PRNGKeyArray

def R2toC(x: Float[Array, 2]) -> Complex:
    return x[0] + 1j * x[1]


class DeterminantBlock(eqx.Module):
    pass


class ProjectorBlock(eqx.Module):
    pass

