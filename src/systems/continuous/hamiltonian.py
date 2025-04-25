from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Float, Complex, Array


type WaveFn = Callable[[Float[Array, "n_particle dim"]], Complex[Array, ""]]
type PotentialFn = Callable[[Float[Array, "dim"]], Float[Array, ""]]
type PairInteraction = Callable[[Float[Array, "dim"], Float[Array, "dim"]], Float[Array, ""]]

def laplacians(psi: WaveFn, r: Float[Array, "n_particle dim"]) -> Complex[Array, ""]:
    pass



def build_hamiltonian(V: PotentialFn):
    
    def H(psi: WaveFn) -> WaveFn:
        pass