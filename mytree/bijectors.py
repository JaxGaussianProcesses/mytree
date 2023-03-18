from __future__ import annotations

__all__ = ["Bijector", "Identity", "Softplus"]

import importlib
import jax.numpy as jnp
from typing import Callable
from dataclasses import dataclass
from simple_pytree import Pytree, static_field

@dataclass
class Bijector(Pytree):
    forward: Callable = static_field()
    inverse: Callable = static_field()

def __init__(self, forward: Callable, inverse: Callable) -> None:
    """Initialise the bijector.

    Args:
        forward(Callable): The forward transformation.
        inverse(Callable): The inverse transformation.

    Returns:
        Bijector: A bijector.
    """
    self.forward = forward
    self.inverse = inverse


Identity = Bijector(forward=lambda x: x, inverse=lambda x: x)

Softplus = Bijector(
    forward=lambda x: jnp.log(1 + jnp.exp(x)),
    inverse=lambda x: jnp.log(jnp.exp(x) - 1.0),
)
