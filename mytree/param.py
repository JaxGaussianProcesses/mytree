from __future__ import annotations

__all__ = ["param_field"]

from typing import Any
import dataclasses
from .bijectors import Bijector

def param_field(
    bijector: Bijector,
    trainable: bool = True,
    **kwargs: Any,
) -> dataclasses.Field:
    """Used for marking default parameter transformations, trainable statuses and prior distributions for Mytree.

    Args:
        transform (Bijector): The default bijector assigned to the given attribute, upon Mytree initialisation.
        trainable (bool): The default trainability status assigned to the given attribute, upon Mytree initialisation.
        **kwargs (Any): If any are passed then they are passed on to `dataclasses.field`.

    Returns:
        dataclasses.Field: A `dataclasses.Field` object with the `bijector`, `trainable` metadata set.
    """

    try:
        metadata = dict(kwargs["metadata"])
    except KeyError:
        metadata = kwargs["metadata"] = {}
    
    if "bijector" in metadata:
        raise ValueError("Cannot use metadata with `bijector` already set.")
    
    if "trainable" in metadata:
        raise ValueError("Cannot use metadata with `trainable` already set.")
    
    if "pytree_node" in metadata:
        raise ValueError("Cannot use metadata with `pytree_node` already set.")
    
    metadata["bijector"] = bijector
    metadata["trainable"] = trainable
    metadata["pytree_node"] = True

    return dataclasses.field(**kwargs)