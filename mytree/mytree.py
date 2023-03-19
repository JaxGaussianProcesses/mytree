from __future__ import annotations

__all__ = ["Mytree", "meta_leaves", "meta"]

from copy import copy, deepcopy
from typing import TYPE_CHECKING, Any, Callable, Dict, List

import jax
import jax.tree_util as jtu
from simple_pytree import Pytree, static_field

from .bijectors import Bijector, Identity


class Mytree(Pytree):
    _pytree__leaf_meta: Dict[str, Any] = static_field()
    _pytree__annotations: List[str] = static_field()

    def __init_subclass__(cls, mutable: bool = False):
        cls._pytree__leaf_meta = dict()
        cls._pytree__annotations = _get_all_annotations(cls)
        super().__init_subclass__(mutable=mutable)

    def replace(self, **kwargs: Any) -> Mytree:
        """
        Replace the values of the fields of the object with the values of the
        keyword arguments. A new object will be created with the same
        type as the original object.
        """
        fields = vars(self)
        for key in kwargs:
            if key not in fields:
                raise ValueError(f"'{key}' is not a field of {type(self).__name__}")

        pytree = copy(self)
        pytree.__dict__.update(kwargs)
        return pytree

    if not TYPE_CHECKING:

        def __setattr__(self, field: str, value: Any):
            if field not in self._pytree__annotations:
                raise AttributeError(f"{type(self)} has no annotation field {field}.")

            super().__setattr__(field, value)

            if field not in self._pytree__static_fields:
                _not_pytree = (
                    jtu.tree_map(
                        lambda x: isinstance(x, Pytree),
                        value,
                        is_leaf=lambda x: isinstance(x, Pytree),
                    )
                    == False
                )

                if _not_pytree:
                    try:
                        field_metadata = {
                            **type(self)
                            .__dict__["__dataclass_fields__"][field]
                            .metadata
                        }
                    except KeyError:
                        try:
                            field_metadata = {**type(self).__dict__[field].metadata}
                        except KeyError:
                            field_metadata = {}

                    if field_metadata.get("pytree_node", True):
                        object.__setattr__(
                            self,
                            "_pytree__leaf_meta",
                            self._pytree__leaf_meta | {field: field_metadata},
                        )

    def replace_meta(self, **kwargs: Any) -> Mytree:
        """
        Replace the values of the fields of the object with the values of the
        keyword arguments. If the object is a dataclass, `dataclasses.replace`
        will be used. Otherwise, a new object will be created with the same
        type as the original object.
        """
        for key in kwargs:
            if key not in self._pytree__leaf_meta.keys():
                raise ValueError(f"'{key}' is not a leaf of {type(self).__name__}")

        pytree = copy(self)
        pytree.__dict__.update(
            _pytree__leaf_meta={**pytree._pytree__leaf_meta, **kwargs}
        )
        return pytree

    def update_meta(self, **kwargs: Any) -> Mytree:
        """
        Replace the values of the fields of the object with the values of the
        keyword arguments. If the object is a dataclass, `dataclasses.replace`
        will be used. Otherwise, a new object will be created with the same
        type as the original object.
        """
        for key in kwargs:
            if key not in self._pytree__leaf_meta.keys():
                raise ValueError(
                    f"'{key}' is not an attribute of {type(self).__name__}"
                )

        pytree = copy(self)
        new = deepcopy(pytree._pytree__leaf_meta)
        for key, value in kwargs.items():
            if key in new:
                new[key].update(value)
            else:
                new[key] = value
        pytree.__dict__.update(_pytree__leaf_meta=new)
        return pytree

    def replace_trainable(Mytree: Mytree, **kwargs: Dict[str, bool]) -> Mytree:
        """Replace the trainability status of local nodes of the Mytree."""
        return Mytree.update_meta(**{k: {"trainable": v} for k, v in kwargs.items()})

    def replace_bijector(Mytree: Mytree, **kwargs: Dict[str, Bijector]) -> Mytree:
        """Replace the bijectors of local nodes of the Mytree."""
        return Mytree.update_meta(**{k: {"bijector": v} for k, v in kwargs.items()})

    def constrain(self) -> Mytree:
        """Transform model parameters to the constrained space according to their defined bijectors.

        Returns:
            Mytree: tranformed to the constrained space.
        """
        return _meta_map(
            lambda leaf, meta: meta.get("bijector", Identity).forward(leaf), self
        )

    def unconstrain(self) -> Mytree:
        """Transform model parameters to the unconstrained space according to their defined bijectors.

        Returns:
            Mytree: tranformed to the unconstrained space.
        """
        return _meta_map(
            lambda leaf, meta: meta.get("bijector", Identity).inverse(leaf), self
        )

    def stop_gradient(self) -> Mytree:
        """Stop gradients flowing through the Mytree.

        Returns:
            Mytree: with gradients stopped.
        """

        # 🛑 Stop gradients flowing through a given leaf if it is not trainable.
        def _stop_grad(leaf: jax.Array, trainable: bool) -> jax.Array:
            return jax.lax.cond(trainable, lambda x: x, jax.lax.stop_gradient, leaf)

        return _meta_map(
            lambda leaf, meta: _stop_grad(leaf, meta.get("trainable", True)), self
        )


def _meta_map(f: Callable[[Any, Dict[str, Any]], Any], pytree: Mytree) -> Mytree:
    """Apply a function to a pytree where the first argument are the pytree leaves, and the second argument are the pytree metadata leaves.

    Args:
        f (Callable[[Any, Dict[str, Any]], Any]): The function to apply to the pytree.
        pytree (Mytree): The pytree to apply the function to.

    Returns:
        Mytree: The transformed pytree.
    """
    leaves, treedef = jtu.tree_flatten(pytree)
    all_leaves = [leaves] + [meta_leaves(pytree)]
    return treedef.unflatten(f(*xs) for xs in zip(*all_leaves))


# Credit goes to Treeo for this function. 🏴‍☠️
def _get_all_annotations(cls: type) -> Dict[str, type]:
    """
    Returns a dictionary of all the annotations of a class.

    Args:
        cls (type): Class to get the annotations of.

    Returns:
        Dict[str, type]: Dictionary of all the annotations of the class.
    """
    d = {}
    for c in reversed(cls.mro()):
        if hasattr(c, "__annotations__"):
            d.update(**c.__annotations__)
    return d


def _toplevel_meta(pytree: Mytree) -> List[Dict[str, Any]]:
    return [
        pytree._pytree__leaf_meta[k] for k in sorted(pytree._pytree__leaf_meta.keys())
    ]


def meta_leaves(pytree: Mytree) -> List[Dict[str, Any]]:
    """
    Returns a list of the Mytree Mytree leaves' metadata.

    Args:
        pytree (Mytree): Mytree to get the metadata of the leaves.

    Returns:
        List[Dict[str, Any]]: List of the Mytree leaves' metadata.
    """
    _leaf_metadata = _toplevel_meta(pytree)

    def _nested_unpack_metadata(pytree: Mytree, *rest: Mytree) -> None:
        if isinstance(pytree, Mytree):
            _leaf_metadata.extend(_toplevel_meta(pytree))
            _unpack_metadata(pytree, *rest)

    def _unpack_metadata(pytree: Mytree, *rest: Mytree) -> None:
        pytrees = (pytree,) + rest
        _ = jax.tree_map(
            _nested_unpack_metadata,
            *pytrees,
            is_leaf=lambda x: isinstance(x, Mytree) and not x in pytrees,
        )

    _unpack_metadata(pytree)

    return _leaf_metadata


def meta(pytree: Mytree) -> Mytree:
    """
    Returns the meta of the Mytree Mytree.

    Args:
        pytree (Mytree): Mytree to get the meta of.

    Returns:
        Mytree: meta of the Mytree.
    """
    return jtu.tree_structure(pytree).unflatten(meta_leaves(pytree))
