from __future__ import annotations

__all__ = ["Mytree", "meta_leaves", "meta_flatten", "meta_map", "meta"]

import dataclasses
from copy import copy, deepcopy
from typing import Any, Callable, Dict, Iterable, Tuple

import jax
import jax.tree_util as jtu
from jax._src.tree_util import _registry
from simple_pytree import Pytree, static_field

from .bijectors import Bijector, Identity


class Mytree(Pytree):
    _pytree__meta: Dict[str, Any] = static_field()

    def __init_subclass__(cls, mutable: bool = False):
        cls._pytree__meta = dict()
        super().__init_subclass__(mutable=mutable)
        class_vars = vars(cls)
        for field, value in class_vars.items():
            if (
                field not in cls._pytree__static_fields
                and isinstance(value, dataclasses.Field)
                and value.metadata is not None
            ):
                cls._pytree__meta[field] = {**value.metadata}

    def replace(self, **kwargs: Any) -> Mytree:
        """
        Replace the values of the fields of the object.

        Args:
            **kwargs: keyword arguments to replace the fields of the object.

        Returns:
            Mytree: with the fields replaced.
        """
        fields = vars(self)
        for key in kwargs:
            if key not in fields:
                raise ValueError(f"'{key}' is not a field of {type(self).__name__}")

        pytree = copy(self)
        pytree.__dict__.update(kwargs)
        return pytree

    def replace_meta(self, **kwargs: Any) -> Mytree:
        """
        Replace the metadata of the fields.

        Args:
            **kwargs: keyword arguments to replace the metadata of the fields of the object.

        Returns:
            Mytree: with the metadata of the fields replaced.
        """
        fields = vars(self)
        for key in kwargs:
            if key not in fields:
                raise ValueError(f"'{key}' is not a field of {type(self).__name__}")

        pytree = copy(self)
        pytree.__dict__.update(_pytree__meta={**pytree._pytree__meta, **kwargs})
        return pytree

    def update_meta(self, **kwargs: Any) -> Mytree:
        """
        Update the metadata of the fields. The metadata must already exist.

        Args:
            **kwargs: keyword arguments to replace the fields of the object.

        Returns:
            Mytree: with the fields replaced.
        """
        fields = vars(self)
        for key in kwargs:
            if key not in fields:
                raise ValueError(f"'{key}' is not a field of {type(self).__name__}")

        pytree = copy(self)
        new = deepcopy(pytree._pytree__meta)
        for key, value in kwargs.items():
            if key in new:
                new[key].update(value)
            else:
                new[key] = value
        pytree.__dict__.update(_pytree__meta=new)
        return pytree

    def replace_trainable(self: Mytree, **kwargs: Dict[str, bool]) -> Mytree:
        """Replace the trainability status of local nodes of the Mytree."""
        return self.update_meta(**{k: {"trainable": v} for k, v in kwargs.items()})

    def replace_bijector(self: Mytree, **kwargs: Dict[str, Bijector]) -> Mytree:
        """Replace the bijectors of local nodes of the Mytree."""
        return self.update_meta(**{k: {"bijector": v} for k, v in kwargs.items()})

    def constrain(self) -> Mytree:
        """Transform model parameters to the constrained space according to their defined bijectors.

        Returns:
            Mytree: tranformed to the constrained space.
        """

        def _apply_constrain(meta_leaf):
            meta, leaf = meta_leaf
            return meta.get("bijector", Identity).forward(leaf)

        return meta_map(_apply_constrain, self)

    def unconstrain(self) -> Mytree:
        """Transform model parameters to the unconstrained space according to their defined bijectors.

        Returns:
            Mytree: tranformed to the unconstrained space.
        """

        def _apply_unconstrain(meta_leaf):
            meta, leaf = meta_leaf
            return meta.get("bijector", Identity).inverse(leaf)

        return meta_map(_apply_unconstrain, self)

    def stop_gradient(self) -> Mytree:
        """Stop gradients flowing through the Mytree.

        Returns:
            Mytree: with gradients stopped.
        """

        # 🛑 Stop gradients flowing through a given leaf if it is not trainable.
        def _stop_grad(leaf: jax.Array, trainable: bool) -> jax.Array:
            return jax.lax.cond(trainable, lambda x: x, jax.lax.stop_gradient, leaf)

        def _apply_stop_grad(meta_leaf):
            meta, leaf = meta_leaf
            return _stop_grad(leaf, meta.get("trainable", True))

        return meta_map(_apply_stop_grad, self)


def _toplevel_meta(pytree: Any) -> List[Dict[str, Any]]:
    """Unpacks a list of meta corresponding to the top-level nodes of the pytree.

    Args:
        pytree (Any): pytree to unpack the meta from.

    Returns:
        List[Dict[str, Any]]: meta of the top-level nodes of the pytree.
    """
    if isinstance(pytree, Iterable):
        return [None] * len(pytree)
    return [
        pytree._pytree__meta.get(field, {})
        for field, _ in sorted(vars(pytree).items())
        if field not in pytree._pytree__static_fields
    ]


def meta_leaves(
    pytree: Mytree,
    *,
    is_leaf: Callable[[Any], bool] | None = None,
) -> List[Tuple[Dict[str, Any], Any]]:
    """
    Returns the meta of the leaves of the pytree.

    Args:
        pytree (Mytree): pytree to get the meta of.
        is_leaf (Callable[[Any], bool]): predicate to determine if a node is a leaf. Defaults to None.

    Returns:
        List[Tuple[Dict[str, Any], Any]]: meta of the leaves of the pytree.
    """

    def _unpack_metadata(
        meta: Any,
        pytree: Mytree,
        is_leaf: Callable[[Any], bool] | None,
    ):
        """Recursively unpack leaf metadata."""
        if is_leaf and is_leaf(pytree):
            yield meta
            return

        if type(pytree) in _registry:  # Registry tree trick, thanks to PyTreeClass!
            leaves_values, _ = _registry[type(pytree)].to_iter(pytree)
            leaves_meta = _toplevel_meta(pytree)

        elif pytree is not None:
            yield meta
            return

        for metadata, leaf in zip(leaves_meta, leaves_values):
            yield from _unpack_metadata((metadata, leaf), leaf, is_leaf)

    return list(_unpack_metadata(pytree, pytree, is_leaf))


def meta_flatten(
    pytree: Mytree, *, is_leaf: Callable[[Any], bool] | None = None
) -> Mytree:
    """
    Returns the meta of the Mytree.

    Args:
        pytree (Mytree): Mytree to get the meta of.
        is_leaf (Callable[[Any], bool]): predicate to determine if a node is a leaf. Defaults to None.

    Returns:
        Mytree: meta of the Mytree.
    """
    return meta_leaves(pytree, is_leaf=is_leaf), jtu.tree_structure(
        pytree, is_leaf=is_leaf
    )


def meta_map(
    f: Callable[[Any, Dict[str, Any]], Any],
    pytree: Mytree,
    *rest: Any,
    is_leaf: Callable[[Any], bool] | None = None,
) -> Mytree:
    """Apply a function to a mytree where the first argument are the pytree leaves, and the second argument are the mytree metadata leaves.
    Args:
        f (Callable[[Any, Dict[str, Any]], Any]): The function to apply to the pytree.
        pytree (Mytree): The pytree to apply the function to.
        rest (Any, optional): Additional pytrees to apply the function to. Defaults to None.
        is_leaf (Callable[[Any], bool], optional): predicate to determine if a node is a leaf. Defaults to None.

    Returns:
        Mytree: The transformed pytree.
    """
    leaves, treedef = meta_flatten(pytree, is_leaf=is_leaf)
    all_leaves = [leaves] + [treedef.treedef.flatten_up_to(r) for r in rest]
    return treedef.unflatten(f(*xs) for xs in zip(*all_leaves))


def meta(pytree: Mytree, *, is_leaf: Callable[[Any], bool] | None = None) -> Mytree:
    """Returns the metadata of the mytree as a pytree.

    Args:
        pytree (Mytree): pytree to get the metadata of.

    Returns:
        Mytree: metadata of the pytree.
    """

    def _filter_meta(meta_leaf):
        meta, _ = meta_leaf
        return meta

    return meta_map(_filter_meta, pytree, is_leaf=is_leaf)
