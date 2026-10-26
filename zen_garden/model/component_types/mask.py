"""Reusable structural masks for optimization-model components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, final

import xarray as xr

from zen_garden.model.component_types.component import GenericComponent
from zen_garden.model.linear_expression import validate_component_indices

if TYPE_CHECKING:
    from zen_garden.model.constructor import ModelConstructor
    from zen_garden.model.registries.mask import MaskRegistry


class GenericMask(GenericComponent, ABC):
    """Base class for reusable variable and constraint construction masks.

    Masks are built after all parameters and before any variables. They may
    therefore depend on sets, parameters, configuration, time mappings, and
    other masks, but never on variables or constraints.
    """

    name: ClassVar[str]
    indices: ClassVar[tuple[str, ...]]
    doc: ClassVar[str]
    dependencies: ClassVar[tuple[type["GenericMask"], ...]] = ()
    _abstract_component = True

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.__dict__.get("_abstract_component", False):
            return
        if "indices" not in cls.__dict__ or not isinstance(cls.indices, tuple):
            raise TypeError(f"{cls.__name__}.indices must be a tuple")
        if not all(isinstance(index, str) and index for index in cls.indices):
            raise TypeError(f"{cls.__name__}.indices must contain non-empty set names")
        if "build" in cls.__dict__:
            raise TypeError(f"{cls.__name__} must not override GenericMask.build()")
        if not all(
            isinstance(dependency, type) and issubclass(dependency, GenericMask)
            for dependency in cls.dependencies
        ):
            raise TypeError(
                f"{cls.__name__}.dependencies must contain GenericMask classes"
            )

    @classmethod
    @final
    def build(
        cls,
        model_constructor: "ModelConstructor",
        registry: "MaskRegistry",
    ) -> xr.DataArray:
        """Construct and validate this structural mask."""
        mask = cls.get_mask(model_constructor, registry)
        if not isinstance(mask, xr.DataArray) or mask.dtype != bool:
            raise TypeError(f"Mask {cls.name!r} must build a boolean DataArray")
        mask = mask.fillna(False)
        validate_component_indices(cls.name, cls.indices, mask)
        return mask

    @classmethod
    @abstractmethod
    def get_mask(
        cls,
        model_constructor: "ModelConstructor",
        registry: "MaskRegistry",
    ) -> xr.DataArray:
        """Return the structural mask using canonical model dimensions."""
