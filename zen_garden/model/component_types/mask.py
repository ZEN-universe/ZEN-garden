"""Reusable structural masks for optimization-model components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

import xarray as xr

if TYPE_CHECKING:
    from zen_garden.model.constructor import ModelConstructor
    from zen_garden.model.registries.mask import MaskRegistry


class GenericMask(ABC):
    """Base class for reusable variable and constraint construction masks.

    Masks are built after all parameters and before any variables. They may
    therefore depend on sets, parameters, configuration, time mappings, and
    other masks, but never on variables or constraints.
    """

    name: ClassVar[str]
    dependencies: ClassVar[list[type["GenericMask"]]] = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "name"):
            raise TypeError(f"{cls.__name__} must define 'name'")
        if not isinstance(cls.dependencies, list):
            raise TypeError(f"{cls.__name__}.dependencies must be a list")
        if not all(
            isinstance(dependency, type) and issubclass(dependency, GenericMask)
            for dependency in cls.dependencies
        ):
            raise TypeError(
                f"{cls.__name__}.dependencies must contain GenericMask classes"
            )

    @classmethod
    @abstractmethod
    def build(
        cls,
        model_constructor: "ModelConstructor",
        registry: "MaskRegistry",
    ) -> xr.DataArray:
        """Return the structural mask using canonical model dimensions."""
