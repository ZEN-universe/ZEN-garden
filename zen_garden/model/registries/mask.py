"""Registry for dependency-ordered structural model masks."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

import xarray as xr

if TYPE_CHECKING:
    from zen_garden.model.component_types.mask import GenericMask
    from zen_garden.model.constructor import ModelConstructor

logger = logging.getLogger(__name__)


class MaskRegistry:
    """Build, cache, align, and combine structural component masks."""

    _TECHNOLOGY_DIMS = (
        "set_technologies",
        "set_conversion_technologies",
        "set_storage_technologies",
        "set_transport_technologies",
        "set_retrofitting_technologies",
    )
    _LOCATION_DIMS = ("set_location", "set_nodes", "set_edges")

    def __init__(self) -> None:
        self._masks: dict[str, xr.DataArray] = {}
        self._types: dict[str, type[GenericMask]] = {}

    def build_all(
        self,
        mask_owners: dict[type[GenericMask], "ModelConstructor"],
    ) -> None:
        """Build requested masks and their dependencies in topological order."""
        self._masks.clear()
        self._types.clear()
        owners = dict(mask_owners)

        def include_dependencies(
            mask_type: type[GenericMask], owner: "ModelConstructor"
        ) -> None:
            existing = self._types.get(mask_type.name)
            if existing is not None and existing is not mask_type:
                raise ValueError(
                    f"Multiple mask classes define {mask_type.name!r}: "
                    f"{existing.__name__} and {mask_type.__name__}"
                )
            self._types[mask_type.name] = mask_type
            owners.setdefault(mask_type, owner)
            for dependency in mask_type.dependencies:
                include_dependencies(dependency, owner)

        for mask_type, owner in list(owners.items()):
            include_dependencies(mask_type, owner)

        remaining = set(owners)
        completed: set[type[GenericMask]] = set()
        while remaining:
            ready = sorted(
                (
                    mask_type
                    for mask_type in remaining
                    if set(mask_type.dependencies).issubset(completed)
                ),
                key=lambda mask_type: mask_type.name,
            )
            if not ready:
                cycle = ", ".join(sorted(mask_type.name for mask_type in remaining))
                raise ValueError(f"Cyclic mask dependencies: {cycle}")
            for mask_type in ready:
                mask = mask_type.build(owners[mask_type], self)
                if not isinstance(mask, xr.DataArray) or mask.dtype != bool:
                    raise TypeError(
                        f"Mask {mask_type.name!r} must build a boolean DataArray"
                    )
                self._masks[mask_type.name] = mask.fillna(False)
                completed.add(mask_type)
                remaining.remove(mask_type)

    def __getitem__(self, mask: str | type[GenericMask]) -> xr.DataArray:
        """Return a previously constructed canonical mask."""
        name = mask if isinstance(mask, str) else mask.name
        try:
            return self._masks[name]
        except KeyError as error:
            raise KeyError(f"Mask {name!r} has not been constructed") from error

    def combine_for_indices(
        self,
        masks: Sequence[type[GenericMask]],
        model_constructor: "ModelConstructor",
        index_sets,
    ) -> xr.DataArray | None:
        """Align masks to a component index and combine them with logical AND."""
        if not masks:
            return None
        index_values, index_names = index_sets
        target = model_constructor.optimization_model.sets.indices_to_mask(
            index_values, index_names, (0, 0)
        )[0]
        if target.size == 0:
            # SetRegistry assigns unique dimension names to empty variables when
            # similarly named non-empty coordinates already exist in Linopy.
            # No structural mask is needed because the index itself is empty.
            return None
        return self._combine(masks, target)

    def combine_for_constraint(
        self,
        masks: Sequence[type[GenericMask]],
        constraint: Any,
    ) -> xr.DataArray | None:
        """Align masks to a Linopy constraint and combine them with AND."""
        if not masks:
            return None
        target = constraint.rhs
        return self._combine(masks, target)

    def _combine(
        self,
        masks: Iterable[type[GenericMask]],
        target: xr.DataArray,
    ) -> xr.DataArray:
        combined = xr.full_like(target, True, dtype=bool)
        for mask_type in masks:
            projected = self._project(self[mask_type], target)
            combined &= projected
        return combined

    def _project(self, mask: xr.DataArray, target: xr.DataArray) -> xr.DataArray:
        """Project canonical technology/location dimensions onto a target array."""
        rename: dict[str, str] = {}
        self._add_alias_rename(
            mask, target, "set_technologies", self._TECHNOLOGY_DIMS, rename
        )
        self._add_alias_rename(
            mask, target, "set_location", self._LOCATION_DIMS, rename
        )
        projected = mask.rename(rename)

        unexpected = set(projected.dims).difference(target.dims)
        if unexpected:
            names = ", ".join(sorted(str(name) for name in unexpected))
            raise ValueError(
                f"Mask dimensions {names} cannot be projected onto "
                f"target dimensions {tuple(target.dims)}"
            )
        indexers = {
            dim: target.coords[dim] for dim in projected.dims if dim in target.coords
        }
        projected = projected.reindex(indexers, fill_value=False)
        return projected.broadcast_like(target).astype(bool)

    @staticmethod
    def _add_alias_rename(
        mask: xr.DataArray,
        target: xr.DataArray,
        canonical: str,
        aliases: tuple[str, ...],
        rename: dict[str, str],
    ) -> None:
        if canonical not in mask.dims or canonical in target.dims:
            return
        candidates = [name for name in aliases if name in target.dims]
        if len(candidates) != 1:
            raise ValueError(
                f"Cannot map canonical mask dimension {canonical!r} to "
                f"target dimensions {tuple(target.dims)}"
            )
        rename[canonical] = candidates[0]

    def log_reduction(self, name: str, mask: xr.DataArray) -> None:
        """Log the number of active entries in a resolved component mask."""
        active = int(mask.sum().item())
        logger.debug("Mask %s retains %d/%d entries", name, active, mask.size)
