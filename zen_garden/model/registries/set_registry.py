"""Canonical model-domain and relation registry."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr

from zen_garden.model.component_types.index import GenericIndex
from zen_garden.model.component_types.set import GenericSet
from zen_garden.model.registries.base import Registry, component_name
from zen_garden.model.registries.set import DomainSet, RelationSet, ResolvedIndex

if TYPE_CHECKING:
    from linopy import Model as LinopyModel

    from zen_garden.model.element_registry import ElementRegistry
    from zen_garden.model.schema import ModelSchema

logger = logging.getLogger(__name__)

SetKey = str | type[GenericSet] | type[GenericIndex]


def set_name(key: SetKey) -> str:
    """Return the canonical name for a set or structural-index key."""
    if isinstance(key, type) and issubclass(key, GenericIndex):
        return key.name
    return component_name(key, GenericSet)


def _coordinate_array(values: Sequence[Any] | pd.Index) -> np.ndarray:
    """Return a one-dimensional coordinate array, including tuple members."""
    members = list(values)
    if any(isinstance(member, tuple) for member in members):
        array = np.empty(len(members), dtype=object)
        array[:] = members
        return array
    if not members:
        return np.array([], dtype=object)
    return np.asarray(members)


class SetRegistry(Registry):
    """Store domains, sparse relations, and their xarray coordinates."""

    def __init__(
        self,
        model_schema: "ModelSchema",
        element_registry: "ElementRegistry",
    ):
        super().__init__()
        self.model_schema = model_schema
        self.element_registry = element_registry
        self.sets: dict[str, DomainSet] = {}
        self.relations: dict[str, RelationSet] = {}
        self.coords_dataset = xr.Dataset()
        self.legacy_indexed_set_relations: dict[str, str] = {}

    @property
    def config(self):
        """Return the canonical configuration from the model schema."""
        return self.model_schema.config

    def add_domain(self, domain: DomainSet) -> None:
        """Validate and register one canonical model domain."""
        if domain.name in self.sets:
            raise ValueError(f"Domain {domain.name!r} already added")
        self.sets[domain.name] = domain
        self.coords_dataset = self.coords_dataset.assign_coords(
            {domain.name: _coordinate_array(domain.members)}
        )
        self.docs[domain.name] = self.compile_doc_string(
            domain.doc,
            name=domain.name,
            index_list=[],
        )

    def add_relation(self, relation: RelationSet) -> None:
        """Validate and register one canonical sparse relation."""
        if relation.name in self.relations:
            raise ValueError(f"Relation {relation.name!r} already added")
        missing_dimensions = set(relation.dimensions).difference(self.sets)
        if missing_dimensions:
            raise KeyError(
                f"Relation {relation.name!r} uses unknown domains "
                f"{sorted(missing_dimensions)!r}"
            )
        for dimension in relation.dimensions:
            invalid = relation.combinations.get_level_values(dimension).difference(
                self.sets[dimension].members
            )
            if len(invalid):
                raise ValueError(
                    f"Relation {relation.name!r} contains members outside domain "
                    f"{dimension!r}: {invalid.tolist()!r}"
                )
        self.relations[relation.name] = relation

    def finalize_structural_sets(self) -> None:
        """Register derived domains and relations after declared sets exist."""
        if "set_location" not in self.sets:
            locations = [*self.sets["set_nodes"], *self.sets["set_edges"]]
            self.add_domain(DomainSet("set_location", locations, "Model locations"))

        relation_name = "relation_technology_location"
        if relation_name not in self.relations:
            node_technologies: list[Any] = []
            for subset in (
                "set_conversion_technologies",
                "set_storage_technologies",
                "set_retrofitting_technologies",
            ):
                node_technologies.extend(self.sets[subset])
            node_technologies = list(dict.fromkeys(node_technologies))
            combinations = [
                (technology, node)
                for technology in node_technologies
                for node in self.sets["set_nodes"]
            ]
            combinations.extend(
                (technology, edge)
                for technology in self.sets["set_transport_technologies"]
                for edge in self.sets["set_edges"]
            )
            self.add_relation(
                RelationSet(
                    relation_name,
                    ("set_technologies", "set_location"),
                    combinations,
                )
            )

        capacity_relation_name = "relation_technology_capacity_type"
        if capacity_relation_name not in self.relations:
            storage_technologies = set(self.sets["set_storage_technologies"])
            combinations = [
                (technology, capacity_type)
                for technology in self.sets["set_technologies"]
                for capacity_type in self.sets["set_capacity_types"]
                if capacity_type != "energy" or technology in storage_technologies
            ]
            self.add_relation(
                RelationSet(
                    capacity_relation_name,
                    ("set_technologies", "set_capacity_types"),
                    combinations,
                    "Meaningful capacity categories for each technology.",
                )
            )
        self._register_technology_subtype_relations()

    def _register_technology_subtype_relations(self) -> None:
        """Register exact-name projections of general technology relations."""
        technology_relations = [
            relation
            for relation in self.relations.values()
            if "set_technologies" in relation.dimensions
        ]
        for relation in technology_relations:
            technology_position = relation.dimensions.index("set_technologies")
            for subtype in (
                "set_conversion_technologies",
                "set_storage_technologies",
                "set_transport_technologies",
                "set_retrofitting_technologies",
            ):
                dimensions = list(relation.dimensions)
                dimensions[technology_position] = subtype
                derived_dimensions = tuple(dimensions)
                if any(
                    existing.dimensions == derived_dimensions
                    for existing in self.relations.values()
                ):
                    continue
                combinations = [
                    row
                    for row in relation.combinations
                    if row[technology_position] in self.sets[subtype]
                ]
                name = f"{relation.name}_{subtype.removeprefix('set_')}"
                self.add_relation(
                    RelationSet(
                        name,
                        derived_dimensions,
                        combinations,
                        relation.doc,
                        relation.apply_by_default,
                    )
                )

    def get_related(
        self,
        source_dimension: SetKey,
        source_member: Any,
        target_dimension: SetKey,
    ) -> pd.Index:
        """Return a unique two-dimensional relation projection."""
        source_name = set_name(source_dimension)
        target_name = set_name(target_dimension)
        matches = [
            relation
            for relation in self.relations.values()
            if set(relation.dimensions) == {source_name, target_name}
        ]
        if len(matches) != 1:
            raise ValueError(
                "Expected exactly one relation between "
                f"{source_dimension!r} and {target_dimension!r}, found "
                f"{[relation.name for relation in matches]!r}"
            )
        return matches[0].project(
            target_name,
            **{source_name: source_member},
        )

    def create_custom_set(
        self,
        list_index: Sequence[str],
        index_relations: str | Sequence[str] = "auto",
    ) -> tuple[list[Any], list[str]]:
        """Resolve readable domains to the legacy registry boundary shape."""
        resolved = self.resolve_indices(list_index, index_relations=index_relations)
        return resolved.values, list(resolved.dimensions)

    def resolve_indices(
        self,
        dimensions: Sequence[str],
        *,
        index_relations: str | Sequence[str] = "auto",
    ) -> ResolvedIndex:
        """Resolve domains and selected relations with deterministic joins."""
        dimensions_tuple = tuple(dimensions)
        unknown = set(dimensions_tuple).difference(self.sets)
        if unknown:
            raise KeyError(f"Unknown index domains {sorted(unknown)!r}")
        if len(set(dimensions_tuple)) != len(dimensions_tuple):
            raise ValueError(f"Index domains must be unique: {dimensions_tuple!r}")

        relation_names = self._select_relations(dimensions_tuple, index_relations)
        if not dimensions_tuple:
            return ResolvedIndex(
                dimensions=(),
                combinations=pd.MultiIndex.from_tuples([(0,)], names=["_scalar"]),
                coordinates={},
                structural_mask=xr.DataArray(True),
                applied_relations=relation_names,
            )

        frame = pd.DataFrame(index=[0])
        for relation_name in relation_names:
            relation_frame = self.relations[relation_name].combinations.to_frame(
                index=False
            )
            if len(frame.columns) == 0:
                frame = relation_frame
                continue
            shared = [column for column in frame if column in relation_frame]
            if shared:
                frame = frame.merge(
                    relation_frame,
                    on=shared,
                    how="inner",
                    sort=False,
                )
            else:
                frame = frame.merge(relation_frame, how="cross", sort=False)

        for dimension in dimensions_tuple:
            if dimension in frame:
                continue
            domain_frame = pd.DataFrame({dimension: self.sets[dimension].members})
            frame = frame.merge(domain_frame, how="cross", sort=False)

        frame = frame.loc[:, list(dimensions_tuple)]
        if frame.duplicated().any():
            duplicates = frame.loc[frame.duplicated(keep=False)].to_dict("records")
            raise ValueError(
                "Resolved index contains duplicate combinations: " f"{duplicates!r}"
            )
        combinations = pd.MultiIndex.from_frame(frame, names=dimensions_tuple)
        coordinates = {
            dimension: self.sets[dimension].members for dimension in dimensions_tuple
        }
        structural_mask = xr.DataArray(
            False,
            coords={
                dimension: _coordinate_array(coordinate)
                for dimension, coordinate in coordinates.items()
            },
            dims=dimensions_tuple,
        )
        if len(combinations):
            selectors = tuple(
                xr.DataArray(
                    combinations.get_level_values(dimension).to_numpy(),
                    dims="_entry",
                )
                for dimension in dimensions_tuple
            )
            structural_mask.loc[selectors] = True
        return ResolvedIndex(
            dimensions=dimensions_tuple,
            combinations=combinations,
            coordinates=coordinates,
            structural_mask=structural_mask,
            applied_relations=relation_names,
        )

    def _select_relations(
        self,
        dimensions: tuple[str, ...],
        index_relations: str | Sequence[str],
    ) -> tuple[str, ...]:
        """Validate and select automatic or explicit relation names."""
        requested = set(dimensions)
        if index_relations == "auto":
            selected = tuple(
                name
                for name, relation in self.relations.items()
                if relation.apply_by_default
                and set(relation.dimensions).issubset(requested)
            )
            signatures: dict[tuple[str, ...], list[str]] = {}
            for name in selected:
                signature = tuple(sorted(self.relations[name].dimensions))
                signatures.setdefault(signature, []).append(name)
            ambiguous = {
                signature: names
                for signature, names in signatures.items()
                if len(names) > 1
            }
            if ambiguous:
                raise ValueError(f"Ambiguous default index relations: {ambiguous!r}")
            return selected
        if isinstance(index_relations, str):
            raise ValueError("index_relations must be 'auto' or a sequence of names")
        selected = tuple(index_relations)
        missing = set(selected).difference(self.relations)
        if missing:
            raise KeyError(f"Unknown index relations {sorted(missing)!r}")
        invalid = [
            name
            for name in selected
            if not set(self.relations[name].dimensions).issubset(requested)
        ]
        if invalid:
            raise ValueError(
                f"Relations do not fit declared domains {dimensions!r}: {invalid!r}"
            )
        return selected

    @staticmethod
    def tuple_to_arr(
        index_values: Sequence[Any] | pd.Series,
        index_list: Sequence[str],
        unique: bool = False,
    ) -> tuple[xr.DataArray | np.ndarray, ...]:
        """Split resolved rows into one pairwise selector per dimension."""
        if not index_list:
            return ()
        if len(index_values) == 0:
            arrays: list[xr.DataArray | np.ndarray] = [
                xr.DataArray([], dims="_entry") for _ in index_list
            ]
        elif len(index_list) == 1:
            arrays = [
                xr.DataArray(_coordinate_array(list(index_values)), dims="_entry")
            ]
        else:
            arrays = [
                xr.DataArray(_coordinate_array(values), dims="_entry")
                for values in zip(*index_values, strict=True)
            ]
        if unique:
            return tuple(np.unique(array.data) for array in arrays)
        return tuple(arrays)

    def get_coord(
        self, data: xr.DataArray | np.ndarray[Any, Any] | Sequence[Any], name: str
    ):
        """Return the canonical domain coordinate or unique supplied values."""
        if name in self.sets:
            return self.coords_dataset.coords[name]
        raw = data.data if isinstance(data, xr.DataArray) else data
        return np.unique(raw)

    def indices_to_mask(
        self,
        index_values: Sequence[Any] | pd.Series,
        index_list: list[str],
        bounds: Any,
        model: "LinopyModel | None" = None,
    ) -> tuple[xr.DataArray, Any, Any]:
        """Build an index mask and bounds for the Linopy boundary."""
        index_arrs = self.tuple_to_arr(index_values, index_list)
        canonical_coords = {
            name: (
                _coordinate_array(self.sets[name].members)
                if name in self.sets
                else np.unique(index_arr.data)
            )
            for index_arr, name in zip(index_arrs, index_list, strict=True)
        }
        backend_names = list(index_list)
        if model is not None:
            for position, name in enumerate(index_list):
                if len(canonical_coords[name]) == 0 and name in model.variables.coords:
                    backend_names[position] = f"{name}_{uuid.uuid4()}"
        coords = {
            backend_name: canonical_coords[canonical_name]
            for canonical_name, backend_name in zip(
                index_list, backend_names, strict=True
            )
        }
        mask = xr.DataArray(False, coords=coords, dims=backend_names)
        if len(index_values):
            mask.loc[index_arrs] = True
        lower, upper = self._create_variable_bounds(
            bounds,
            mask,
            index_arrs,
            index_values,
        )
        return mask, lower, upper

    @staticmethod
    def _create_variable_bounds(
        bounds: Any,
        mask: xr.DataArray,
        index_arrs: tuple[xr.DataArray | np.ndarray, ...],
        index_values: Sequence[Any] | pd.Series,
    ) -> tuple[Any, Any]:
        """Create explicit-dimension lower and upper bound arrays."""
        if bounds is None:
            return -np.inf, np.inf
        lower = xr.full_like(mask, -np.inf, dtype=float)
        upper = xr.full_like(mask, np.inf, dtype=float)
        if isinstance(bounds, tuple):
            if isinstance(bounds[0], xr.DataArray):
                lower.loc[index_arrs] = bounds[0].loc[index_arrs]
                upper.loc[index_arrs] = bounds[1].loc[index_arrs]
            else:
                lower[...] = bounds[0]
                upper[...] = bounds[1]
        elif isinstance(bounds, np.ndarray):
            lower.loc[index_arrs] = bounds[:, 0]
            upper.loc[index_arrs] = bounds[:, 1]
        elif callable(bounds):
            low_values: list[Any] = []
            high_values: list[Any] = []
            for value in index_values:
                arguments = (
                    value
                    if len(index_arrs) > 1 and isinstance(value, tuple)
                    else (value,)
                )
                low, high = bounds(*arguments)
                low_values.append(low)
                high_values.append(high)
            lower.loc[index_arrs] = low_values
            upper.loc[index_arrs] = high_values
        else:
            raise ValueError(
                "bounds must be None, a tuple, an array, or a callable; "
                f"received {type(bounds).__name__}"
            )
        return lower, upper

    def on_off_technologies(self, domain: SetKey = "set_technologies") -> DomainSet:
        """Return technologies for which on/off operation must be modeled."""
        domain_name = set_name(domain)
        return DomainSet(
            domain_name,
            (
                technology
                for technology in self.sets[domain_name]
                if self._check_on_off_modeled(str(technology))
            ),
        )

    def _check_on_off_modeled(self, technology: str) -> bool:
        """Return whether a technology needs explicit on/off behavior."""
        element = self.element_registry.get_element_by_name(technology)
        if element is None:
            raise KeyError(f"Technology {technology!r} is not registered")
        if not hasattr(element, "min_load"):
            raise AttributeError(f"Technology {technology!r} has no min_load")
        unique_min_load = list(set(element.min_load.values))
        return not (len(unique_min_load) == 1 and unique_min_load[0] == 0)

    def __getitem__(self, name: SetKey) -> DomainSet:
        """Return a canonical domain by name or declarative set/index class."""
        return self.sets[set_name(name)]

    def __contains__(self, item: object) -> bool:
        if isinstance(item, type) and issubclass(item, (GenericSet, GenericIndex)):
            return item.name in self.sets
        return item in self.sets

    def __iter__(self):
        return iter(self.sets.values())
