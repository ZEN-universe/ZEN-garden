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
from zen_garden.model.component_types.set import GenericDomainSet, GenericRelationSet
from zen_garden.model.registries.base import Registry, component_name
from zen_garden.model.registries.set import DomainSet, RelationSet, ResolvedIndex

if TYPE_CHECKING:
    from linopy import Model as LinopyModel

    from zen_garden.model.element_registry import ElementRegistry
    from zen_garden.model.schema import ModelSchema

logger = logging.getLogger(__name__)

SetKey = str | type[GenericDomainSet] | type[GenericIndex]
RelationKey = str | type[GenericRelationSet]


def set_name(key: SetKey) -> str:
    """Return the canonical name for a set or structural-index key."""
    if isinstance(key, type) and issubclass(key, GenericIndex):
        return key.name
    return component_name(key, GenericDomainSet)


def relation_name(key: RelationKey) -> str:
    """Return the canonical name for a relation key."""
    return component_name(key, GenericRelationSet)


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
        self.relation_docs: dict[str, str] = {}
        self.subset_parents: dict[str, str] = {}
        self.coords_dataset = xr.Dataset()

    @property
    def config(self):
        """Return the canonical configuration from the model schema."""
        return self.model_schema.config

    def add_domain(
        self,
        domain: DomainSet,
        *,
        subset_of: type[GenericDomainSet] | None = None,
    ) -> None:
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
        if subset_of is not None:
            self.subset_parents[domain.name] = subset_of.name

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
        self.relation_docs[relation.name] = relation.doc

    def validate_subsets(self) -> None:
        """Validate declared domain-subset relationships after domain creation."""
        missing = set(self.subset_parents.values()).difference(self.sets)
        if missing:
            raise KeyError(f"Unknown subset parent domains: {sorted(missing)!r}")

        for child in self.subset_parents:
            visited: set[str] = set()
            current = child
            while current in self.subset_parents:
                if current in visited:
                    cycle = " -> ".join([*visited, current])
                    raise ValueError(f"Cyclic domain subset declarations: {cycle}")
                visited.add(current)
                current = self.subset_parents[current]

        for child, parent in self.subset_parents.items():
            invalid = self.sets[child].members.difference(self.sets[parent].members)
            if len(invalid):
                raise ValueError(
                    f"Domain {child!r} contains members outside parent "
                    f"{parent!r}: {invalid.tolist()!r}"
                )

    def is_subset(self, child: SetKey, parent: SetKey) -> bool:
        """Return whether ``child`` is ``parent`` or its declared descendant."""
        child_name = set_name(child)
        parent_name = set_name(parent)
        current = child_name
        visited: set[str] = set()
        while current not in visited:
            if current == parent_name:
                return True
            visited.add(current)
            if current not in self.subset_parents:
                return False
            current = self.subset_parents[current]
        raise ValueError(f"Cyclic domain subset declarations involving {child_name!r}")

    def project_relation(
        self,
        relation: RelationKey,
        target_dimension: SetKey,
        **selectors: Any,
    ) -> pd.Index:
        """Project a relation onto one exact or subset domain."""
        name = relation_name(relation)
        if name not in self.relations:
            raise KeyError(f"Unknown relation {name!r}")
        relation_set = self.relations[name]
        target_name = set_name(target_dimension)
        canonical_target = self._canonical_relation_dimension(relation_set, target_name)
        frame = relation_set.combinations.to_frame(index=False)
        if target_name != canonical_target:
            frame = frame.loc[
                frame[canonical_target].isin(self.sets[target_name].members)
            ]
        for selector, value in selectors.items():
            canonical_selector = self._canonical_relation_dimension(
                relation_set, selector
            )
            if selector != canonical_selector:
                if value not in self.sets[selector]:
                    return pd.Index([], name=target_name)
                frame = frame.loc[
                    frame[canonical_selector].isin(self.sets[selector].members)
                ]
            frame = frame.loc[frame[canonical_selector] == value]
        return pd.Index(
            frame[canonical_target].drop_duplicates(),
            tupleize_cols=False,
            name=target_name,
        )

    def relation_mask(
        self,
        relation: RelationKey,
        dimensions: Sequence[SetKey] | None = None,
    ) -> xr.DataArray:
        """Return a relation mask over canonical or requested subset domains."""
        name = relation_name(relation)
        if name not in self.relations:
            raise KeyError(f"Unknown relation {name!r}")
        relation_set = self.relations[name]
        requested = (
            relation_set.dimensions
            if dimensions is None
            else tuple(set_name(dimension) for dimension in dimensions)
        )
        return self.resolve_indices(
            requested,
            index_relations=(relation,),
        ).structural_mask

    def resolve_indices(
        self,
        dimensions: Sequence[str],
        *,
        index_relations: str | Sequence[RelationKey] = "auto",
    ) -> ResolvedIndex:
        """Resolve domains and selected relations with deterministic joins."""
        dimensions_tuple = tuple(dimensions)
        unknown = set(dimensions_tuple).difference(self.sets)
        if unknown:
            raise KeyError(f"Unknown index domains {sorted(unknown)!r}")
        if len(set(dimensions_tuple)) != len(dimensions_tuple):
            raise ValueError(f"Index domains must be unique: {dimensions_tuple!r}")

        selected_relations = self._select_relations(dimensions_tuple, index_relations)
        relation_names = tuple(name for name, _ in selected_relations)
        if not dimensions_tuple:
            return ResolvedIndex(
                dimensions=(),
                combinations=pd.MultiIndex.from_tuples([(0,)], names=["_scalar"]),
                coordinates={},
                structural_mask=xr.DataArray(True),
                applied_relations=relation_names,
            )

        frame = pd.DataFrame(index=[0])
        for name, dimension_mapping in selected_relations:
            relation_frame = self._adapt_relation_frame(
                self.relations[name], dimension_mapping
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
        index_relations: str | Sequence[RelationKey],
    ) -> tuple[tuple[str, dict[str, str]], ...]:
        """Select relations and map canonical dimensions to requested domains."""
        if index_relations == "auto":
            selected: list[tuple[str, dict[str, str]]] = []
            for name, relation in self.relations.items():
                if not relation.apply_by_default:
                    continue
                mapping = self._relation_dimension_mapping(relation, dimensions)
                if mapping is not None:
                    selected.append((name, mapping))
            signatures: dict[tuple[str, ...], list[str]] = {}
            for name, mapping in selected:
                signature = tuple(sorted(mapping.values()))
                signatures.setdefault(signature, []).append(name)
            ambiguous = {
                signature: names
                for signature, names in signatures.items()
                if len(names) > 1
            }
            if ambiguous:
                raise ValueError(f"Ambiguous default index relations: {ambiguous!r}")
            return tuple(selected)
        if isinstance(index_relations, str):
            raise ValueError(
                "index_relations must be 'auto' or a sequence of relation classes"
            )
        selected_names = tuple(relation_name(relation) for relation in index_relations)
        missing = set(selected_names).difference(self.relations)
        if missing:
            raise KeyError(f"Unknown index relations {sorted(missing)!r}")
        selected = []
        for name in selected_names:
            mapping = self._relation_dimension_mapping(self.relations[name], dimensions)
            if mapping is None:
                raise ValueError(
                    f"Relation {name!r} does not fit declared domains "
                    f"{dimensions!r}"
                )
            selected.append((name, mapping))
        return tuple(selected)

    def _relation_dimension_mapping(
        self,
        relation: RelationSet,
        requested_dimensions: tuple[str, ...],
    ) -> dict[str, str] | None:
        """Map each canonical relation dimension to one requested domain."""
        mapping: dict[str, str] = {}
        for canonical in relation.dimensions:
            if canonical in requested_dimensions:
                mapping[canonical] = canonical
                continue
            candidates = [
                requested
                for requested in requested_dimensions
                if self.is_subset(requested, canonical)
            ]
            if not candidates:
                return None
            if len(candidates) > 1:
                raise ValueError(
                    f"Relation {relation.name!r} dimension {canonical!r} "
                    f"ambiguously matches subsets {candidates!r}"
                )
            mapping[canonical] = candidates[0]
        if len(set(mapping.values())) != len(mapping):
            raise ValueError(
                f"Relation {relation.name!r} maps multiple dimensions to the "
                f"same requested domain: {mapping!r}"
            )
        return mapping

    def _adapt_relation_frame(
        self,
        relation: RelationSet,
        dimension_mapping: dict[str, str],
    ) -> pd.DataFrame:
        """Filter and rename relation tuples for requested subset domains."""
        frame = relation.combinations.to_frame(index=False)
        for canonical, requested in dimension_mapping.items():
            if canonical != requested:
                frame = frame.loc[frame[canonical].isin(self.sets[requested].members)]
        return frame.rename(columns=dimension_mapping)

    def _canonical_relation_dimension(
        self,
        relation: RelationSet,
        requested_dimension: str,
    ) -> str:
        """Resolve one exact or subset domain against a relation signature."""
        if requested_dimension not in self.sets:
            raise KeyError(f"Unknown domain {requested_dimension!r}")
        if requested_dimension in relation.dimensions:
            return requested_dimension
        candidates = [
            dimension
            for dimension in relation.dimensions
            if self.is_subset(requested_dimension, dimension)
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"Domain {requested_dimension!r} does not match exactly one "
                f"dimension of relation {relation.name!r}: {candidates!r}"
            )
        return candidates[0]

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
        if isinstance(item, type) and issubclass(
            item, (GenericDomainSet, GenericIndex)
        ):
            return item.name in self.sets
        return item in self.sets

    def __iter__(self):
        return iter(self.sets.values())
