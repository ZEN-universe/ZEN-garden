"""Canonical domain and relation representations used by the model registry."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd
import xarray as xr


def _empty_multi_index(dimensions: tuple[str, ...]) -> pd.MultiIndex:
    """Return an empty named MultiIndex with the requested arity."""
    return pd.MultiIndex.from_arrays([[] for _ in dimensions], names=dimensions)


@dataclass(frozen=True, init=False, slots=True)
class DomainSet:
    """An ordered, unique one-dimensional model domain."""

    name: str
    members: pd.Index
    doc: str = ""

    def __init__(
        self,
        name: str,
        members: pd.Index | Iterable[Any],
        doc: str = "",
    ) -> None:
        normalized = pd.Index(list(members), tupleize_cols=False, name=name)
        if normalized.has_duplicates:
            duplicates = normalized[normalized.duplicated()].unique().tolist()
            raise ValueError(
                f"Domain {name!r} contains duplicate members: {duplicates!r}"
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "members", normalized)
        object.__setattr__(self, "doc", doc)

    @property
    def data(self) -> pd.Index:
        """Return canonical members for existing registry consumers."""
        return self.members

    @property
    def coordinate_values(self) -> pd.Index:
        """Return canonical coordinate values in domain order."""
        return self.members

    def __iter__(self) -> Iterator[Any]:
        return iter(self.members)

    def __len__(self) -> int:
        return len(self.members)

    def __contains__(self, item: object) -> bool:
        return item in self.members

    def __getitem__(self, item: int | slice) -> Any:
        return self.members[item]


@dataclass(frozen=True, init=False, slots=True)
class RelationSet:
    """Allowed combinations across two or more named model domains."""

    name: str
    dimensions: tuple[str, ...]
    combinations: pd.MultiIndex
    doc: str = ""
    apply_by_default: bool = True

    def __init__(
        self,
        name: str,
        dimensions: tuple[str, ...],
        combinations: (
            pd.MultiIndex | Iterable[tuple[Any, ...]] | Mapping[Any, Iterable[Any]]
        ),
        doc: str = "",
        apply_by_default: bool = True,
    ) -> None:
        if len(dimensions) < 2:
            raise ValueError("A relation requires at least two dimensions")
        if len(set(dimensions)) != len(dimensions):
            raise ValueError("Relation dimensions must be unique")

        if isinstance(combinations, Mapping):
            if len(dimensions) != 2:
                raise ValueError(
                    "Mapping input is supported only for two-dimensional relations"
                )
            tuples = [
                (parent, child)
                for parent, children in combinations.items()
                for child in children
            ]
            index = (
                pd.MultiIndex.from_tuples(tuples, names=dimensions)
                if tuples
                else _empty_multi_index(dimensions)
            )
        elif isinstance(combinations, pd.MultiIndex):
            if combinations.nlevels != len(dimensions):
                raise ValueError("Relation arity does not match its dimensions")
            index = combinations.set_names(dimensions)
        else:
            tuples = list(combinations)
            if any(len(item) != len(dimensions) for item in tuples):
                raise ValueError("Relation contains tuples with invalid arity")
            index = (
                pd.MultiIndex.from_tuples(tuples, names=dimensions)
                if tuples
                else _empty_multi_index(dimensions)
            )

        if index.has_duplicates:
            duplicates = index[index.duplicated()].unique().tolist()
            raise ValueError(
                f"Relation {name!r} contains duplicate combinations: {duplicates!r}"
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "dimensions", dimensions)
        object.__setattr__(self, "combinations", index)
        object.__setattr__(self, "doc", doc)
        object.__setattr__(self, "apply_by_default", apply_by_default)

    @property
    def data(self) -> pd.MultiIndex:
        """Return the canonical relation combinations."""
        return self.combinations

    def __iter__(self) -> Iterator[tuple[Any, ...]]:
        return iter(self.combinations)

    def __len__(self) -> int:
        return len(self.combinations)

    def __contains__(self, item: object) -> bool:
        return item in self.combinations

    def select(self, **selectors: Any) -> pd.MultiIndex:
        """Return relation rows matching exact named-dimension selectors."""
        unknown = set(selectors).difference(self.dimensions)
        if unknown:
            raise KeyError(f"Unknown relation dimensions: {sorted(unknown)!r}")
        frame = self.combinations.to_frame(index=False)
        for dimension, value in selectors.items():
            frame = frame.loc[frame[dimension] == value]
        if frame.empty:
            return _empty_multi_index(self.dimensions)
        return pd.MultiIndex.from_frame(frame, names=self.dimensions)

    def project(self, dimension: str, **selectors: Any) -> pd.Index:
        """Select rows and return ordered unique members of one dimension."""
        if dimension not in self.dimensions:
            raise KeyError(f"Unknown relation dimension {dimension!r}")
        selected = self.select(**selectors).to_frame(index=False)
        return pd.Index(
            selected[dimension].drop_duplicates(),
            tupleize_cols=False,
            name=dimension,
        )


@dataclass(frozen=True, slots=True)
class ResolvedIndex:
    """Canonical structural index resolved for a model component."""

    dimensions: tuple[str, ...]
    combinations: pd.MultiIndex
    coordinates: dict[str, pd.Index]
    structural_mask: xr.DataArray
    applied_relations: tuple[str, ...]

    @property
    def values(self) -> list[Any]:
        """Return the scalar/tuple representation consumed by legacy boundaries."""
        if not self.dimensions:
            return [()]
        if len(self.dimensions) == 1:
            return self.combinations.get_level_values(0).tolist()
        return self.combinations.tolist()
