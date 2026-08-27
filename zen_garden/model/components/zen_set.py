"""Set representations used by ZEN-garden model components."""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from ordered_set import OrderedSet


class BaseSet(ABC):
    """Common interface for simple and indexed model sets."""

    def __init__(self, name: str = "", doc: str = "") -> None:
        self.name = name
        self.doc = doc

    @property
    @abstractmethod
    def coordinate_values(self) -> OrderedSet[Any]:
        """Return all values used for this set's shared model coordinate."""

    @abstractmethod
    def __iter__(self) -> Iterator[Any]:
        """Iterate over members or index keys."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of members or index keys."""

    @abstractmethod
    def __contains__(self, item: object) -> bool:
        """Test membership among members or index keys."""

    @abstractmethod
    def is_indexed(self) -> bool:
        """Return whether this is a mapping of child sets."""


class SimpleSet(BaseSet):
    """An ordered collection of model-set members."""

    def __init__(self, data: Iterable[Any], name: str = "", doc: str = "") -> None:
        super().__init__(name=name, doc=doc)
        self.members: OrderedSet[Any] = OrderedSet(data)
        self.data = self.members

    @property
    def coordinate_values(self) -> OrderedSet[Any]:
        return self.members

    def __iter__(self) -> Iterator[Any]:
        return iter(self.members)

    def __len__(self) -> int:
        return len(self.members)

    def __contains__(self, item: object) -> bool:
        return item in self.members

    def __getitem__(self, item: int | slice):
        return self.members[item]

    def is_indexed(self) -> bool:
        return False

    def get_index_name(self) -> None:
        return None

    def __repr__(self) -> str:
        return f"SimpleSet({list(self.members)!r})"


class IndexedSet(BaseSet):
    """A mapping from index keys to ordered child sets."""

    def __init__(
        self,
        data: Mapping[Any, Iterable[Any]],
        name: str = "",
        doc: str = "",
        index_set: str | None = None,
    ) -> None:
        super().__init__(name=name, doc=doc)
        self.data = data
        self.index_set = index_set or "UnnamedIndex"
        self.children = {
            key: SimpleSet(values, name=f"{name}[{key}]")
            for key, values in data.items()
        }
        self._coordinate_values: OrderedSet[Any] = OrderedSet()
        for child in self.children.values():
            self._coordinate_values.update(child.members)

    @property
    def coordinate_values(self) -> OrderedSet[Any]:
        return self._coordinate_values

    def __iter__(self) -> Iterator[Any]:
        return iter(self.children)

    def __len__(self) -> int:
        return len(self.children)

    def __contains__(self, item: object) -> bool:
        return item in self.children

    def __getitem__(self, item: Any) -> SimpleSet:
        return self.children[item]

    def is_indexed(self) -> bool:
        return True

    def get_index_name(self) -> str:
        return self.index_set

    def __repr__(self) -> str:
        return f"IndexedSet(keys={list(self.children)!r})"
