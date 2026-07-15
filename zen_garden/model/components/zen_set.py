"""A set class that is similar to pyomo.Set."""

from typing import List, Sequence, overload

from ordered_set import OrderedSet
from typing_extensions import override


class ZenSet(OrderedSet):
    """Similiar to pyomo.Set."""

    def __init__(self, data, name="", doc="", index_set: str | None = None):
        """Initialize the set.

        :param data: The data of the set, either an iterable or a dictionary for
            an indexed set
        :param name: The name of the set
        :param doc: The corresponding docstring
        :param index_set: The name of the index set
        """
        if index_set is None:
            index_set = "UnnamedIndex"

        # set attributes
        self.data = data
        self.name = name
        self.doc = doc
        self.superset = OrderedSet()

        if isinstance(data, dict):
            # init the children
            self.ordered_data = {
                k: ZenSet(v, name=f"{name}[{k}]") for k, v in data.items()
            }

            # we set all the supersets
            for child in self.ordered_data.values():
                self.superset.update(child)
            for child in self.ordered_data.values():
                child.superset.update(self.superset)

            # for an indexed sets the init data are the keys
            data = data.keys()
            self.indexed = True
            self.index_set = index_set

        else:
            self.indexed = False
            # index set it None
            self.index_set = None
            # the superset is just the set itself
            self.superset.update(data)

        # proper init
        super().__init__(data)

    def is_indexed(self):
        """Check if the set is indexed, just here because pyomo has it."""
        return self.indexed

    def get_index_name(self):
        """Returns the index name if indexed."""
        return self.index_set

    def __repr__(self):
        """Return a string representation of the set."""
        return f"{super().__repr__()} indexed={self.indexed}"

    @overload
    def __getitem__(self, item: slice) -> "OrderedSet | ZenSet": ...

    @overload
    def __getitem__(self, item: Sequence[int]) -> List: ...

    @overload
    def __getitem__(self, item: int) -> object: ...

    @override
    def __getitem__(self, item):
        """Get an item from the set, if it is indexed.

        :param item: The item to retrieve
        :return: The item
        """
        if self.indexed:
            return self.ordered_data[item]
        else:
            return super().__getitem__(item)
