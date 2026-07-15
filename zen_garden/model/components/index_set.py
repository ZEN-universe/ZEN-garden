"""Class to prepare parameter data for pyomo parameter prerequisites."""

import logging
import uuid

import numpy as np
import xarray as xr

from zen_garden.model.components.component import Component
from zen_garden.model.components.zen_set import ZenSet

logger = logging.getLogger(__name__)


class IndexSet(Component):
    """Class to prepare parameter data for pyomo parameter prerequisites."""

    def __init__(self):
        """Initialization of the IndexSet object."""
        # base class init
        super().__init__()

        # attributes for the actual sets and index sets of the indexed sets
        self.sets: dict[str, ZenSet] = {}
        self.index_sets: dict[str, str] = {}

        # this is the Dataset with the coords
        self.coords_dataset = xr.Dataset()

    def add_set(self, name, data, doc, index_set: str | None = None):
        """Adds a set to the IndexSets (this set it not indexed).

        :param name: The name of the set
        :param data: The data used for the init
        :param doc: The docstring of the set
        :param index_set: The name of the index set if the set it self is indexed
        """
        if name in self.sets:
            logger.warning(f"{name} already added. Will be overwritten!")

        # added data and docs
        self.sets[name] = ZenSet(data=data, name=name, doc=doc, index_set=index_set)
        self.coords_dataset = self.coords_dataset.assign_coords(
            {name: np.array(list(self.sets[name].superset))}
        )
        self.docs[name] = self.compile_doc_string(
            doc, name=name, index_list=[index_set] if index_set is not None else []
        )
        if index_set is not None:
            self.index_sets[name] = index_set

    def is_indexed(self, name: str) -> bool:
        """Checks if the set with the name is indexed.

        :param name: The name of the set
        :return: True if indexed, False otherwise
        """
        return name in self.index_sets

    def get_index_name(self, name: str) -> str:
        """Returns the index name of an indexed set.

        :param name: The name of the indexed set
        :return: The name of the index set
        """
        if not self.is_indexed(name):
            raise ValueError(f"Set {name} is not an indexed set!")
        return self.index_sets[name]

    @staticmethod
    def tuple_to_arr(index_values, index_list, unique=False):
        """Transforms list of tuples into a list of xarrays with everything from tuple.

        :param index_values: The list of tuples with the index values
        :param index_list: The names of the indices, used in case of emtpy values
        :param unique: If True, the values are unique
        :return: A list of arrays
        """
        # if the list is empty
        if len(index_values) == 0:
            return tuple(xr.DataArray([]) for _ in index_list)

        # multiple indices
        if isinstance(index_values[0], tuple):
            # there might be more index names than tuple members
            ndims = len(index_values[0])
            tmp_vals = [[] for _ in range(ndims)]
            for t in index_values:
                for i in range(ndims):
                    tmp_vals[i].append(t[i])
            index_arrs = [xr.DataArray(t) for t in tmp_vals]
        else:
            index_arrs = [xr.DataArray(index_values)]

        # make unique
        if unique:
            index_arrs = [np.unique(t.data) for t in index_arrs]

        return tuple(index_arrs)

    def indices_to_mask(self, index_values, index_list, bounds, model=None):
        """Transforms a list of index values into a mask.

        :param index_values: A list of index values (tuples)
        :param index_list: The list of the names of the indices
        :param bounds: Either None, tuple, array or callable to define the bounds of
            the variable
        :param model: The model to which the mask belongs, note that indices which don't
            match existing indices are renamed to match the model
        :return: The mask as xarray
        """
        # get the coords
        index_arrs = IndexSet.tuple_to_arr(index_values, index_list)
        coords = [
            self.get_coord(data, name)
            for data, name in zip(index_arrs, index_list, strict=False)
        ]

        index_list, mask = self.create_variable_mask(
            coords, index_arrs, index_list, model
        )

        lower, upper = self.create_variable_bounds(
            bounds, coords, index_arrs, index_list, index_values
        )

        return mask, lower, upper

    def create_variable_bounds(
        self, bounds, coords, index_arrs, index_list, index_values
    ):
        """Creates the bounds for the variables.

        :param bounds: The bounds of the variable
        :param coords: The coordinates of the variable
        :param index_arrs: The index values as xarrays
        :param index_list: The list of the index names
        :param index_values: The list of the index values
        :return: The lower and upper bounds as xarrays
        """
        # get the bounds
        lower = xr.DataArray(-np.inf, coords=coords, dims=index_list)
        upper = xr.DataArray(np.inf, coords=coords, dims=index_list)
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
            tmp_low = []
            tmp_up = []
            for t in index_values:
                b = bounds(*t)
                tmp_low.append(b[0])
                tmp_up.append(b[1])
            lower.loc[index_arrs] = tmp_low
            upper.loc[index_arrs] = tmp_up
        elif bounds is None:
            lower = -np.inf
            upper = np.inf
        else:
            raise ValueError(
                f"bounds should be None, tuple, array or callable, is: {bounds}"
            )
        return lower, upper

    def create_variable_mask(self, coords, index_arrs, index_list, model):
        """Creates the mask for the variables.

        :param coords: The coordinates of the variable
        :param index_arrs: The index values as xarrays
        :param index_list: The list of the index names
        :param model: The model to which the mask belongs, note that indices which
            don't match existing indices are renamed to match the model
        :return: The mask as xarray
        """
        # save the index names under different names if they are empty
        if model is not None:
            index_names = []
            for index_name, coord in zip(index_list, coords, strict=False):
                # Check if there is already an index with same name but different size
                if coord.size == 0 and index_name in model.variables.coords:
                    index_names.append(index_name + f"_{uuid.uuid4()}")
                else:
                    index_names.append(index_name)
            index_list = index_names
        # init the mask
        mask = xr.DataArray(False, coords=coords, dims=index_list)
        mask.loc[index_arrs] = True
        return index_list, mask

    def get_coord(self, data, name):
        """Transforms data into a coordinate to avoid same name with different values.

        Transforms the data into a proper coordinate. If the name of the data is in
        a set, the sets superset is returned otherwise all unique data values are
        returned, this is to avoid having sets with the same name and different values.

        :param data: The data to transform
        :param name: The name of the set
        :return: The proper coordinate
        """
        if name in self and len(data) > 0:
            return self.coords_dataset.coords[name]
        else:
            return np.unique(data)

    def __getitem__(self, name) -> ZenSet:
        """Returns a set.

        :param name: The name of the set to get
        :return: The set that has the name
        """
        return self.sets[name]

    def __contains__(self, item):
        """The is for the "in" keyword.

        :param item: The item to check
        :return: True if item is contained, False otherwies
        """
        return item in self.sets

    def __iter__(self):
        """Returns an iterator over the sets.

        :return: The iterator
        """
        return iter(self.sets.values())
