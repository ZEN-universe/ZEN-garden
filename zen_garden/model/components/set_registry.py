"""Class to prepare parameter data for pyomo parameter prerequisites."""

import itertools
import logging
import uuid
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

from zen_garden.model.components.component import Component
from zen_garden.model.components.zen_set import BaseSet, IndexedSet, SimpleSet

if TYPE_CHECKING:
    from zen_garden.elements.element import Element
    from zen_garden.model.config import Config
    from zen_garden.services.element_registry import ElementRegistry

logger = logging.getLogger(__name__)


class SetRegistry(Component):
    """Class to prepare parameter data for pyomo parameter prerequisites.
    Formerly known as IndexSet."""

    def __init__(
        self,
        config: "Config",
        element_registry: "ElementRegistry",
        indexing_sets: list[str],
    ):
        """Initialization of the SetRegistry object."""
        # base class init
        super().__init__()

        self.config = config
        self.element_registry = element_registry
        self.indexing_sets = indexing_sets

        # attributes for the actual sets and index sets of the indexed sets
        self.sets: dict[str, BaseSet] = {}
        self.index_sets: dict[str, str] = {}

        # this is the Dataset with the coords
        self.coords_dataset = xr.Dataset()

    def add_set(self, name, data, doc, index_set: str | None = None):
        """Adds a set to the SetRegistry (this set it not indexed).

        :param name: The name of the set
        :param data: The data used for the init
        :param doc: The docstring of the set
        :param index_set: The name of the index set if the set it self is indexed
        """
        if name in self.sets:
            logger.warning(f"{name} already added. Will be overwritten!")

        # added data and docs
        if isinstance(data, Mapping):
            model_set: BaseSet = IndexedSet(
                data=data, name=name, doc=doc, index_set=index_set
            )
        else:
            model_set = SimpleSet(data=data, name=name, doc=doc)
        self.sets[name] = model_set
        self.coords_dataset = self.coords_dataset.assign_coords(
            {name: np.array(list(model_set.coordinate_values))}
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
        return isinstance(self.sets[name], IndexedSet)

    def get_index_name(self, name: str) -> str:
        """Returns the index name of an indexed set.

        :param name: The name of the indexed set
        :return: The name of the index set
        """
        if not self.is_indexed(name):
            raise ValueError(f"Set {name} is not an indexed set!")
        zen_set = self.sets[name]
        if not isinstance(zen_set, IndexedSet):
            raise ValueError(f"Set {name} is not an indexed set!")
        return zen_set.index_set

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
        index_arrs = SetRegistry.tuple_to_arr(index_values, index_list)
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

    def __getitem__(self, name) -> BaseSet:
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

    def create_custom_set(self, list_index: list[str], element_class: "type[Element]"):
        """Creates custom set for model component.

        :param list_index: list of names of indices
        :param element_class: class of the element to get attributes from
        :return: list_index: list of names of indices
        """
        list_index = list(list_index)  # make a copy of the list to avoid side effects

        # Case 1: all index sets are already defined in model and no set is indexed
        if all(
            index in self.sets and not self.is_indexed(index) for index in list_index
        ):
            base_sets = [self.sets[index] for index in list_index if index in self.sets]
            # return indices as cartesian product of sets
            base_custom_set: list[Any] = (
                list(itertools.product(*base_sets))
                if len(base_sets) > 1
                else list(base_sets[0])
            )
            return base_custom_set, list_index

        if list_index[0] not in self.indexing_sets:
            raise NotImplementedError(
                f"Index <{list_index[0]}> is not in the indexing sets."
            )

        # Case 2: first index is indexed, build custom set based on first index
        custom_set: list[Any] = []
        for element in self.sets[list_index[0]]:
            append_element = True
            list_sets: list[Any] = []

            for index in list_index[1:]:
                # if the set already exist in model
                if index in self.sets:
                    append = self._handle_existing_set(index, element, list_sets)
                    if not append:
                        raise NotImplementedError(
                            f"Index <{index}> is not known in sets."
                        )
                    continue

                # if index is set_location
                if index == "set_location":
                    self._handle_set_location_index(element, list_sets)
                    continue

                # if set is used to determine if on-off behavior is modeled
                # exclude technologies which have no min_load
                if "on_off" in index:
                    append_element = self._append_on_off_modeled(
                        element, index, element_class
                    )
                    continue

                # split in capacity types of power and energy
                if index == "set_capacity_types":
                    self._handle_set_capacity_types_index(element, list_sets)
                    continue

                raise NotImplementedError(f"Index <{index}> not known")

            # append indices to custom_set if element is supposed to be appended
            if append_element:
                if list_sets:
                    custom_set.extend(list(itertools.product([element], *list_sets)))
                else:
                    custom_set.extend([element])
        return custom_set, list_index

    def _handle_existing_set(self, index: str, element: str, list_sets: list[Any]):
        """Handles existing sets in the model.
        Returns True if handled, False if unknown.

        :param index: index to handle
        :param element: element to handle
        :param sets: sets of the optimization setup
        :param list_sets: list of sets to append
        """
        if not self.is_indexed(index):
            list_sets.append(self.sets[index])
            return True
        elif self.get_index_name(index) in self.sets:
            indexed_set = self.sets[index]
            if not isinstance(indexed_set, IndexedSet):
                raise TypeError(f"Set {index} is not indexed")
            list_sets.append(indexed_set[element])
            return True
        return False

    def _append_on_off_modeled(
        self, element: str, index: str, element_class: "type[Element]"
    ) -> bool:
        """Checks if the on-off-behavior (min-load) of a technology needs to be modeled.

        :param element: technology in model
        :param index: index to check
        :return model_on_off: Bool indicating if on-off-behavior needs to be modeled
        """
        model_on_off = self._check_on_off_modeled(element, element_class)
        return not (("set_no_on_off" in index and model_on_off) or (not model_on_off))

    def _handle_set_location_index(self, element: str, list_sets: list[Any]):
        """Handles the set_location index for the custom set.

        :param element: element to handle
        :param sets: sets of the optimization setup
        :param list_sets: list of sets to append
        """
        if (
            element in self.sets["set_conversion_technologies"]
            or element in self.sets["set_storage_technologies"]
            or element in self.sets["set_retrofitting_technologies"]
        ):
            list_sets.append(self.sets["set_nodes"])
        elif element in self.sets["set_transport_technologies"]:
            list_sets.append(self.sets["set_edges"])

    def _handle_set_capacity_types_index(self, element: str, list_sets: list[Any]):
        """Handles the set_capacity_types index for the custom set.

        :param element: element to handle
        :param sets: sets of the optimization setup
        :param list_sets: list of sets to append
        """
        if element in self.sets["set_storage_technologies"]:
            list_sets.append(self.config.system.set_capacity_types)
        else:
            list_sets.append([self.config.system.set_capacity_types[0]])

    def _check_on_off_modeled(self, tech: str, element_class: "type[Element]"):
        """Classmethod checks if on-off-behavior of a technology needs to be modeled.

        If the technology has a minimum load of 0 for all nodes and time steps, and all
        dependent carriers have a lower bound of 0 (only for conversion technologies
        modeled as pwa), then on-off-behavior is not necessary to model.

        :param tech: technology in model
        :return model_on_off: Bool indicating if on-off-behaviour needs to be modeled
        """
        # check if any min load
        unique_min_load = list(
            set(
                self.element_registry.get_attribute_of_specific_element(
                    element_class, tech, "min_load"
                ).values
            )
        )
        # disable if only one unique min_load which is zero
        return not (len(unique_min_load) == 1 and unique_min_load[0] == 0)
