import numpy as np
import pandas as pd
import xarray as xr

from zen_garden.model.components.index_set import IndexSet


class ZenIndex(object):
    """A multiindex class that can be easily used with xarray."""

    def __init__(self, index_values, index_names=None):
        """Initialize the multiindex.

        :param index_values: A list of index values as tuples
        :param index_names: Optional list of index names as strings, if the length does
                            not match the tuple length, it is ignored
        """
        # if there are no values, we create a dummy index
        if len(index_values) == 0:
            self.index = None
            self.df = None
        else:
            # set the index
            self.index = pd.MultiIndex.from_tuples(index_values)
            if index_names is not None and len(self.index.names) == len(index_names):
                self.index.names = index_names

            # dummy dataframe
            self.df = pd.Series(np.ones(len(self.index)), index=self.index)

    def get_unique(self, levels, as_array=False):
        """Returns a list of unique tuples across potentially multiple levels.

        :param levels: A list of levels eithes by position or name
        :param as_array: If True, the return value a list of xarrays
        :return: A list of tuples if given multiple levels, otherwise a list of values
        """
        # empty index
        if self.index is None:
            return []

        # get the values
        assert self.df is not None
        vals = self.df.groupby(level=levels).first().index.to_list()

        if as_array:
            return IndexSet.tuple_to_arr(vals, levels)

        return vals

    def get_values(self, locs, levels, dtype=list, unique=False):
        """Get all values of the levels over a given set of locations.

        :param locs: A list of locs used for the "get_locs" method of the index
        :param levels: A single level or a list of levels to get the values for
        :param dtype: The dtype of the return value, either list or xr.DataArray
        :param unique: If True, only unique values are returned
        :return: A single list or xr.DataArray if only one level is given, otherwise
            a list of lists or xr.DataArrays
        """
        # empty index
        if self.index is None:
            return []

        # handle single or multiple input
        if isinstance(levels, list):
            return_list = True
        else:
            levels = [levels]
            return_list = False

        # clycle
        vals = []
        for level in levels:
            indices = self.index.get_locs(locs)
            val = self.index[indices].get_level_values(level)
            if unique:
                val = val.unique()
            if dtype is list:
                vals.append(val.to_list())
            elif dtype is xr.DataArray:
                vals.append(xr.DataArray(val.values, dims=val.name))
            else:
                raise ValueError("dytpe should be list or xr.DataArray got {dtype}")

        if return_list:
            return vals

        return vals[0]

    def __repr__(self):
        """The representation of the ZenIndex."""
        # empty index
        if self.index is None:
            return "ZenIndex: Empty"

        return self.index.__repr__()
