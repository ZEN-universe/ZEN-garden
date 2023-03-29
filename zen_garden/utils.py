"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      October-2021
Authors:      Janis Fluri (janis.fluri@id.ethz.ch)
Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

Description:  Class is defining to read in the results of an Optimization problem.
==========================================================================================================================================================================="""

import sys
from collections import UserDict
from contextlib import contextmanager
from datetime import datetime
import linopy as lp
import xarray as xr

import h5py
import numpy as np
import pandas as pd
import yaml
from numpy import string_


def get_inheritors(klass):
    """
    Get all child classes of a given class
    :param klass: The class to get all children
    :return: All children as a set
    """

    subclasses = set()
    work = [klass]
    while work:
        parent = work.pop()
        for child in parent.__subclasses__():
            if child not in subclasses:
                subclasses.add(child)
                work.append(child)
    return subclasses

# This redirects output streams to files

class RedirectStdStreams(object):
    """
    A context manager that redirects the output to a file
    """

    def __init__(self, stdout=None, stderr=None):
        """
        Initializes the context manager
        :param stdout: Stream for stdout
        :param stderr: Stream for stderr
        """
        self._stdout = stdout or sys.stdout
        self._stderr = stderr or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush();
        self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        """
        The exit function of the context manager
        :param exc_type: Type of the exit
        :param exc_value: Value of the exit
        :param traceback:  traceback of the error
        """
        self._stdout.flush()
        self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr


class ZenIndex(object):
    """
    A multiindex class that can be easily used with xarray
    """

    def __init__(self, index_values, index_names=None):
        """
        Initialize the multiindex
        :param index_values: A list of index values as tuples
        :param index_names: Optional list of index names as strings, if the length does not match the tuple length,
                            it is ignored
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

    def get_unique(self, levels):
        """
        Returns a list of unique tuples across potentially multiple levels
        :param levels: A list of levels eithes by position or name
        :return: A list of tuples if multiple levels are given, otherwise a list of values
        """

        # empty index
        if self.index is None:
            return []

        return self.df.groupby(level=levels).first().index.to_list()

    def get_values(self, locs, levels, dtype=list):
        """
        Get all values of the levels over a given set of locations
        :param locs: A list of locs used for the "get_locs" method of the index
        :param levels: A single level or a list of levels to get the values for
        :param dtype: The dtype of the return value, either list or xr.DataArray
        :return: A single list or xr.DataArray if only one level is given, otherwise a list of lists or xr.DataArrays
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
        """
        The representation of the ZenIndex
        """

        # empty index
        if self.index is None:
            return "ZenIndex: Empty"

        return self.index.__repr__()


def linexpr_from_tuple_np(tuples, coords, model):
    """
    Transforms tuples of (coeff, var) into a linopy linear expression, but uses numpy broadcasting
    :param tuples: Tuple of (coeff, var)
    :param coords: The coordinates of the final linear expression
    :param model: The model to which the linear expression belongs
    :return: A linear expression
    """

    # get actual coords
    if not isinstance(coords, xr.core.dataarray.DataArrayCoordinates):
        coords = xr.DataArray(coords=coords).coords

    # numpy stack everything
    coefficients = []
    variables = []
    for coeff, var in tuples:
        var = var.labels.data
        if isinstance(coeff, (float, int)):
            coeff = np.full(var.shape, 1.0 * coeff)
        coefficients.append(coeff)
        variables.append(var)

    # to linear expression
    variables = xr.DataArray(np.stack(variables, axis=0), coords=coords, dims=["_term", *coords])
    coefficients = xr.DataArray(np.stack(coefficients, axis=0), coords=coords, dims=["_term", *coords])
    xr_ds = xr.Dataset({"coeffs": coefficients, "vars": variables}).transpose(..., "_term")

    return lp.LinearExpression(xr_ds, model)



# This is to lazy load h5 file most of it is taken from the hdfdict package
###########################################################################

TYPEID = '_type_'


@contextmanager
def hdf_file(hdf, lazy=True, *args, **kwargs):
    """
    Context manager yields h5 file if hdf is str,
    otherwise just yield hdf as is.
    """
    if isinstance(hdf, str):
        if not lazy:
            with h5py.File(hdf, *args, **kwargs) as hdf:
                yield hdf
        else:
            yield h5py.File(hdf, *args, **kwargs)
    else:
        yield hdf


def unpack_dataset(item):
    """
    Reconstruct a hdfdict dataset.
    Only some special unpacking for yaml and datetime types.
    :param item: h5py.Dataset
    :return: Unpacked Data
    """

    value = item[()]
    if TYPEID in item.attrs:
        if item.attrs[TYPEID].astype(str) == 'datetime':
            if hasattr(value, '__iter__'):
                value = [datetime.fromtimestamp(
                    ts) for ts in value]
            else:
                value = datetime.fromtimestamp(value)

        if item.attrs[TYPEID].astype(str) == 'yaml':
            value = yaml.safe_load(value.decode())

    # bytes to strings
    if isinstance(value, bytes):
        value = value.decode("utf-8")

    return value


class LazyHdfDict(UserDict):
    """
    Helps loading data only if values from the dict are requested.
    This is done by reimplementing the __getitem__ method.
    """

    def __init__(self, _h5file=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._h5file = _h5file  # used to close the file on deletion.

    def __getitem__(self, key):
        """Returns item and loads dataset if needed."""
        item = super().__getitem__(key)
        if isinstance(item, h5py.Dataset):
            item = unpack_dataset(item)
            self.__setitem__(key, item)
        return item

    def unlazy(self, return_dict=False):
        """
        Unpacks all datasets.
        You can call dict(this_instance) then to get a real dict.
        """
        load(self, lazy=False)

        # Load loads all the data but we need to transform the lazydict into normal dicts
        def _recursive(lazy_dict):
            for k in list(lazy_dict.keys()):
                if isinstance(lazy_dict[k], LazyHdfDict):
                    _recursive(lazy_dict[k])
                    lazy_dict[k] = dict(lazy_dict[k])

        _recursive(self)

        if return_dict:
            return dict(self)

    def close(self):
        """
        Closes the h5file if provided at initialization.
        """
        if self._h5file and hasattr(self._h5file, 'close'):
            self._h5file.close()

    def __del__(self):
        self.close()

    def _ipython_key_completions_(self):
        """
        Returns a tuple of keys.
        Special Method for ipython to get key completion
        """
        return tuple(self.keys())


def fill_dict(hdfobject, datadict, lazy=True, unpacker=unpack_dataset):
    """
    Recursivley unpacks a hdf object into a dict
    :param hdfobject: Object to recursively unpack
    :param datadict: A dict option to add the unpacked values to
    :param lazy: If True, the datasets are lazy loaded at the moment an item is requested.
    :param unpacker: Unpack function gets `value` of type h5py.Dataset. Must return the data you would like to
                     have it in the returned dict.
    :return: a dict
    """

    for key, value in hdfobject.items():
        if type(value) == h5py.Group or isinstance(value, LazyHdfDict):
            if lazy:
                datadict[key] = LazyHdfDict()
            else:
                datadict[key] = {}
            datadict[key] = fill_dict(value, datadict[key], lazy, unpacker)
        elif isinstance(value, h5py.Dataset):
            if not lazy:
                value = unpacker(value)
            datadict[key] = value

    return datadict

def load(hdf, lazy=True, unpacker=unpack_dataset, *args, **kwargs):
    """
    Returns a dictionary containing the groups as keys and the datasets as values from given hdf file.
    :param hdf: string (path to file) or `h5py.File()` or `h5py.Group()`
    :param lazy: If True, the datasets are lazy loaded at the moment an item is requested.
    :param unpacker: Unpack function gets `value` of type h5py.Dataset. Must return the data you would like to
                     have it in the returned dict.
    :param args: Additional arguments for the hdf_file handler
    :param kwargs: Additional keyword arguments for the hdf_file handler
    :return: The dictionary containing all groupnames as keys and datasets as values.
    """

    with hdf_file(hdf, lazy=lazy, *args, **kwargs) as hdf:
        if lazy:
            data = LazyHdfDict(_h5file=hdf)
        else:
            data = {}
        return fill_dict(hdf, data, lazy=lazy, unpacker=unpacker)


def pack_dataset(hdfobject, key, value):
    """
    Packs a given key value pair into a dataset in the given hdfobject.
    """

    isdt = None
    if isinstance(value, datetime):
        value = value.timestamp()
        isdt = True

    if hasattr(value, '__iter__'):
        if all(isinstance(i, datetime) for i in value):
            value = [item.timestamp() for item in value]
            isdt = True

    try:
        ds = hdfobject.create_dataset(name=key, data=value)
        if isdt:
            ds.attrs.create(
                name=TYPEID,
                data=string_("datetime"))
    except TypeError:
        # Obviously the data was not serializable. To give it
        # a last try; serialize it to yaml
        # and save it to the hdf file:
        ds = hdfobject.create_dataset(
            name=key,
            data=string_(yaml.safe_dump(value))
        )
        ds.attrs.create(
            name=TYPEID,
            data=string_("yaml"))
        # if this fails again, restructure your data!


def dump(data, hdf, packer=pack_dataset, *args, **kwargs):
    """
    Adds keys of given dict as groups and values as datasets to the given hdf-file (by string or object) or group object.
    :param data: The dictionary containing only string keys and data values or dicts again.
    :param hdf: string (path to file) or `h5py.File()` or `h5py.Group()`
    :param packer: Callable gets `hdfobject, key, value` as input.
                   `hdfobject` is considered to be either a h5py.File or a h5py.Group.
                   `key` is the name of the dataset.
                   `value` is the dataset to be packed and accepted by h5py.
    :param args: Additional arguments for the hdf_file handler
    :param kwargs: Additional keyword arguments for the hdf_file handler
    :return: `h5py.Group()` or `h5py.File()` instance
    """

    def _recurse(datadict, hdfobject):
        for key, value in datadict.items():
            if isinstance(key, tuple):
                key = '_'.join((str(i) for i in key))
            if isinstance(value, (dict, LazyHdfDict)):
                hdfgroup = hdfobject.create_group(key)
                _recurse(value, hdfgroup)
            else:
                packer(hdfobject, key, value)

    with hdf_file(hdf, *args, **kwargs) as hdf:
        _recurse(data, hdf)
        return hdf