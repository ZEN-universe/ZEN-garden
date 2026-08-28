"""Class to define parameters for the optimization model."""

import copy

import numpy as np
import pandas as pd
import xarray as xr

from zen_garden.model.registries.base import Registry
from zen_garden.model.registries.set_registry import SetRegistry
from zen_garden.model.zen_set import BaseSet


class DictParameter(object):
    """This is a helper class to store the dictionary parameters."""

    def add_param(self, name, data):
        """Add a parameter.

        :param name: The name of the param
        :param data: The data of the param
        """
        setattr(self, name, data)

    def __getattr__(self, name: str):
        """Fallback for dynamically-added parameters (see add_parameter).

        Only called when normal attribute lookup fails, so this covers
        parameters set via setattr(self, name, xr_data).
        """
        raise AttributeError(
            f"Parameter '{name}' does not exist. Did you forget to define it?"
        )


class ParameterRegistry(Registry):
    def __init__(self, sets: SetRegistry):
        """Initialization of the parameter object."""
        self.sets = sets
        super().__init__()
        self.min_parameter_value = {"name": None, "value": None}
        self.max_parameter_value = {"name": None, "value": None}
        self.dict_parameters = DictParameter()
        self.units: dict[str, pd.Series | str | None] = {}

    def add_parameter(
        self,
        name: str,
        doc: str,
        data: BaseSet | tuple | list | xr.DataArray,
        dict_of_units: dict | None = None,
    ):
        """Initialization of a parameter.

        :param name: name of parameter
        :param doc: docstring of parameter
        :param data: non default data of parameter and index_names
        :param dict_of_units: units of parameter
        """
        if dict_of_units is None:
            dict_of_units = {}

        if name in self.docs.keys():
            raise ValueError(f"Parameter {name} already added. Can only be added once")

        index_values, index_list = self.get_index_names_data(data)
        # save if highest or lowest value
        self.save_min_max(index_values, name)
        # convert to arr and dict
        xr_data = self.convert_to_xarr(copy.copy(index_values), index_list)
        dict_data = self.convert_to_dict(index_values)
        # set parameter
        setattr(self, name, xr_data)
        self.dict_parameters.add_param(name, dict_data)

        # save additional parameters
        self.docs[name] = self.compile_doc_string(doc, index_list, name)
        # save parameter units
        self.units[name] = self.get_param_units(
            index_values, dict_of_units, index_list, name
        )

    def _ensure_pd_series_multi_index(self, component_data):
        """Convert pd.Series index to pd.MultiIndex.

        :param component_data: extracted data as pd.Series
        :return: component_data: extracted data as pd.Series with MultiIndex
        """
        if isinstance(component_data, pd.Series) and not isinstance(
            component_data.index, pd.MultiIndex
        ):
            component_data.index = pd.MultiIndex.from_product(
                [component_data.index.to_list()]
            )
        return component_data

    def add_helper_parameter(self, name, data):
        """Adding a helper parameter that is not added to the docs and results.

        Adds a helper param. Note that this param is not added to the docs and therefore
         not saved in the results. Also, the data is taken as is and is not transformed.

        :param name: The name of the param
        :param data: The data
        """
        # set parameter
        setattr(self, name, data)

    def save_min_max(self, data, name):
        """Stores min and max parameter.

        :param data: non default data of parameter and index_names
        :param name: name of parameter
        """
        if isinstance(data, dict) and data:
            data = pd.Series(data)
        if isinstance(data, pd.Series):
            if not pd.api.types.is_numeric_dtype(data):
                return
            abs_val = data.abs()
            abs_val = abs_val[(abs_val != 0) & abs_val.notna() & np.isfinite(abs_val)]
            if abs_val.empty:
                return

            if isinstance(abs_val.index, pd.MultiIndex):
                idxmax = (
                    name
                    + "_"
                    + "_".join(map(str, abs_val.index[abs_val.argmax(skipna=True)]))
                )
                idxmin = (
                    name
                    + "_"
                    + "_".join(map(str, abs_val.index[abs_val.argmin(skipna=True)]))
                )
            else:
                idxmax = f"{name}_{abs_val.index[abs_val.argmax(skipna=True)]}"
                idxmin = f"{name}_{abs_val.index[abs_val.argmin(skipna=True)]}"
            valmax = abs_val.max()
            valmin = abs_val.min()

        else:
            if not data or (abs(data) == 0) or (abs(data) == np.inf):
                return
            abs_val = abs(data)
            idxmax = name
            valmax = abs_val
            idxmin = name
            valmin = abs_val
        if not self.max_parameter_value["name"]:
            self.max_parameter_value["name"] = idxmax
            self.max_parameter_value["value"] = valmax
            self.min_parameter_value["name"] = idxmin
            self.min_parameter_value["value"] = valmin
        else:
            if valmax > self.max_parameter_value["value"]:
                self.max_parameter_value["name"] = idxmax
                self.max_parameter_value["value"] = valmax
            if valmin < self.min_parameter_value["value"]:
                self.min_parameter_value["name"] = idxmin
                self.min_parameter_value["value"] = valmin

    @staticmethod
    def get_param_units(data, dict_of_units, index_list, name):
        """Creates series of units with identical multi-index as data has.

        :param data: non default data of parameter and index_names
        :param dict_of_units: units of parameter
        :param index_list: list of index names
        """
        if dict_of_units:
            if not isinstance(data, pd.Series):
                return str(dict_of_units["unit_in_base_units"].units)
            else:
                unit_series = pd.Series(index=data.index, dtype=str)
                unit_series = unit_series.rename_axis(index=index_list)
                unit_series = unit_series.sort_index()
                if "unit_in_base_units" in dict_of_units:
                    unit_series[:] = str(dict_of_units["unit_in_base_units"].units)
                    return unit_series
            for key, value in dict_of_units.items():
                unit_series.loc[pd.IndexSlice[key]] = str(value)
            return unit_series

    @staticmethod
    def convert_to_dict(data):
        """Converts the data to a dict if pd.Series.

        :param data: non default data of parameter and index_names
        :return data: data as dict
        """
        if isinstance(data, pd.Series):
            # if single entry in index
            if len(data.index[0]) == 1:
                data.index = pd.Index(sum(data.index.values, ()))
            data = data.to_dict()
        return data

    def convert_to_xarr(self, data, index_list):
        """Converts the data to a dict if pd.Series.

        :param data: non default data of parameter and index_names
        :param index_list: list of indices
        :return data: data as xarray
        """
        if isinstance(data, pd.Series):
            # if single entry in index
            if len(data.index[0]) == 1:
                data.index = pd.Index(sum(data.index.values, ()))
            if len(data.index.names) == len(index_list):
                data.index.names = index_list
            # transform the type of the coords to str if necessary
            data = data.to_xarray().astype(float)

            # objects to string
            coords_dict = {}
            for k, v in data.coords.dtypes.items():
                if v.hasobject:
                    coords_dict[k] = data.coords[k].astype(str)
                else:
                    coords_dict[k] = data.coords[k]
            data = data.assign_coords(coords_dict)

            # now we need to align the coords TODO try to speed up
            data, _ = xr.align(data, self.sets.coords_dataset, join="right")

        # sometimes we get empty parameters
        if isinstance(data, dict) and len(data) == 0:
            data = xr.DataArray([])
        return data

    def __getattr__(self, name) -> xr.DataArray:
        """Fallback for dynamically-added parameters (see add_parameter).

        Only called when normal attribute lookup fails, so this covers
        parameters set via setattr(self, name, xr_data).
        """
        if getattr(self.dict_parameters, name, None) is None:
            raise AttributeError(
                f"Parameter '{name}' does not exist. Did you forget to define it?"
            )
        return getattr(self.dict_parameters, name)
