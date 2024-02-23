"""
:Title:          ZEN-GARDEN
:Created:        July-2022
:Authors:        Jacob Mannhardt (jmannhardt@ethz.ch)
:Organization:   Laboratory of Reliability and Risk Engineering, ETH Zurich

Class containing parameters. This is a proxy for pyomo parameters, since the construction of parameters has a significant overhead.
"""
import copy
import itertools
import logging
import uuid
from itertools import combinations
from itertools import zip_longest

import linopy as lp
import numpy as np
import pandas as pd
import pint
import xarray as xr
from ordered_set import OrderedSet

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

    def get_unique(self, levels, as_array=False):
        """
        Returns a list of unique tuples across potentially multiple levels
        :param levels: A list of levels eithes by position or name
        :param as_array: If True, the return value a list of xarrays
        :return: A list of tuples if multiple levels are given, otherwise a list of values
        """

        # empty index
        if self.index is None:
            return []

        # get the values
        vals = self.df.groupby(level=levels).first().index.to_list()

        if as_array:
            return IndexSet.tuple_to_arr(vals, levels)

        return vals

    def get_values(self, locs, levels, dtype=list, unique=False):
        """
        Get all values of the levels over a given set of locations
        :param locs: A list of locs used for the "get_locs" method of the index
        :param levels: A single level or a list of levels to get the values for
        :param dtype: The dtype of the return value, either list or xr.DataArray
        :param unique: If True, only unique values are returned
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
        """
        The representation of the ZenIndex
        """

        # empty index
        if self.index is None:
            return "ZenIndex: Empty"

        return self.index.__repr__()


class ZenSet(OrderedSet):
    """
    Similiar to pyomo.Set
    """

    def __init__(self, data, name="", doc="", index_set="UnnamedIndex"):
        """
        Initialize the set
        :param data: The data of the set, either an iterable or a dictionary for an indexed set
        :param name: The name of the set
        :param doc: The corresponding docstring
        :param index_set: The name of the index set
        """
        # set attributes
        self.data = data
        self.name = name
        self.doc = doc
        self.superset = OrderedSet()

        if isinstance(data, dict):
            # init the children
            self.ordered_data = {k: ZenSet(v, name=f"{name}[{k}]") for k, v in data.items()}

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
        """
        Check if the set is indexed, just here because pyomo has it
        """
        return self.indexed

    def get_index_name(self):
        """
        Returns the index name if indexed
        """
        return self.index_set

    def __repr__(self):
        """
        Return a string representation of the set
        """
        return f"{super().__repr__()} indexed={self.indexed}"

    def __getitem__(self, item):
        """
        Get an item from the set, if it is indexed
        :param item: The item to retrieve
        :return: The item
        """
        if self.indexed:
            return self.ordered_data[item]
        else:
            return super().__getitem__(item)

class Component:
    """
    Class to prepare parameter, variable and constraint data such that it suits the pyomo prerequisites
    """
    def __init__(self):
        """
        instantiate object of Component class
        """
        self.docs = {}

    @staticmethod
    def compile_doc_string(doc, index_list, name, domain=None):
        """ compile docstring from doc and index_list

        :param doc: docstring to be compiled
        :param index_list: list of indices
        :param name: name of parameter/variable/constraint
        :param domain: domain of parameter/variable/constraint (e.g., reals, non negative reals, ...)
        :return complete_doc: complete docstring composed of name, doc and dims
        """
        assert type(doc) == str, f"Docstring {doc} has wrong format. Must be 'str' but is '{type(doc).__name__}'"
        # check for prohibited strings
        prohibited_strings = [",", ";", ":", "/", "name", "doc", "dims", "domain"]
        original_doc = copy.copy(doc)
        for string in prohibited_strings:
            if string in doc:
                logging.warning(f"Docstring '{original_doc}' contains prohibited string '{string}'. Occurrences are dropped.")
                doc = doc.replace(string, "")
        # joined index names
        joined_index = ",".join(index_list)
        # complete doc string
        complete_doc = f"name:{name};doc:{doc};dims:{joined_index}"
        if domain:
            complete_doc += f";domain:{domain}"
        return complete_doc

    @staticmethod
    def get_index_names_data(index_list):
        """ splits index_list in data and index names

        :param index_list: list of indices (names and values)
        :return index_values: names of indices
        :return index_names:  values of indices
        """
        if isinstance(index_list, ZenSet):
            index_values = list(index_list)
            index_names = [index_list.name]
        elif isinstance(index_list, tuple):
            index_values, index_names = index_list
        elif isinstance(index_list, list):
            index_values = list(itertools.product(*index_list[0]))
            index_names = index_list[1]
        elif isinstance(index_list, xr.DataArray):
            index_values = index_list.to_series().dropna()
            index_names = list(index_list.coords.dims)
        else:
            raise TypeError(f"Type {type(index_list)} unknown to extract index names.")
        return index_values, index_names


class IndexSet(Component):
    """
        Class to prepare parameter data for pyomo parameter prerequisites
    """
    def __init__(self):
        """ initialization of the IndexSet object """
        # base class init
        super().__init__()

        # attributes for the actual sets and index sets of the indexed sets
        self.sets = {}
        self.index_sets = {}

        # this is the Dataset with the coords
        self.coords_dataset = xr.Dataset()

    def add_set(self, name, data, doc, index_set=None):
        """
        Adds a set to the IndexSets (this set it not indexed)
        :param data: The data used for the init
        :param doc: The docstring of the set
        :param index_set: The name of the index set if the set it self is indexed
        """

        if name in self.sets:
            logging.warning(f"{name} already added. Will be overwritten!")

        # added data and docs
        self.sets[name] = ZenSet(data=data, name=name, doc=doc, index_set=index_set)
        self.coords_dataset = self.coords_dataset.assign_coords({name: np.array(list(self.sets[name].superset))})
        self.docs[name] = self.compile_doc_string(doc,name=name, index_list= [index_set] if index_set is not None else [])
        if index_set is not None:
            self.index_sets[name] = index_set

    def is_indexed(self, name):
        """
        Checks if the set with the name is indexed, convenience method for ZenSet["name"].is_indexed()
        :param name: The name of the set
        :return: True if indexed, False otherwise
        """

        return name in self.index_sets

    def get_index_name(self, name):
        """
        Returns the index name of an indexed set, convenience method for ZenSet["name"].get_index_name()
        :param name: The name of the indexed set
        :return: The name of the index set
        """

        if not self.is_indexed(name=name):
            raise ValueError(f"Set {name} is not an indexed set!")
        return self.index_sets[name]

    @staticmethod
    def tuple_to_arr(index_values, index_list, unique=False):
        """
        Transforms a list of tuples into a list of xarrays containing all elements from the corresponding tuple entry
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
        """
        Transforms a list of index values into a mask
        :param index_values: A list of index values (tuples)
        :param index_list: The list of the names of the indices
        :param bounds: Either None, tuple, array or callable to define the bounds of the variable
        :param model: The model to which the mask belongs, note that indices which don't match existing indices are
                      renamed to match the model
        :return: The mask as xarray
        """

        # get the coords
        index_arrs = IndexSet.tuple_to_arr(index_values, index_list)
        coords = [self.get_coord(data, name) for data, name in zip(index_arrs, index_list)]

        # init the mask
        if model is not None:
            index_names = []
            for index_name, coord in zip(index_list, coords):
                # we check if there is already an index with the same name but a different size
                if index_name in model.variables.coords and coord.size == 0:
                    index_names.append(index_name + f"_{uuid.uuid4()}")
                else:
                    index_names.append(index_name)
            index_list = index_names

        mask = xr.DataArray(False, coords=coords, dims=index_list)
        mask.loc[index_arrs] = True

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
            lower.loc[index_arrs] = bounds[:,0]
            upper.loc[index_arrs] = bounds[:,1]
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
            raise ValueError(f"bounds should be None, tuple, array or callable, is: {bounds}")

        return mask, lower, upper

    def get_coord(self, data, name):
        """
        Transforms the data into a proper coordinate. If the name of the data is in a set, the sets superset is
        returned otherwise all unique data values are returned, this is to avoid having sets with the same name
        and different values
        :param data: The data to transform
        :param name: The name of the set
        :return: The proper coordinate
        """

        if name in self and len(data) > 0:
            return self.coords_dataset.coords[name]
        else:
            return np.unique(data)

    def __getitem__(self, name):
        """
        Returns a set
        :param name: The name of the set to get
        :return: The set that has the name
        """

        return self.sets[name]

    def __contains__(self, item):
        """
        The is for the "in" keyword
        :param item: The item to check
        :return: True if item is contained, False otherwies
        """

        return item in self.sets

    def __iter__(self):
        """
        Returns an iterator over the sets
        :return: The iterator
        """

        return iter(self.sets.values())


class DictParameter(object):
    """
    This is a helper class to store the dictionary parameters
    """

    def add_param(self, name, data):
        """
        Add a parameter
        :param name: The name of the param
        :param data: The data of the param
        """

        setattr(self, name, data)


class Parameter(Component):
    def __init__(self, optimization_setup):
        """ initialization of the parameter object """
        self.optimization_setup = optimization_setup
        self.index_sets = optimization_setup.sets
        super().__init__()
        self.min_parameter_value = {"name": None, "value": None}
        self.max_parameter_value = {"name": None, "value": None}
        self.dict_parameters = DictParameter()
        self.units = {}

    def add_parameter(self, name, doc, data=None, calling_class=None, index_names=None, set_time_steps=None, capacity_types=False):
        """ initialization of a parameter

        :param name: name of parameter
        :param doc: docstring of parameter
        :param data: non default data of parameter and index_names
        :param calling_class: class type of the object add_parameter is called for
        :param index_names: names of index sets, only if calling_class is not EnergySystem
        :param set_time_steps: time steps, only if calling_class is EnergySystem
        :param capacity_types: boolean if extracted for capacities
        """

        dict_of_units = {}
        if name == "capex_specific_conversion":
            component_data, index_list, dict_of_units = calling_class.get_capex_all_elements(optimization_setup=self.optimization_setup, index_names=index_names)
            data = component_data, index_list
        elif data is None:
            component_data, index_list, dict_of_units = self.optimization_setup.initialize_component(calling_class, name, index_names, set_time_steps, capacity_types)
            data = component_data, index_list
        if name not in self.docs.keys():
            data, index_list = self.get_index_names_data(data)
            # save if highest or lowest value
            self.save_min_max(data, name)
            # convert to arr and dict
            xr_data = self.convert_to_xarr(copy.copy(data), index_list)
            dict_data = self.convert_to_dict(data)
            # set parameter
            setattr(self, name, xr_data)
            self.dict_parameters.add_param(name, dict_data)

            # save additional parameters
            self.docs[name] = self.compile_doc_string(doc, index_list, name)
            # save parameter units
            self.units[name] = self.get_param_units(data, dict_of_units, index_list, name)
        else:
            logging.warning(f"Parameter {name} already added. Can only be added once")

    def add_helper_parameter(self, name, data):
        """Adds a helper param. Note that this param won't be added to the docs and therefore not saved in the results.
        Also, the data is taken as is and is not transformed
        :param name: The name of the param
        :param data: The data
        """

        # set parameter
        setattr(self, name, data)

    def save_min_max(self, data, name):
        """ stores min and max parameter

        :param data: non default data of parameter and index_names
        :param name: name of parameter
        """
        if isinstance(data, dict) and data:
            data = pd.Series(data)
        if isinstance(data, pd.Series):
            abs_val = data.abs()
            abs_val = abs_val[(abs_val != 0) & (abs_val != np.inf)]
            if not abs_val.empty and not abs_val.isna().all():
                if isinstance(abs_val.index,pd.MultiIndex):
                    idxmax = name + "_" + "_".join(map(str, abs_val.index[abs_val.argmax(skipna=True)]))
                    idxmin = name + "_" + "_".join(map(str, abs_val.index[abs_val.argmin(skipna=True)]))
                else:
                    idxmax = f"{name}_{abs_val.index[abs_val.argmax(skipna=True)]}"
                    idxmin = f"{name}_{abs_val.index[abs_val.argmin(skipna=True)]}"
                valmax = abs_val.max()
                valmin = abs_val.min()
            else:
                return
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
        """ creates series of units with identical multi-index as data has

        :param data: non default data of parameter and index_names
        :param dict_of_units: units of parameter
        :param index_list: list of index names
        """
        if dict_of_units:
            if not isinstance(data, pd.Series):
                return str(dict_of_units["unit_in_base_units"].units)
            else:
                unit_series = pd.Series(index=data.index, dtype=str)
                if name == "capex_specific_conversion":
                    index_list = [index_list[0]] + index_list[2:]
                unit_series = unit_series.rename_axis(index=index_list)
                unit_series = unit_series.sort_index()
                if "unit_in_base_units" in dict_of_units:
                    unit_series[:] = dict_of_units["unit_in_base_units"].units
                    return unit_series.astype(str)
            for key, value in dict_of_units.items():
                unit_series.loc[pd.IndexSlice[key]] = value
            return unit_series.astype(str)

    @staticmethod
    def convert_to_dict(data):
        """ converts the data to a dict if pd.Series

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
        """ converts the data to a dict if pd.Series

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

            # now we need to align the coords
            data, _ = xr.align(data, self.index_sets.coords_dataset, join="right")

        # sometimes we get empty parameters
        if isinstance(data, dict) and len(data) == 0:
            data = xr.DataArray([])
        return data


class Variable(Component):
    def __init__(self, optimization_setup):
        """
        Initialization of a variable
        :param index_sets: A reference to the index sets of the model
        """
        self.optimization_setup = optimization_setup
        self.index_sets = optimization_setup.sets
        self.unit_handling = optimization_setup.energy_system.unit_handling
        self.units = {}
        super().__init__()

    def add_variable(self, model: lp.Model, name, index_sets, unit_category, integer=False, binary=False, bounds=None, doc="", mask=None):
        """ initialization of a variable

        :param model: parent block component of variable, must be linopy model
        :param name: name of variable
        :param index_sets: Tuple of index values and index names
        :param unit_category: dict defining the dimensionality of the variable's unit
        :param integer: If it is an integer variable
        :param binary: If it is a binary variable
        :param bounds:  bounds of variable
        :param doc: docstring of variable
        :param mask: mask of variable
        """

        if name not in self.docs.keys():
            index_values, index_list = self.get_index_names_data(index_sets)
            mask_index, lower, upper = self.index_sets.indices_to_mask(index_values, index_list, bounds, model)
            if mask is not None:
                mask_index = mask_index & mask
            model.add_variables(lower=lower, upper=upper, integer=integer, binary=binary, name=name, mask=mask_index, coords=mask_index.coords)

            # save variable doc
            if integer:
                domain = "Integers"
            elif binary:
                domain = "Binary"
            else:
                if isinstance(bounds, tuple) and isinstance(bounds[0], xr.DataArray):
                    domain = "BoundedReals"
                elif isinstance(bounds, tuple) and bounds[0] == 0:
                    domain = "NonNegativeReals"
                elif callable(bounds) or isinstance(bounds, np.ndarray):
                    domain = "BoundedReals"
                else:
                    domain = "Reals"
            self.docs[name] = self.compile_doc_string(doc, index_list, name, domain)
            self.units[name] = self.get_var_units(unit_category, index_values, index_list)
        else:
            logging.warning(f"Variable {name} already added. Can only be added once")

    def get_var_units(self, unit_category, var_index_values, index_list):
        """
         creates series of units with identical multi-index as variable has

        :param unit_category: dict defining the dimensionality of the variable's unit
        :param var_index_values: list of variable index values
        :param index_list: list of index names
        :return: series of variable units
        """
        # binary variables
        if not unit_category:
            return
        if all(isinstance(item, tuple) for item in var_index_values):
            index = pd.MultiIndex.from_tuples(var_index_values, names=index_list)
        else:
            index = pd.Index(var_index_values)
        unit = self.unit_handling.ureg("dimensionless")
        distinct_dims = {"money": "[currency]", "distance": "[length]", "time": "[time]", "emissions": "[mass]"}
        for dim, dim_name in distinct_dims.items():
            if dim in unit_category:
                dim_unit = [key for key, value in self.unit_handling.base_units.items() if value == dim_name][0]
                unit = unit * self.unit_handling.ureg(dim_unit)**unit_category[dim]
        var_units = pd.Series(index=index,dtype=str)
        # variable can have different units
        if "energy_quantity" in unit_category:
            # energy_quantity depends on carrier index level (e.g. flow_import)
            if any("carrier" in carrier_name for carrier_name in var_units.index.names):
                carrier_level = [level for level in var_units.index.names if "carrier" in level][0]
                for carrier, energy_quantity in self.unit_handling.carrier_energy_quantities.items():
                    carrier_idx = var_units.index.get_level_values(carrier_level) == carrier
                    var_units[carrier_idx] = (unit * energy_quantity ** unit_category["energy_quantity"]).units
            # energy_quantity depends on technology index level (e.g. capacity)
            else:
                tech_level = [level for level in var_units.index.names if "technologies" in level][0]
                for technology in self.optimization_setup.dict_elements["Technology"]:
                    reference_carrier = technology.reference_carrier[0]
                    energy_quantity = [energy_quantity for carrier, energy_quantity in self.unit_handling.carrier_energy_quantities.items() if carrier == reference_carrier][0]
                    tech_idx = var_units.index.get_level_values(tech_level) == technology.name
                    var_units[tech_idx] = (unit * energy_quantity ** unit_category["energy_quantity"]).units
                if "set_capacity_types" in var_units.index.names:
                    energy_idx = var_units.index.get_level_values("set_capacity_types") == "energy"
                    var_units[energy_idx] = var_units[energy_idx].apply(lambda u: (u*self.unit_handling.ureg("hour")).units)
        # variable has constant unit
        else:
            var_units[:] = unit.units
        return var_units.astype(str)

class Constraint(Component):
    def __init__(self, index_sets):
        """
        Initialization of a constraint
        :param index_sets: A reference to the index sets of the model
        """

        self.index_sets = index_sets
        super().__init__()
        # This is the big-M for the constraints if the variables inside an expression are not bounded
        self.M = np.iinfo(np.int32).max

    def add_constraint_block(self, model: lp.Model, name, constraint, doc="", disjunction_var=None):
        """ initialization of a constraint block (list of constraints)
        :param model: The linopy model
        :param name: name of variable
        :param constraint: The constraint to add
        :param doc: docstring of variable
        :param disjunction_var: An optional binary variable. The constraints will only be enforced if this variable is
                                True
        """

        # convert to list
        use_suffix = True
        if not isinstance(constraint, list):
            constraint = [constraint]
            use_suffix = False

        for num, cons in enumerate(constraint):
            current_name = f"{name}"
            if use_suffix:
                current_name += f"_{num}"
            if current_name not in self.docs.keys():
                # if the cons is a tuple we have a mask
                if isinstance(cons, tuple):
                    cons, mask = cons
                else:
                    mask = None

                # drop all unnecessary dimensions
                # FIXME: we do this via data to avoid the drop deprecation warning, update once lp implements drop_vars
                # lhs = cons.lhs.drop(list(set(cons.lhs.coords) - set(cons.lhs.dims)))
                lhs = lp.LinearExpression(cons.lhs.data.drop_vars(list(set(cons.lhs.coords) - set(cons.lhs.dims))), model)
                # add constraint
                self._add_con(model, current_name, lhs, cons.sign, cons.rhs, disjunction_var=disjunction_var, mask=mask)
                # save constraint doc
                index_list = list(cons.coords.dims)
                self.docs[name] = self.compile_doc_string(doc, index_list, current_name)
            else:
                logging.warning(f"{name} already added. Can only be added once")

    def add_constraint_rule(self, model: lp.Model, name, index_sets, rule, doc="", disjunction_var=None):
        """ initialization of a variable
        :param model: The linopy model
        :param name: name of variable
        :param index_sets: indices and sets by which the variable is indexed
        :param rule: constraint rule
        :param disjunction_var: An optional binary variable. The constraints will only be enforced if this variable is
                                True
        :param doc: docstring of variable"""


        if name not in self.docs.keys():
            index_values, index_list = self.get_index_names_data(index_sets)

            # if the list of values is emtpy, there is nothing to add
            if len(index_values) == 0:
                return

            # save constraint doc
            self.docs[name] = self.compile_doc_string(doc, index_list, name)

            # eval the rule
            xr_lhs, xr_sign, xr_rhs = self.rule_to_cons(model=model, rule=rule, index_values=index_values, index_list=index_list)
            self._add_con(model, name, xr_lhs, xr_sign, xr_rhs, disjunction_var=disjunction_var)
        else:
            logging.warning(f"{name} already added. Can only be added once")

    def _get_M(self, model, lin_expr):
        """
        Calculates a conservative bound (max abs value) of the given linear expression
        :param model: The model of the linear expression
        :param lin_expr: The linear expression
        :return: An array with the same shape as vars containing the bounds
        """

        # extract
        vars = lin_expr.vars.data
        coeffs = lin_expr.coeffs.data

        # get the shape of the vars
        shape = vars.shape

        # get all the coords
        bounds = []
        for coeff, var in zip(coeffs.ravel(), vars.ravel()):
            # if dummy, continue
            if var == -1:
                bounds.append(0)
                continue

            # get the name and coord
            var_name, var_coord = model.variables.get_label_position(var)
            lower = model.variables[var_name].lower.loc[var_coord].item()
            upper = model.variables[var_name].upper.loc[var_coord].item()

            # conservative bound
            bounds.append(np.abs(coeff) * np.maximum(np.abs(lower), np.abs(upper)))

        # sum over the _term dim and set coords
        return xr.DataArray(np.sum(np.array(bounds).reshape(shape) + 1, axis=-1),
                            coords=[lin_expr.vars.coords[d] for d in lin_expr.vars.dims[:-1]])

    def _add_con(self, model, name, lhs, sign, rhs, disjunction_var=None, mask=None):
        """ Adds a constraint to the model
        :param model: The linopy model
        :param name: name of the constraint
        :param lhs: left hand side of the constraint
        :param sign: sign of the constraint
        :param rhs: right hand side of the constraint
        :param disjunction_var: An optional binary variable. The constraints will only be enforced if this variable is
                                True
        :param mask: An optional mask to only add the constraint for certain indices
        """

        # get the mask, where rhs is not nan and rhs is finite
        if mask is not None:
            mask = ~np.isnan(rhs) & np.isfinite(rhs) & mask
        else:
            mask = ~np.isnan(rhs) & np.isfinite(rhs)

        if disjunction_var is not None:
            # get the bounds
            bounds = self._get_M(model, lhs)
            if not np.isfinite(bounds).all():
                logging.warning(f"Constraint {name} has infinite bounds. Can not extract big-M resorting to default")
                bounds = self.M

            # if we have any equal cons, we need to transform them into <= and >=
            if (sign == "=").any():
                # the "<=" cons
                sign_c = sign.where(sign != "=", "<=")
                m_arr = xr.zeros_like(rhs).where(sign_c != "<=", bounds).where(sign_c != ">=", -bounds)
                model.add_constraints(lhs + m_arr * disjunction_var, sign_c, rhs + m_arr, name=name + "<=", mask=mask)
                sign_c = sign.where(sign != "=", ">=")
                m_arr = xr.zeros_like(rhs).where(sign_c != "<=", bounds).where(sign_c != ">=", -bounds)
                model.add_constraints(lhs + m_arr * disjunction_var, sign_c, rhs + m_arr, name=name + ">=", mask=mask)
            # create the arr
            else:
                m_arr = xr.zeros_like(rhs).where(sign != "<=", bounds).where(sign != ">=", -bounds)
                model.add_constraints(lhs + m_arr * disjunction_var, sign, rhs + m_arr, name=name + ">=", mask=mask)
        else:
            model.add_constraints(lhs, sign, rhs, name=name, mask=mask)

    def rule_to_cons(self, model, rule, index_values, index_list, cons=None):
        """
        Evaluates the rule on the index_values
        :param model: The linopy model
        :param rule: The rule to call
        :param index_values: A list of index_values to evaluate the rule
        :param index_list: a list of index names
        :param cons: A list of constraints, if the rule was already evaluated, if provided, the rule will not be called
        :return: xarrays of the lhs, sign and the rhs
        """

        # create the mask
        index_arrs = IndexSet.tuple_to_arr(index_values, index_list)
        coords = [self.index_sets.get_coord(data, name) for data, name in zip(index_arrs, index_list)]

        # there might be an extra label
        if len(index_list) > 1 and len(index_list) != len(index_values[0]):
            index_list = [f"dim_{i}" for i in range(len(index_values[0]))]
        coords = xr.DataArray(coords=coords, dims=index_list).coords
        shape = tuple(map(len, coords.values()))

        # if we only have a single index, there is no need to unpack
        if cons is None:
            if len(index_list) == 1:
                cons = [rule(arg) for arg in index_values]
            else:
                cons = [rule(*arg) for arg in index_values]

        # catch Nones
        placeholder_lhs = lp.expressions.ScalarLinearExpression((np.nan,), (-1,), model)
        emtpy_cons = lp.constraints.AnonymousScalarConstraint(placeholder_lhs, "=", np.nan)
        cons = [c if c is not None else emtpy_cons for c in cons]

        # low level magic
        exprs = [con.lhs for con in cons]
        # complicated expressions might have been initialized with loc arrays
        coeffs = np.array(tuple(zip_longest(*(e.coeffs.data if isinstance(e.coeffs, xr.DataArray) else e.coeffs for e in exprs), fillvalue=np.nan)))
        vars = np.array(tuple(zip_longest(*(e.vars.data if isinstance(e.vars, xr.DataArray) else e.vars for e in exprs), fillvalue=-1)))

        nterm = vars.shape[0]
        coeffs = coeffs.reshape((nterm, -1))
        vars = vars.reshape((nterm, -1))

        xr_coeffs = xr.DataArray(np.full(shape=(nterm,) + shape, fill_value=np.nan), coords, dims=("_term", *coords))
        xr_coeffs.loc[(slice(None), ) + index_arrs] = coeffs
        xr_vars = xr.DataArray(np.full(shape=(nterm,) + shape, fill_value=-1), coords, dims=("_term", *coords))
        xr_vars.loc[(slice(None), ) + index_arrs] = vars
        xr_ds = xr.Dataset({"coeffs": xr_coeffs, "vars": xr_vars}).transpose(..., "_term")
        xr_lhs = lp.LinearExpression(xr_ds, model)
        xr_sign = xr.DataArray("=", coords, dims=index_list).astype("U2")
        xr_sign.loc[index_arrs] = [c.sign.data if isinstance(c.sign, xr.DataArray) else c.sign for c in cons]
        xr_rhs = xr.DataArray(0.0, coords, dims=index_list)
        # Here we catch infinities in the constraints (gurobi does not care but glpk does)
        rhs_vals = np.array([c.rhs.data if isinstance(c.rhs, xr.DataArray) else c.rhs for c in cons])
        xr_rhs.loc[index_arrs] = rhs_vals

        return xr_lhs, xr_sign, xr_rhs

    def add_pw_constraint(self, model, name, index_values, yvar, xvar, break_points, f_vals, cons_type="EQ"):
        """
        Adds a piece-wise linear constraint of the type f(x) = y for each index in the index_values, where f is defined
        by the breakpoints and f_vals (x_1, y_1), ..., (x_n, y_n)
        Note that these method will create helper variables in form of a S0S2, sources:
         https://support.gurobi.com/hc/en-us/articles/360013421331-How-do-I-model-piecewise-linear-functions-
         https://medium.com/bcggamma/hands-on-modeling-non-linearity-in-linear-optimization-problems-f9da34c23c9a
        :param model: The model to add the constraints to
        :param name: The name of the constraint
        :param index_values: A list of index values that will be used to build the constraints
        :param yvar: The name of the yvar, a variable compatible with the index values used for y
        :param xvar: The name of the xvar, a variable compatible with the index values used for x
        :param break_points: A mapping index -> list that provides the breakpoints for each index
        :param f_vals: A mapping index -> list that provides the function values for each index
        :param cons_type: Type of the constraint (currently only EQ supported)
        """

        if cons_type != "EQ":
            raise NotImplementedError("Currently only EQ constraints are supported")

        # get the variables
        xvar = model.variables[xvar]
        yvar = model.variables[yvar]

        # cycle through all indices
        for num, index_val in enumerate(index_values):
            # extract everyting
            x = xvar[index_val]
            y = yvar[index_val]
            br = break_points[index_val]
            fv = f_vals[index_val]
            if len(br) != len(fv):
                raise ValueError("Number of break points should be equal to number of function values for each "
                                 "index value.")

            # create sos vars, assure same coords
            sos2_vars = self._get_nonnegative_sos2_vars(model, len(br))
            br = xr.DataArray(br, coords=sos2_vars.coords)
            fv = xr.DataArray(fv, coords=sos2_vars.coords)

            # add the constraints, give it a valid name
            model.add_constraints(x.to_linexpr() - (br * sos2_vars).sum() == 0, name=f"{name}_br_{num}")
            model.add_constraints(y.to_linexpr() - (fv * sos2_vars).sum() == 0, name=f"{name}_fv_{num}")

    def _get_nonnegative_sos2_vars(self, model, n):
        """
        Creates a list of continues nonnegative variables in an SOS2
        :param model: The model to add the variables
        :param n: The number of variables to create
        :return: A list of variables that are SOS2 constrained
        """

        # vars and binaries, we need to take care of all the annoying dimension names
        dim_name = f"sos2_dim_{uuid.uuid1()}"
        sos2_var = model.add_variables(lower=np.zeros(n), binary=False, name=f"sos2_var_{uuid.uuid1()}", coords=(xr.DataArray(np.arange(n), dims=dim_name), ))
        sos2_var_bin = model.add_variables(binary=True, name=f"sos2_var_bin_{uuid.uuid1()}", coords=(xr.DataArray(np.arange(n), dims=dim_name), ))

        # add the constraints
        model.add_constraints(sos2_var.sum() == 1.0)
        model.add_constraints(sos2_var - sos2_var_bin <= 0.0)
        model.add_constraints(sos2_var_bin.sum() <= 2.0)
        combi_index = xr.DataArray([c for c in combinations(np.arange(n), 2) if c[0] + 1 != c[1]], dims=[dim_name, "combi_dim"])
        model.add_constraints(sos2_var_bin.sel({dim_name: combi_index[:, 0]}).rename({dim_name: f"{dim_name}_1"})
                              + sos2_var_bin.sel({dim_name: combi_index[:, 1]}).rename({dim_name: f"{dim_name}_1"})
                              <= 1.0)

        return sos2_var

    def remove_constraint(self, model, name):
        """
        Removes a constraint from the model
        :param model: The model to remove the constraint from
        :param name: The name of the constraint
        """

        # remove all constraints and sub-constraints from the model and docs
        for cname in list(model.constraints):
            if cname.startswith(name):
                model.constraints.remove(cname)
        for cname in list(self.docs.keys()):
            if cname.startswith(name):
                del self.docs[cname]

    @staticmethod
    def combine_constraints(constraints, stack_dim, model):
        """
        Combines a list of constraints into a single constraint
        :param constraints: A list of constraints
        :param stack_dim: The name of the stack dimension
        :param model: The model to add the constraints to
        :return: A single constraint
        """

        # catch empty constraints
        if len(constraints) == 0:
            return constraints

        # get the shape of the constraints
        max_terms = max([c.lhs.shape[-1] for c in constraints])
        c = constraints[0]
        lhs_shape = c.lhs.shape[:-1] + (max_terms, )
        coords = [xr.DataArray(np.arange(len(constraints)), dims=[stack_dim]), ] + [c.lhs.coords[d] for d in c.lhs.dims][:-1] + [xr.DataArray(np.arange(max_terms), dims=["_term"])]
        coeffs = xr.DataArray(np.full((len(constraints), ) + lhs_shape, fill_value=np.nan), coords=coords,
                              dims=(stack_dim, *constraints[0].lhs.dims.keys()))
        variables = xr.DataArray(np.full((len(constraints), ) + lhs_shape, fill_value=-1), coords=coords,
                                 dims=(stack_dim, *constraints[0].lhs.dims.keys()))
        sign = xr.DataArray("=", coords=coords[:-1]).astype("U2")
        rhs = xr.DataArray(np.nan, coords=coords[:-1])

        for num, con in enumerate(constraints):
            terms = con.lhs.dims["_term"]
            coeffs[num, ..., :terms] = con.lhs.coeffs.data
            variables[num, ..., :terms] = con.lhs.vars.data
            sign[num, ...] = con.sign
            rhs[num, ...] = con.rhs
            # make sure all dims are in the right order and deal with subsets
            sign[num, ...] = con.sign
            rhs[num, ...] = rhs[num].where(~con.rhs.isnull(), con.rhs + rhs[num])

        xr_ds = xr.Dataset({"coeffs": coeffs, "vars": variables})
        lhs = lp.LinearExpression(xr_ds, model)

        return lp.constraints.AnonymousConstraint(lhs, sign, rhs)

    def reorder_group(self, lhs, sign, rhs, index_values, index_names, model, drop=None):
        """
        Reorders the constraints in a group to have full shape according to index values and names
        :param lhs: The lhs of the constraints
        :param sign: The sign of the constraints, can be None if only lhs should be restructured
        :param rhs: The rhs of the constraints, can be None if only lhs should be restructured
        :param index_values: The index values corresponding to the group numbers
        :param index_names: The index names of the indices
        :param model: The model
        :param drop: Which group to drop (the dummy group
        :return: An anonymous constraint
        """

        # drop if necessary
        lhs = lhs.data
        if drop is not None:
            lhs = lhs.drop_sel(group=drop, errors="ignore")
            rhs = rhs.drop_sel(group=drop, errors="ignore")
            sign = sign.drop_sel(group=drop, errors="ignore")

        # drop the unncessessary dimensions
        lhs = lhs.drop_vars(list(set(lhs.coords) - set(lhs.dims)))

        # get the coordinates
        index_arrs = IndexSet.tuple_to_arr(index_values, index_names)
        coords = {name: np.unique(arr.data) for name, arr in zip(index_names, index_arrs)}
        coords.update({cname: lhs.coords[cname] for cname in lhs.coords if cname != "group" and cname != "_term"})
        coords_shape = tuple(len(c) for c in coords.values())
        dims = index_names + list(lhs.dims)
        dims.remove("group")

        # create the full arrays, note that the lhs needs a _term dimension
        xr_coeffs = xr.DataArray(np.full(shape=coords_shape + (lhs.coeffs.shape[-1], ), fill_value=np.nan), dims=dims, coords=coords)
        xr_vars = xr.DataArray(np.full(shape=coords_shape + (lhs.vars.shape[-1], ), fill_value=-1), dims=dims, coords=coords)

        # rhs and sign do not have a _term dimension
        xr_rhs = xr.DataArray(np.full(shape=coords_shape, fill_value=np.nan), dims=dims[:-1], coords=coords)
        xr_sign = xr.DataArray(np.full(shape=coords_shape, fill_value="="), dims=dims[:-1], coords=coords).astype("U2")

        # Assign everything
        for num, index_val in enumerate(index_values):
            if num in lhs.coords["group"]:
                xr_coeffs.loc[index_val] = lhs.coeffs.sel(group=num).data
                xr_vars.loc[index_val] = lhs.vars.sel(group=num).data
                if rhs is not None:
                    xr_rhs.loc[index_val] = rhs.sel(group=num).data
                if sign is not None:
                    xr_sign.loc[index_val] = sign.sel(group=num).data

        # to full arrays
        xr_lhs = lp.LinearExpression(xr.Dataset({"coeffs": xr_coeffs, "vars": xr_vars}), model)

        if rhs is None and sign is None:
            return xr_lhs

        return lp.constraints.AnonymousConstraint(xr_lhs, xr_sign, xr_rhs)

    def reorder_list(self, constraints, index_values, index_names, model):
        """
        Reorders a list of constraints to full shape according to index values and names
        :param constraints: A list of constraints to reorder
        :param index_values: List of index values corresponding to the group numbers
        :param index_names: List of index names of the indices
        :param model: The model
        :return: A single constraint with the correct dimensions
        """

        # catch empty constraints
        if len(constraints) == 0:
            return []

        # combine constraints to a group
        combined_constraints = self.combine_constraints(constraints, "group", model)

        # reorder the group
        reordered = self.reorder_group(combined_constraints.lhs, combined_constraints.sign, combined_constraints.rhs, index_values, index_names, model)
        return reordered

    def align_constraint(self, constraint, mask=None):
        """
        Aligns a single constraint the coordinates
        :param constraint: The constraint to align
        :param mask: mask of constraint
        :return: The aligned constraint and mask
        """

        # we start with the lhs
        vars, _ = xr.align(constraint.lhs.data.vars, self.index_sets.coords_dataset, join="right", fill_value=-1)
        coeffs, _ = xr.align(constraint.lhs.data.coeffs, self.index_sets.coords_dataset, join="right", fill_value=np.nan)
        lhs = lp.LinearExpression(xr.Dataset({"coeffs": coeffs, "vars": vars}), constraint.lhs.model)

        # now the rhs
        rhs, _ = xr.align(constraint.rhs, self.index_sets.coords_dataset, join="right", fill_value=np.nan)

        # sign
        sign, _ = xr.align(constraint.sign, self.index_sets.coords_dataset, join="right", fill_value="=")

        # mask
        if mask is not None:
            mask, _ = xr.align(mask, self.index_sets.coords_dataset, join="right", fill_value=False)

        return lp.constraints.AnonymousConstraint(lhs, sign, rhs), mask

    def return_contraints(self, constraints, model=None, mask=None, index_values=None, index_names=None,
                          stack_dim_name=None):
        """
        This is a high-level function that returns the constraints in the correct format, i.e. with reordering, masks,
        etc.
        :param constraints: A single constraints or a potentially empty list of constraints.
        :param model: The model to which the constraints belong
        :param mask: A mask with the same shape as the constraints
        :param index_values: The index values corresponding to the group numbers, if reorder is necessary
        :param index_names: The names of the indices, if reorder is necessary
        :param stack_dim_name: If a list of constraints is provided along with index_values and index_names, the
                               constraints are reordered with the provided indices, if a stack_dim_name is provided, the
                               constraints are stacked along a single dimension with the provided name.
        :return: Constraints with can be added
        """

        # nothing to do, this is the skip of a rule constraint
        if constraints is None:
            return constraints

        # no need to do anything special
        if not isinstance(constraints, list):
            # rule based constraints
            if isinstance(constraints, lp.constraints.AnonymousScalarConstraint):
                return constraints

            # align
            constraints, mask = self.align_constraint(constraints, mask)
            if mask is None:
                return constraints
            return constraints, mask

        # if there are no constraints, return an empty list
        if len(constraints) == 0:
            return []
        elif model is None:
            raise ValueError("If constraints is a list, model must be provided!")

        # normal reordering
        if index_names is not None and index_values is not None:
            constraints = self.reorder_list(constraints, index_values, index_names, model)

            # align
            constraints, mask = self.align_constraint(constraints, mask)
            if mask is None:
                return constraints
            return constraints, mask

        # stack along a dimension
        if stack_dim_name is not None:
            constraints = self.combine_constraints(constraints, stack_dim_name, model)

            # align
            constraints, mask = self.align_constraint(constraints, mask)
            if mask is None:
                return constraints
            return constraints, mask

        # Error
        raise ValueError("Either single constraint or a list with index_values and index_names or stack_dim_name must be provided!")
