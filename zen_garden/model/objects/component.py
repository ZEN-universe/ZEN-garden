"""===========================================================================================================================================================================
Title:          ZEN-GARDEN
Created:        July-2022
Authors:        Jacob Mannhardt (jmannhardt@ethz.ch)
Organization:   Laboratory of Reliability and Risk Engineering, ETH Zurich

Description:    Class containing parameters. This is a proxy for pyomo parameters, since the construction of parameters has a significant overhead.
==========================================================================================================================================================================="""
import copy
import itertools
import logging
import uuid
from itertools import combinations
from itertools import zip_longest
import time

import linopy as lp
import numpy as np
import pandas as pd
import xarray as xr
from ordered_set import OrderedSet

logging.basicConfig(level=logging.DEBUG)

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

        if isinstance(data, dict):
            # init the children
            self.ordered_data = {k: ZenSet(v, name=f"{name}[{k}]") for k, v in data.items()}

            # for an indexed sets the init data are the keys
            data = data.keys()
            self.indexed = True
            self.index_set = index_set

        else:
            self.indexed = False
            # index set it None
            self.index_set = None

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

    def __init__(self):
        self.docs = {}

    @staticmethod
    def compile_doc_string(doc, index_list, name, domain=None):
        """ compile docstring from doc and index_list"""
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
        """ splits index_list in data and index names """
        if isinstance(index_list, ZenSet):
            index_values = list(index_list)
            index_names = [index_list.name]
        elif isinstance(index_list, tuple):
            index_values, index_names = index_list
        elif isinstance(index_list, list):
            index_values = list(itertools.product(*index_list[0]))
            index_names = index_list[1]
        else:
            raise TypeError(f"Type {type(index_list)} unknown to extract index names.")
        return index_values, index_names


class IndexSet(Component):
    def __init__(self):
        """ initialization of the IndexSet object """
        # base class init
        super().__init__()

        # attributes for the actual sets and index sets of the indexed sets
        self.sets = {}
        self.index_sets = {}

    def add_set(self, name, data, doc, index_set=None):
        """
        Adds a set to the IndexSets (this set it not indexed)
        :param data: The data used for the init
        :param doc: The docstring of the set
        :param index_set: The name of the index set if the set itself is indexed
        """

        if name in self.sets:
            logging.warning(f"{name} already added. Will be overwritten!")

        # added data and docs
        self.sets[name] = ZenSet(data=data, name=name, doc=doc, index_set=index_set)
        self.docs[name] = doc
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
    def tuple_to_arr(index_values, index_list):
        """
        Transforms a list of tuples into a list of xarrays containing all elements from the corresponding tuple entry
        :param index_values: The list of tuples with the index values
        :param index_list: The names of the indices, used in case of emtpy values
        :return: A list of arrays
        """

        # if the list is empty
        if len(index_values) == 0:
            return [xr.DataArray([]) for _ in index_list]

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

        return index_arrs

    @staticmethod
    def indices_to_mask(index_values, index_list, bounds, model=None):
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
        coords = [np.unique(t.data) for t in index_arrs]

        # init the mask
        if model is not None:
            index_names = []
            for index_name, coord in zip(index_list, coords):
                # we check if there is already an index with the same name but a different size
                if index_name in model.variables.coords and coord.size != model.variables.coords[index_name].data.size:
                    index_names.append(index_name + f"_{uuid.uuid4()}")
                else:
                    index_names.append(index_name)
            index_list = index_names

        mask = xr.DataArray(False, coords=coords, dims=index_list)
        mask.loc[*index_arrs] = True

        # get the bounds
        lower = xr.DataArray(-np.inf, coords=coords, dims=index_list)
        upper = xr.DataArray(np.inf, coords=coords, dims=index_list)
        if isinstance(bounds, tuple):
            lower[...] = bounds[0]
            upper[...] = bounds[1]
        elif isinstance(bounds, np.ndarray):
            lower.loc[*index_arrs] = bounds[:,0]
            upper.loc[*index_arrs] = bounds[:,1]
        elif callable(bounds):
            tmp_low = []
            tmp_up = []
            for t in index_values:
                b = bounds(*t)
                tmp_low.append(b[0])
                tmp_up.append(b[1])
            lower.loc[*index_arrs] = tmp_low
            upper.loc[*index_arrs] = tmp_up
        elif bounds is None:
            lower = -np.inf
            upper = np.inf
        else:
            raise ValueError(f"bounds should be None, tuple, array or callable, is: {bounds}")

        return mask, lower, upper

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


class Parameter(Component):
    def __init__(self):
        """ initialization of the parameter object """
        super().__init__()
        self.min_parameter_value = {"name": None, "value": None}
        self.max_parameter_value = {"name": None, "value": None}

    def add_parameter(self, name, data, doc):
        """ initialization of a parameter
        :param name: name of parameter
        :param data: non default data of parameter and index_names
        :param doc: docstring of parameter """

        if name not in self.docs.keys():
            data, index_list = self.get_index_names_data(data)
            # save if highest or lowest value
            self.save_min_max(data, name)
            # convert to arr
            data = self.convert_to_xarr(data, index_list)
            # set parameter
            setattr(self, name, data)

            # save additional parameters
            self.docs[name] = self.compile_doc_string(doc, index_list, name)
        else:
            logging.warning(f"Parameter {name} already added. Can only be added once")

    def save_min_max(self, data, name):
        """ stores min and max parameter """
        if isinstance(data, dict) and data:
            data = pd.Series(data)
        if isinstance(data, pd.Series):
            _abs = data.abs()
            _abs = _abs[(_abs != 0) & (_abs != np.inf)]
            if not _abs.empty:
                _idxmax = name + "_" + "_".join(map(str, _abs.index[_abs.argmax()]))
                _valmax = _abs.max()
                _idxmin = name + "_" + "_".join(map(str, _abs.index[_abs.argmin()]))
                _valmin = _abs.min()
            else:
                return
        else:
            if not data or (abs(data) == 0) or (abs(data) == np.inf):
                return
            _abs = abs(data)
            _idxmax = name
            _valmax = _abs
            _idxmin = name
            _valmin = _abs
        if not self.max_parameter_value["name"]:
            self.max_parameter_value["name"] = _idxmax
            self.max_parameter_value["value"] = _valmax
            self.min_parameter_value["name"] = _idxmin
            self.min_parameter_value["value"] = _valmin
        else:
            if _valmax > self.max_parameter_value["value"]:
                self.max_parameter_value["name"] = _idxmax
                self.max_parameter_value["value"] = _valmax
            if _valmin < self.min_parameter_value["value"]:
                self.min_parameter_value["name"] = _idxmin
                self.min_parameter_value["value"] = _valmin

    @staticmethod
    def convert_to_xarr(data, index_list):
        """ converts the data to a dict if pd.Series"""
        if isinstance(data, pd.Series):
            # if single entry in index
            if len(data.index[0]) == 1:
                data.index = pd.Index(sum(data.index.values, ()))
            if len(data.index.names) == len(index_list):
                data.index.names = index_list
            # transform the type of the coords to str if necessary
            data = data.to_xarray()
            coords_dict = {}
            for k, v in data.coords.dtypes.items():
                if v.hasobject:
                    coords_dict[k] = data.coords[k].astype(str)
                else:
                    coords_dict[k] = data.coords[k]
            data = data.assign_coords(coords_dict)
        return data

    def as_xarray(self, pname, indices):
        """
        Returns a xarray of the param
        :param pname: The name of the param
        :param indices: A list of indices to extract
        :return: An xarray of the param
        """

        p = getattr(self, pname)
        if isinstance(indices[0], tuple):
            return xr.DataArray([p[*args] for args in indices])
        else:
            return xr.DataArray([p[i] for i in indices])

    def __getitem__(self, item):
        """
        The get item method to directly access the underlying dataset
        :param item: The item to retireve
        :return: The xarray paramter
        """

        return self.data_set[item]


class Variable(Component):
    def __init__(self):
        super().__init__()

    def add_variable(self, model: lp.Model, name, index_sets, integer=False, binary=False, bounds=None, doc=""):
        """ initialization of a variable
        :param model: parent block component of variable, must be linopy model
        :param name: name of variable
        :param index_sets: Tuple of index values and index names
        :param integer: If it is an integer variable
        :param binary: If it is a binary variable
        :param bounds:  bounds of variable
        :param doc: docstring of variable """

        if name not in self.docs.keys():
            index_values, index_list = self.get_index_names_data(index_sets)
            mask, lower, upper = IndexSet.indices_to_mask(index_values, index_list, bounds, model)
            model.add_variables(lower=lower, upper=upper, integer=integer, binary=binary, name=name, mask=mask, coords=mask.coords)

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
        else:
            logging.warning(f"Variable {name} already added. Can only be added once")


class Constraint(Component):
    def __init__(self):
        super().__init__()
        # This is the big-M for the constraints
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

        t0 = time.perf_counter()

        # convert to list
        if not isinstance(constraint, list):
            constraint = [constraint]

        for num, cons in enumerate(constraint):
            current_name = f"{name}_{num}"
            if current_name not in self.docs.keys():
                # drop all unnecessary dimensions
                lhs = cons.lhs.drop(list(set(cons.lhs.coords) - set(cons.lhs.dims)))
                # add constraint
                self._add_con(model, current_name, lhs, cons.sign, cons.rhs, disjunction_var=disjunction_var)
                # save constraint doc
                index_list = list(cons.coords.dims)
                self.docs[name] = self.compile_doc_string(doc, index_list, current_name)
            else:
                logging.warning(f"{name} already added. Can only be added once")

        t1 = time.perf_counter()
        logging.debug(f"Adding constraint block {name} took {t1 - t0:0.4f} seconds")

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
            t0 = time.perf_counter()
            xr_lhs, xr_sign, xr_rhs = self.rule_to_cons(model=model, rule=rule, index_values=index_values, index_list=index_list)
            t1 = time.perf_counter()
            logging.debug(f"Evaluating constraint rule {name} took {t1 - t0:0.4f} seconds")
            self._add_con(model, name, xr_lhs, xr_sign, xr_rhs, disjunction_var=disjunction_var)
            t2 = time.perf_counter()
            logging.debug(f"Adding constraint rule {name} took {t2 - t1:0.4f} seconds")

        else:
            logging.warning(f"{name} already added. Can only be added once")

    def _add_con(self, model, name, lhs, sign, rhs, disjunction_var=None):
        """ Adds a constraint to the model
        :param model: The linopy model
        :param name: name of the constraint
        :param lhs: left hand side of the constraint
        :param sign: sign of the constraint
        :param rhs: right hand side of the constraint
        :param disjunction_var: An optional binary variable. The constraints will only be enforced if this variable is
                                True
        """

        # get the mask, where rhs is not nan and rhs is finite
        mask = ~np.isnan(rhs) & np.isfinite(rhs)

        if disjunction_var is not None:
            # if we have any equal cons, we need to transform them into <= and >=
            if (sign == "=").any():
                # the "<=" cons
                sign_c = sign.where(sign != "=", "<=")
                m_arr = xr.zeros_like(rhs).where(sign_c != "<=", self.M).where(sign_c != ">=", -self.M)
                model.add_constraints(lhs + m_arr * disjunction_var, sign_c, rhs + m_arr, name=name + "<=", mask=mask)
                sign_c = sign.where(sign != "=", ">=")
                m_arr = xr.zeros_like(rhs).where(sign_c != "<=", self.M).where(sign_c != ">=", -self.M)
                model.add_constraints(lhs + m_arr * disjunction_var, sign_c, rhs + m_arr, name=name + ">=", mask=mask)
            # create the arr
            else:
                m_arr = xr.zeros_like(rhs).where(sign != "<=", self.M).where(sign != ">=", -self.M)
                model.add_constraints(lhs + m_arr * disjunction_var, sign, rhs + m_arr, name=name + ">=", mask=mask)
        else:
            model.add_constraints(lhs, sign, rhs, name=name, mask=mask)

    def rule_to_cons(self, model, rule, index_values, index_list):
        """
        Evaluates the rule on the index_values
        :param model: The linopy model
        :param rule: The rule to call
        :param index_values: A list of index_values to evaluate the rule
        :param index_list: a list of index names
        :return: xarrays of the lhs, sign and the rhs
        """

        # create the mask
        index_arrs = IndexSet.tuple_to_arr(index_values, index_list)
        coords = [np.unique(t.data) for t in index_arrs]

        # there might be an extra label
        if len(index_list) > 1 and len(index_list) != len(index_values[0]):
            index_list = [f"dim_{i}" for i in range(len(index_values[0]))]
        coords = xr.DataArray(coords=coords, dims=index_list).coords
        shape = tuple(map(len, coords.values()))

        # if we only have a single index, there is no need to unpack
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
        xr_coeffs.loc[:, *index_arrs] = coeffs
        xr_vars = xr.DataArray(np.full(shape=(nterm,) + shape, fill_value=-1), coords, dims=("_term", *coords))
        xr_vars.loc[:, *index_arrs] = vars
        xr_ds = xr.Dataset({"coeffs": xr_coeffs, "vars": xr_vars}).transpose(..., "_term")
        xr_lhs = lp.LinearExpression(xr_ds, model)
        xr_sign = xr.DataArray("==", coords, dims=index_list)
        xr_sign.loc[*index_arrs] = [c.sign.data if isinstance(c.sign, xr.DataArray) else c.sign for c in cons]
        xr_rhs = xr.DataArray(0.0, coords, dims=index_list)
        # Here we catch infinities in the constraints (gurobi does not care but glpk does)
        rhs_vals = np.array([c.rhs.data if isinstance(c.rhs, xr.DataArray) else c.rhs for c in cons])
        xr_rhs.loc[*index_arrs] = rhs_vals

        return xr_lhs, xr_sign, xr_rhs

    def add_pw_constraint(self, model, index_values, yvar, xvar, break_points, f_vals, cons_type="EQ"):
        """
        Adds a piece-wise linear constraint of the type f(x) = y for each index in the index_values, where f is defined
        by the breakpoints and f_vals (x_1, y_1), ..., (x_n, y_n)
        Note that these method will create helper variables in form of a S0S2, sources:
         https://support.gurobi.com/hc/en-us/articles/360013421331-How-do-I-model-piecewise-linear-functions-
         https://medium.com/bcggamma/hands-on-modeling-non-linearity-in-linear-optimization-problems-f9da34c23c9a
         :param model: The model to add the constraints to
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
        for index_val in index_values:
            # extract everyting
            x = xvar[*index_val]
            y = yvar[*index_val]
            br = xr.DataArray(break_points[index_val])
            fv = xr.DataArray(f_vals[index_val])

            if len(br) != len(fv):
                raise ValueError("Number of break points should be equal to number of function values for each "
                                 "index value.")

            # create sos vars
            sos2_vars = self._get_nonnegative_sos2_vars(model, len(br))

            # add the constraints
            model.add_constraints(x.to_linexpr() - (br * sos2_vars).sum() == 0)
            model.add_constraints(y.to_linexpr() - (fv * sos2_vars).sum() == 0)

    def _get_nonnegative_sos2_vars(self, model, n):
        """
        Creates a list of continues nonnegative variables in an SOS2
        :param model: The model to add the variables
        :param n: The number of variables to create
        :return: A list of variables that are SOS2 constrained
        """

        # vars and binaries
        sos2_var = model.add_variables(lower=np.zeros(n), binary=False, name=f"sos2_var_{uuid.uuid1()}", coords=(np.arange(n), ))
        sos2_var_bin = model.add_variables(binary=True, name=f"sos2_var_bin_{uuid.uuid1()}", coords=(np.arange(n), ))

        # add the constraints
        model.add_constraints(sos2_var.sum() == 1.0)
        model.add_constraints(sos2_var - sos2_var_bin <= 0.0)
        model.add_constraints(sos2_var_bin.sum() <= 2.0)
        combi_index = xr.DataArray([c for c in combinations(np.arange(n), 2) if c[0] + 1 != c[1]])
        model.add_constraints(sos2_var_bin.sel({"dim_0": combi_index[:, 0]})
                              + sos2_var_bin.sel({"dim_0": combi_index[:, 1]})
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
        sign = xr.DataArray("=", coords=coords[:-1])
        rhs = xr.DataArray(np.nan, coords=coords[:-1])

        for num, con in enumerate(constraints):
            terms = con.lhs.dims["_term"]
            coeffs[num, ..., :terms] = con.lhs.coeffs.data
            variables[num, ..., :terms] = con.lhs.vars.data
            sign[num, ...] = con.sign
            rhs[num, ...] = con.rhs

        xr_ds = xr.Dataset({"coeffs": coeffs, "vars": variables})
        lhs = lp.LinearExpression(xr_ds, model)

        return lp.constraints.AnonymousConstraint(lhs, sign, rhs)





