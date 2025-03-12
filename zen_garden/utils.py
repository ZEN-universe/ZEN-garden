"""
Class is defining to read in the results of an Optimization problem.
"""
import json
import logging
import os
import sys
import warnings
import importlib.util
from collections import defaultdict
import re
from ordered_set import OrderedSet
import linopy as lp
import numpy as np
import pandas as pd
import xarray as xr
import shutil
from copy import deepcopy
from pathlib import Path

def setup_logger(level=logging.INFO):
    """ set up logger

    :param level: logging level
    """
    logging.basicConfig(stream=sys.stdout, level=level,format="%(message)s",datefmt='%Y-%m-%d %H:%M:%S')
    logging.captureWarnings(True)


def get_inheritors(klass):
    """
    Get all child classes of a given class

    :param klass: The class to get all children
    :return: All children as a set
    """

    subclasses = OrderedSet()
    work = [klass]
    while work:
        parent = work.pop()
        for child in parent.__subclasses__():
            if child not in subclasses:
                subclasses.add(child)
                work.append(child)
    return subclasses


def copy_dataset_example(example):
    """ copies a dataset example to the current working directory

    :param example: The name of the example to copy
    :return: The local path of the copied example
    :return: The local path of the copied config.json
    """
    import requests
    from importlib.metadata import metadata
    import zipfile
    import io
    url = metadata("zen_garden").get_all("Project-URL")
    url = [u.split(", ")[1] for u in url if u.split(", ")[0] == "Zenodo"][0]
    zenodo_meta = requests.get(url,allow_redirects=True)
    zenodo_meta.raise_for_status()
    zenodo_data = zenodo_meta.json()
    zenodo_zip_url = zenodo_data["files"][0]["links"]["self"]
    zenodo_zip = requests.get(zenodo_zip_url)
    zenodo_zip = zipfile.ZipFile(io.BytesIO(zenodo_zip.content))
    base_path = zenodo_zip.filelist[0].filename
    example_path = f"{base_path}docs/dataset_examples/{example}/"
    config_path = f"{base_path}docs/dataset_examples/config.json"
    notebook_path = f"{base_path}docs/dataset_examples/example_notebook.ipynb"
    local_dataset_path = os.path.join(os.getcwd(), "dataset_examples")
    if not os.path.exists(local_dataset_path):
        os.mkdir(local_dataset_path)
    local_example_path = os.path.join(local_dataset_path, example)
    if not os.path.exists(local_example_path):
        os.mkdir(local_example_path)
    example_found = False
    config_found = False
    notebook_found = False
    for file in zenodo_zip.filelist:
        if file.filename.startswith(example_path):
            filename_ending = file.filename.split(example_path)[1]
            local_folder_path = os.path.join(local_example_path, filename_ending)
            if file.is_dir():
                if not os.path.exists(local_folder_path):
                    os.mkdir(os.path.join(local_example_path, filename_ending))
            else:
                local_file_path = os.path.join(local_example_path, filename_ending)
                with open(local_file_path, "wb") as f:
                    f.write(zenodo_zip.read(file))
            example_found = True
        elif file.filename == config_path:
            with open(os.path.join(local_dataset_path, "config.json"), "wb") as f:
                f.write(zenodo_zip.read(file))
            config_found = True
        elif file.filename == notebook_path:
            notebook_path_local = os.path.join(local_dataset_path, "example_notebook.ipynb")
            notebook = json.loads(zenodo_zip.read(file))
            for cell in notebook['cells']:
                if cell['cell_type'] == 'code':  # Check only code cells
                    for i, line in enumerate(cell['source']):
                        if "<dataset_name>" in line:
                            cell['source'][i] = line.replace("<dataset_name>", example)
            with open(notebook_path_local, "w") as f:
                json.dump(notebook, f)
            notebook_found = True
    assert example_found, f"Example {example} could not be downloaded from the dataset examples!"
    assert config_found, f"Config.json file could not be downloaded from the dataset examples!"
    if not notebook_found:
        logging.warning("Example jupyter notebook could not be downloaded from the dataset examples!")
    logging.info(f"Example dataset {example} downloaded to {local_example_path}")
    return local_example_path, os.path.join(local_dataset_path, "config.json")


# linopy helpers
# --------------

def lp_sum(exprs, dim='_term'):
    """Sum of linear expressions with lp.expressions.merge, returns 0 if list is emtpy

    :param exprs: The expressions to sum
    :param dim: Along which dimension to merge
    :return: The sum of the expressions
    """

    # emtpy sum
    if len(exprs) == 0:
        return 0
    # no sum
    if len(exprs) == 1:
        return exprs[0]
    # normal sum
    return lp.expressions.merge(exprs, dim=dim)


def align_like(da, other, fillna=0.0, astype=None):
    """Aligns a data array like another data array

    :param da: The data array to align
    :param other: The data array to align to
    :param fillna: The value to fill na values with
    :param astype: The type to cast the data array to
    :return: The aligned data array
    """
    if isinstance(other, lp.Variable):
        other = other.lower
    elif isinstance(other, lp.LinearExpression):
        other = other.const
    elif isinstance(other, xr.DataArray):
        other = other
    else:
        raise TypeError(f"other must be a Variable, LinearExpression or DataArray, not {type(other)}")
    da = xr.align(da, other, join="right")[0]
    da = da.broadcast_like(other)
    if fillna is not None:
        da = da.fillna(fillna)
    if astype is not None:
        da = da.astype(astype)
    return da


def linexpr_from_tuple_np(tuples, coords, model):
    """Transforms tuples of (coeff, var) into a linopy linear expression, but uses numpy broadcasting

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


def xr_like(fill_value, dtype, other, dims):
    """Creates an xarray with fill value and dtype like the other object but only containing the given dimensions

    :param fill_value: The value to fill the data with
    :param dtype: dtype of the data
    :param other: The other object to use as base
    :param dims: The dimensions to use
    :return: An object like the other object but only containing the given dimensions
    """

    # get the coords
    coords = {}
    for dim in dims:
        coords[dim] = other.coords[dim]

    # create the data array
    da = xr.DataArray(np.full([len(other.coords[dim]) for dim in dims], fill_value, dtype=dtype), coords=coords,
                      dims=dims)

    # return
    return da

def reformat_slicing_index(index, component) -> tuple[str]:
        """ reformats the slicing index to a tuple of strings that is readable by pytables
        :param index: slicing index of the resulting dataframe
        :param component: component for which the index is reformatted
        :return: reformatted index
        """
        if index is None:
            return tuple()
        index_names = component.index_names
        if isinstance(index, str) or isinstance(index, float) or isinstance(index, int):
            index_name = index_names[0]
            ref_index = (f"{index_name} == {index}",)
        elif isinstance(index, list):
            index_name = index_names[0]
            ref_index = (f"{index_name} in {index}",)
        elif isinstance(index, dict):
            ref_index = []
            for key, value in index.items():
                if key not in index_names:
                    logging.warning(f"Invalid index name '{key}' in index. Skipping.")
                    continue
                if isinstance(value, list):
                    ref_index.append(f"{key} in {value}")
                else:
                    ref_index.append(f"{key} == {value}")
            ref_index = tuple(ref_index)
        elif isinstance(index, tuple):
            ref_index = []
            if len(index) > len(index_names):
                logging.warning(f"Index length {len(index)} is longer than the number of index dimensions {len(index_names)}. Check selected index.")
            for i, index_name in enumerate(index_names):
                if i >= len(index):
                    break
                if index[i] is None:
                    continue
                elif isinstance(index[i], list):
                    ref_index.append(f"{index_name} in {index[i]}")
                else:
                    ref_index.append(f"{index_name} == {index[i]}")
            ref_index = tuple(ref_index)
        else:
            logging.warning(f"Invalid index type {type(index)}. Skipping.")
            ref_index = tuple()

        return ref_index

# This functionality is for the IIS constraints
# ---------------------------------------------


class IISConstraintParser(object):
    """
    This class is used to parse the IIS constraints and print them in a nice way
    Most functions here are just copied from linopy 0.2.x
    """

    EQUAL = "="
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    SIGNS = {EQUAL, GREATER_EQUAL, LESS_EQUAL}
    SIGNS_pretty = {EQUAL: "=", GREATER_EQUAL: "≥", LESS_EQUAL: "≤"}

    def __init__(self, iis_file, model):
        """
        Initializes the IIS constraint parser

        :param iis_file: The file containing the IIS
        :param model: The model to which the IIS belongs
        """
        # disable logger temporarily
        logging.disable(logging.WARNING)
        self.iis_file = iis_file
        self.model = model
        # write gurobi IIS to file
        self.write_gurobi_iis()
        # get the labels
        self.constraint_labels, self.var_labels, self.var_lines = self.read_labels()
        # enable logger again
        logging.disable(logging.NOTSET)

    def write_parsed_output(self, outfile=None):
        """Writes the parsed output to a file

        :param outfile: The file to write to
        """
        # avoid truncating the expression
        # write the outfile
        if outfile is None:
            outfile = self.iis_file
        seen_constraints = []
        seen_variables = []
        with open(outfile, "w") as f:
            f.write("Constraints:\n")
            constraints = self.model.constraints
            for label in self.constraint_labels:
                name, coord = self.get_label_position(constraints, label)
                constraint = constraints[name]
                expr_str = self.print_single_constraint(constraint, coord)
                coords_str = self.print_coord(coord)
                cons_str = f"\t{coords_str}:\t{expr_str}\n"
                if name not in seen_constraints:
                    seen_constraints.append(name)
                    cons_str = f"\n{name}:\n{cons_str}"
                f.write(cons_str)
            f.write("\n\nVariables:\n")
            variables = self.model.variables
            for label in self.var_labels:
                pos = variables.get_label_position(label)
                if pos is not None:
                    name, coord = pos
                    var_str = f"\t{self.print_coord(coord)}:\t{self.var_lines[label]}\n"
                    if name not in seen_variables:
                        seen_variables.append(name)
                        var_str = f"\n{name}:\n{var_str}"
                else:
                    var_str = f"\t{label}:\t{self.var_lines[label]}\n"
                f.write(var_str)

    def write_gurobi_iis(self):
        """ writes IIS to file """
        # get the gurobi model
        gurobi_model = self.model.solver_model
        # write the IIS
        gurobi_model.computeIIS()
        gurobi_model.write(self.iis_file)

    def read_labels(self):
        """Reads the labels from the IIS file

        :return: A list of labels
        """

        labels_c = []
        labels_v = []
        lines_v = {}
        with open(self.iis_file, "rb") as f:
            for line in f.readlines():
                line = line.decode()
                if line.startswith(" c"):
                    labels_c.append(int(line.split(":")[0][2:]))
                elif line.startswith(" x"):
                    pattern = r'\sx(\d+)\s(.*)'
                    match = re.match(pattern, line)
                    if match:
                        labels_v.append(int(match.group(1)))
                        lines_v[int(match.group(1))] = match.group(2).rstrip()
        return labels_c, labels_v, lines_v

    def print_single_constraint(self, constraint, coord):
        """
        Print a single constraint based on the constraint object

        :param constraint: The constraint object
        :param coord: The coordinates of the constraint
        :return: The string representation of the constraint
        """
        coeffs, vars, sign, rhs = xr.broadcast(constraint.coeffs,
                                               constraint.vars,
                                               constraint.sign,
                                               constraint.rhs)
        coeffs = coeffs.sel(coord).values
        vars = vars.sel(coord).values
        sign = sign.sel(coord)[0].item()
        rhs = rhs.sel(coord)[0].item()

        expr = self.print_single_expression(coeffs, vars, self.model)
        # sign = self.SIGNS_pretty[sign]

        return f"{expr} {sign} {rhs:.12g}"

    @staticmethod
    def print_coord(coord):
        """
        Print the coordinates

        :param coord: The coordinates to print
        :return: The string representation of the coordinates
        """

        if isinstance(coord, dict):
            coord = coord.values()
        return "[" + ", ".join([str(c) for c in coord]) + "]" if len(coord) else ""

    @staticmethod
    def get_label_position(constraints, value):
        """
        Get tuple of name and coordinate for variable labels.

        :param constraints: The constraints object
        :param value: The value to get the label for
        :return: The name and coordinate of the variable
        """

        name = constraints.get_name_by_label(value)
        con = constraints[name]
        indices = [i[0] for i in np.where(con.values == value)]

        # Extract the coordinates from the indices
        coord = {
            dim: con.labels.indexes[dim][i] for dim, i in zip(con.labels.dims, indices)
        }

        return name, coord

    @staticmethod
    def print_single_expression(c, v, model):
        """
        This is a linopy routine but without max terms
        Print a single linear expression based on the coefficients and variables.

        :param c: The coefficients of the expression
        :param v: The variables of the expression
        :param model: The model to which the expression belongs
        :return: The string representation of the expression
        """

        # catch case that to many terms would be printed
        def print_line(expr):
            res = []
            for i, (coeff, (name, coord)) in enumerate(expr):
                coord_string = IISConstraintParser.print_coord(coord)
                if i:
                    # split sign and coefficient
                    coeff_string = f"{float(coeff):+.4}"
                    res.append(f"{coeff_string[0]} {coeff_string[1:]} {name}{coord_string}")
                else:
                    res.append(f"{float(coeff):.4} {name}{coord_string}")
            return " ".join(res) if len(res) else "None"

        if isinstance(c, np.ndarray):
            mask = v != -1
            c, v = c[mask], v[mask]

        expr = list(zip(c, model.variables.get_label_position(v)))
        return print_line(expr)


# This class is for the scenario analysis
# ---------------------------------------

class ScenarioDict(dict):
    """
    This is a dictionary for the scenario analysis that has some convenience functions
    """

    _param_dict_keys = {"file", "file_op", "default", "default_op", "value"}
    _special_elements = ["base_scenario", "sub_folder", "param_map"]
    _setting_elements = ["system", "analysis", "solver"]

    def __init__(self, init_dict, optimization_setup, paths):
        """Initializes the dictionary from a normal dictionary

        :param init_dict: The dictionary to initialize from
        :param optimization_setup: The optimization setup corresponding to the scenario
        :param paths: The paths to the elements
        """

        # avoid circular imports
        from . import inheritors
        self.element_classes = reversed(inheritors.copy())

        # set the attributes and expand the dict
        self.system = optimization_setup.system
        self.analysis = optimization_setup.analysis
        self.solver = optimization_setup.solver
        self.init_dict = init_dict
        self.paths = paths
        expanded_dict = self.expand_subsets(init_dict)
        self.validate_dict(expanded_dict)
        self.dict = expanded_dict

        # super init TODO adds both system and "system"  (same for analysis) to the dict - necessary?
        super().__init__(self.dict)

        # finally we update the analysis, system, and solver in the config
        self.update_config()

    def update_config(self):
        """
        Updates the analysis, system, and solver in the config
        """
        config_parts = {"analysis": self.analysis, "system": self.system, "solver": self.solver}
        for key, value in config_parts.items():
            if key in self.dict:
                for sub_key, sub_value in self.dict[key].items():
                    assert sub_key in value, f"Trying to update {key} with key {sub_key} and value {sub_value}, but the {key} does not have this key!"
                    if type(value[sub_key]) == type(sub_value):
                        value[sub_key] = sub_value
                    elif isinstance(sub_value, dict): #ToDO check this and make more general -> here only for SolverOptions
                        try:
                            for sub_sub_key, sub_sub_value in sub_value.items():
                                value[sub_key][sub_sub_key] = sub_sub_value
                        except:
                            raise ValueError(f"Trying to update {key} with key {sub_key} and value {sub_value} of type {type(sub_value)},"
                                             f""f"but the {key} has already a value of type {type(value[sub_key])}")
                    else:
                        raise ValueError(f"Trying to update {key} with key {sub_key} and value {sub_value} of type {type(sub_value)}, "
                                         f"but the {key} has already a value of type {type(value[sub_key])}")

    @staticmethod
    def expand_lists(scenarios: dict):
        """Expands all lists of parameters in the all scenarios and returns a new dict

        :param scenarios: The initial dict of scenarios
        :return: The expanded dict, where all necessary parameters are expanded and subpaths are set
        """

        # Important, all for-loops through keys or items in this routine should be sorted!

        expanded_scenarios = dict()
        for scenario_name, scenario_dict in sorted(scenarios.items(), key=lambda x: x[0]):
            assert type(scenario_dict) == dict, f"Scenario {scenario_name} is not a dictionary!"
            scenario_dict["base_scenario"] = scenario_name
            scenario_dict["sub_folder"] = ""
            scenario_dict["param_map"] = dict()
            scenario_list = ScenarioDict._expand_scenario(scenario_dict)

            # add the scenarios to the dict
            for scenario in scenario_list:
                if scenario["sub_folder"] == "":
                    name = scenario["base_scenario"]
                else:
                    name = scenario["base_scenario"] + "_" + scenario["sub_folder"]
                expanded_scenarios[name] = scenario

        return expanded_scenarios

    @staticmethod
    def _expand_scenario(scenario: dict, param_map=None, counter=0):
        """Expands a scenario, returns a list of scenarios

        :param scenario: The scenario to expand
        :param param_map: The parameter map for the scenario
        :param counter: The counter for the scenario
        :return: A list of scenarios
        """

        # get the default
        if param_map is None:
            param_map = dict()

        # list for the expanded scenarios
        expanded_scenarios = []

        # iterate over all elements
        for element, element_dict in sorted(scenario.items(), key=lambda x: x[0]):
            # we do not expand these
            if element in ScenarioDict._special_elements:
                continue
            # check for dict items in settings elements
            # if element in ScenarioDict._setting_elements and dict not in [type(v) for v in element_dict.values()]:
            #     continue

            # check for 'system' analysis' and 'solver' keys and see whether they are dicts and have a list in them,
            # on ly then do the list expansion, otherwise proceed as always.
            for param, param_dict in sorted(element_dict.items(), key=lambda x: x[0]):
                if element in ScenarioDict._setting_elements:
                    if not isinstance(param_dict, dict):
                        continue
                    elif isinstance(param_dict, dict) and not isinstance(param_dict['value'], list):
                        scenario[element][param] = param_dict['value']
                for key in sorted(ScenarioDict._param_dict_keys):
                    if key in param_dict and isinstance(param_dict[key], list):
                        # get the old param dict entry
                        if scenario["sub_folder"] != "":
                            old_param_map_entry = param_map.pop(scenario["sub_folder"])
                        else:
                            old_param_map_entry = dict()

                        # we need to expand this
                        for num, value in enumerate(param_dict[key]):
                            # copy the scenario
                            new_scenario = deepcopy(scenario)

                            # set the new value
                            if element in ScenarioDict._setting_elements:
                                new_scenario[element][param] = value
                            else:
                                new_scenario[element][param][key] = value

                            # create the name
                            if key + "_fmt" in param_dict:
                                if "{}" not in param_dict[key + "_fmt"]:
                                    raise SyntaxError("When setting a format for a name, you need to include a "
                                                      "placeholder '{}' for its value! No placeholder found in "
                                                      f"for {key} in {param} in {element} in {scenario['base_scenario']}")
                                name = param_dict[key + "_fmt"].format(value)
                                if element not in ScenarioDict._setting_elements:
                                    del new_scenario[element][param][key + "_fmt"]
                                # we don't need to increment the param for the next expansion
                                param_up = 0
                            else:
                                name = f"p{counter:02d}_{num:03d}"
                                # we need to increment the param for the next expansion
                                param_up = 1

                            # set the sub_folder
                            if new_scenario["sub_folder"] == "":
                                new_scenario["sub_folder"] = name
                            else:
                                new_scenario["sub_folder"] += "_" + name

                            # update the param_map
                            param_map[new_scenario["sub_folder"]] = deepcopy(old_param_map_entry)
                            if element not in param_map[new_scenario["sub_folder"]]:
                                param_map[new_scenario["sub_folder"]][element] = dict()
                            if param not in param_map[new_scenario["sub_folder"]][element]:
                                param_map[new_scenario["sub_folder"]][element][param] = dict()
                            if element in ScenarioDict._setting_elements:
                                param_map[new_scenario["sub_folder"]][element][param] = value
                            else:
                                param_map[new_scenario["sub_folder"]][element][param][key] = value

                            # set the param_map of the scenario
                            new_scenario["param_map"] = param_map

                            # expand this scenario as well
                            expanded_scenarios.extend(
                                ScenarioDict._expand_scenario(new_scenario, param_map, counter + param_up))

                        # expansion done
                        return expanded_scenarios

        # nothing was expanded, so we just return the scenario
        expanded_scenarios.append(scenario)

        # return the list
        return expanded_scenarios

    def expand_subsets(self, init_dict):
        """Expands a dictionary, e.g. expands sets etc.

        :param init_dict: The initial dict
        :return: A new dict which can be used for the scenario analysis
        """

        new_dict = init_dict.copy()
        for element_class in self.element_classes:
            current_set = element_class.label
            if current_set in new_dict:
                for param, param_dict in new_dict[current_set].items():
                    # dict for expansion
                    base_dict = param_dict

                    # get the exlusion list
                    if "exclude" in base_dict:
                        exclude_list = base_dict["exclude"]
                        del base_dict["exclude"]
                    else:
                        exclude_list = []

                    # expand the sets
                    for element in self.paths[current_set].keys():
                        if element != "folder" and element not in exclude_list:
                            # create dicts if necessary
                            if element not in new_dict:
                                new_dict[element] = {}
                            # we only set the param dict if it is not already set
                            if param not in new_dict[element]:
                                new_dict[element][param] = base_dict.copy()
                # delete the old set
                del new_dict[current_set]
        return new_dict

    def validate_dict(self, vali_dict):
        """Validates a dictionary, raises an error if it is not valid

        :param vali_dict: The dictionary to validate
        """

        for element, element_dict in vali_dict.items():
            if element in self._special_elements or element in self._setting_elements:
                continue

            if not isinstance(element_dict, dict):
                raise ValueError(f"The entry for {element} is not a dictionary!")

            for param, param_dict in element_dict.items():
                if len(diff := (set(param_dict.keys()) - self._param_dict_keys)) > 0:
                    raise ValueError(
                        f"The entry for element {element} and param {param} contains invalid entries: {diff}!")

    @staticmethod
    def validate_file_name(fname):
        """Checks if the file name has an extension, it is expected to not have an extension

        :param fname: The file name to validte
        :return: The validated file name
        """

        fname, ext = os.path.splitext(fname)
        if ext != "":
            warnings.warn(f"The file name {fname}{ext} has an extension {ext}, removing it.")
        return fname

    def get_default(self, element, param):
        """Return the name where the default value should be read out

        :param element: The element name
        :param param: The parameter of the element
        :return: If the entry is overwritten by the scenario analysis the entry and factor are returned, otherwise
                 the default entry is returned with a factor of 1
        """

        # These are the default values
        default_f_name = "attributes"
        default_factor = 1.0

        if element in self.dict and param in (element_dict := self.dict[element]):
            param_dict = element_dict[param]
            default_f_name = param_dict.get("default", default_f_name)
            default_f_name = self.validate_file_name(default_f_name)
            default_factor = param_dict.get("default_op", default_factor)
            self._check_if_numeric_default_factor(default_factor, element=element, param=param, default_f_name=default_f_name, op_type="default_op")

        return default_f_name, default_factor

    def get_param_file(self, element, param):
        """Return the file name where the parameter values should be read out

        :param element: The element name
        :param param: The parameter of the element
        :return: If the entry is overwritten by the scenario analysis the entry and factor are returned, otherwise
                 the default entry is returned with a factor of 1
        """

        # These are the default values
        default_f_name = param
        default_factor = 1.0

        if element in self.dict and param in (element_dict := self.dict[element]):
            param_dict = element_dict[param]
            default_f_name = param_dict.get("file", default_f_name)
            default_f_name = self.validate_file_name(default_f_name)
            default_factor = param_dict.get("file_op", default_factor)
            self._check_if_numeric_default_factor(default_factor, element=element, param=param, default_f_name=default_f_name, op_type="file_op")

        return default_f_name, default_factor

    def _check_if_numeric_default_factor(self, default_factor, element, param, default_f_name, op_type):
        """Check if the default factor is numeric

        :param default_factor: The default factor to check
        """
        if not isinstance(default_factor, (int, float)):
            raise ValueError(f"Default factor {default_factor} of type {type(default_factor)} in {op_type} ({element} -> {param} -> {default_f_name}) is not numeric!")

class InputDataChecks:
    """
    This class checks if the input data (folder/file structure, system.py settings, element definitions, etc.) is defined correctly
    """
    PROHIBITED_DATASET_CHARACTERS = [" ", ".", ":", ",", ";", "!", "?", "(", ")", "[", "]", "{", "}", "<", ">", "&", "|", "*", "^", "%", "$", "#", "@", "`", "~", "\\", "/"]

    def __init__(self, config, optimization_setup):
        """
        Initialize the class

        :param config: config object used to extract the analysis, system and solver dictionaries
        :param optimization_setup: OptimizationSetup instance
        """
        self.system = config.system
        self.analysis = config.analysis
        self.optimization_setup = optimization_setup

    def check_technology_selections(self):
        """
        Checks selection of different technologies in system.py file
        """
        # Checks if at least one technology is selected in the system.py file
        assert len(self.system.set_conversion_technologies + self.system.set_transport_technologies + self.system.set_storage_technologies) > 0, f"No technology selected in system.py"
        # Checks if identical technologies are selected multiple times in system.py file and removes possible duplicates
        for tech_list in ["set_conversion_technologies", "set_transport_technologies", "set_storage_technologies"]:
            techs_selected = self.system[tech_list]
            unique_elements = list(np.unique(techs_selected))
            self.system[tech_list] = unique_elements

    def check_year_definitions(self):
        """
        Check if year-related parameters are defined correctly
        """
        # assert that number of optimized years is a positive integer
        assert isinstance(self.system.optimized_years, int) and self.system.optimized_years > 0, f"Number of optimized years must be a positive integer, however it is {self.system.optimized_years}"
        # assert that interval between years is a positive integer
        assert isinstance(self.system.interval_between_years, int) and self.system.interval_between_years > 0, f"Interval between years must be a positive integer, however it is {self.system.interval_between_years}"
        assert isinstance(self.system.reference_year, int) and self.system.reference_year >= self.analysis.earliest_year_of_data, f"Reference year must be an integer and larger than the defined earliest_year_of_data: {self.analysis.earliest_year_of_data}"
        # check if the number of years in the rolling horizon isn't larger than the number of optimized years
        if self.system.years_in_rolling_horizon > self.system.optimized_years and self.system.use_rolling_horizon:
            warnings.warn(f"The chosen number of years in the rolling horizon step is larger than the total number of years optimized!")

    def check_primary_folder_structure(self):
        """
        Checks if the primary folder structure (set_conversion_technology, set_transport_technology, ..., energy_system) is provided correctly
        """

        for set_name, subsets in self.analysis.subsets.items():
            if not os.path.exists(os.path.join(self.analysis.dataset, set_name)):
                raise AssertionError(f"Folder {set_name} does not exist!")
            if isinstance(subsets, dict):
                for subset_name, subset in subsets.items():
                    if not os.path.exists(os.path.join(self.analysis.dataset, set_name, subset_name)):
                        raise AssertionError(f"Folder {subset_name} does not exist!")
                else:
                    for subset_name in subsets:
                        if not os.path.exists(os.path.join(self.analysis.dataset, set_name, subset_name)):
                            raise AssertionError(f"Folder {subset_name} does not exist!")

        for file_name in ["attributes.json", "base_units.csv", "set_edges.csv", "set_nodes.csv", "unit_definitions.txt"]:
            if file_name not in os.listdir(os.path.join(self.analysis.dataset, "energy_system")) and file_name.replace('.csv', '.json') not in os.listdir(os.path.join(self.analysis.dataset, "energy_system")):
                raise FileNotFoundError(f"File {file_name} is missing in the energy_system directory")

    def check_existing_technology_data(self):
        """
        This method checks the existing technology input data and only regards those technology elements for which folders containing the attributes.json file exist.
        """
        # TODO works for two levels of subsets, but not for more
        self.optimization_setup.system.set_technologies = []
        for set_name, subsets in self.optimization_setup.analysis.subsets["set_technologies"].items():
            for technology in self.optimization_setup.system[set_name]:
                if technology not in self.optimization_setup.paths[set_name].keys():
                    # raise error if technology is not in input data
                    raise FileNotFoundError(f"Technology {technology} selected in config does not exist in input data")
                elif "attributes.json" not in self.optimization_setup.paths[set_name][technology]:
                    raise FileNotFoundError(f"The file attributes.json does not exist for the technology {technology}")
            self.optimization_setup.system.set_technologies.extend(self.optimization_setup.system[set_name])
            # check subsets of technology_subset
            assert isinstance(subsets, list), f"Subsets of {set_name} must be a list, dict not implemented"
            for subset in subsets:
                for technology in self.optimization_setup.system[subset]:
                    if technology not in self.optimization_setup.paths[subset].keys():
                        # raise error if technology is not in input data
                        raise FileNotFoundError(f"Technology {technology} selected in config does not exist in input data")
                    elif "attributes.json" not in self.optimization_setup.paths[subset][technology]:
                        raise FileNotFoundError(f"The file attributes.json does not exist for the technology {technology}")
                    self.optimization_setup.system[set_name].extend(self.optimization_setup.system[subset])
                    self.optimization_setup.system.set_technologies.extend(self.optimization_setup.system[subset])

    def check_existing_carrier_data(self):
        """
        Checks the existing carrier data and only regards those carriers for which folders exist
        """
        # check if carriers exist
        for carrier in self.optimization_setup.system.set_carriers:
            if carrier not in self.optimization_setup.paths["set_carriers"].keys():
                # raise error if carrier is not in input data
                raise FileNotFoundError(f"Carrier {carrier} selected in config does not exist in input data")
            elif "attributes.json" not in self.optimization_setup.paths["set_carriers"][carrier]:
                raise FileNotFoundError(f"The file attributes.json does not exist for the carrier {carrier}")

    def check_dataset(self):
        """
        Ensures that the dataset chosen in the config does exist and contains a system.py file
        """
        dataset = os.path.basename(self.analysis.dataset)
        dirname = os.path.dirname(self.analysis.dataset)
        assert os.path.exists(dirname), f"Requested folder {dirname} is not a valid path"
        assert os.path.exists(self.analysis.dataset), f"The chosen dataset {dataset} does not exist at {self.analysis.dataset} as it is specified in the config"
        # check if any character in the dataset name is prohibited
        for char in self.PROHIBITED_DATASET_CHARACTERS:
            if char in dataset:
                raise ValueError(f"Character {char} is not allowed in the dataset name {dataset}\nProhibited characters: {self.PROHIBITED_DATASET_CHARACTERS}")
        # check if chosen dataset contains a system.py file
        if not os.path.exists(os.path.join(self.analysis.dataset, "system.py")) and not os.path.exists(os.path.join(self.analysis.dataset, "system.json")):
            raise FileNotFoundError(f"Neither system.json nor system.py not found in dataset: {self.analysis.dataset}")

    def check_single_directed_edges(self, set_edges_input):
        """
        Checks if single-directed edges exist in the dataset (e.g. CH-DE exists, DE-CH doesn't) and raises a warning

        :param set_edges_input: DataFrame containing set of edges defined in set_edges.csv
        """
        for edge in set_edges_input.values:
            reversed_edge = edge[2] + "-" + edge[1]
            if reversed_edge not in [edge_string[0] for edge_string in set_edges_input.values] and edge[1] in self.system.set_nodes and edge[2] in self.system.set_nodes:
                warnings.warn(f"The edge {edge[0]} is single-directed, i.e., the edge {reversed_edge} doesn't exist!")

    @staticmethod
    def check_carrier_configuration(input_carrier, output_carrier, reference_carrier, name):
        """
        Checks if the chosen input/output and reference carrier combination is reasonable

        :param input_carrier: input carrier of conversion technology
        :param output_carrier: output carrier of conversion technology
        :param reference_carrier: reference carrier of technology
        :param name: name of conversion technology
        """
        # assert that conversion technology has at least an input or an output carrier
        assert len(input_carrier+output_carrier) > 0, f"Conversion technology {name} has neither an input nor an output carrier!"
        # check if reference carrier in input and output carriers and set technology to correspondent carrier
        assert reference_carrier[0] in (input_carrier + output_carrier), f"reference carrier {reference_carrier} of technology {name} not in input and output carriers {input_carrier + output_carrier}"
        set_input_carrier = set(input_carrier)
        set_output_carrier = set(output_carrier)
        # assert that input and output carrier of conversion tech are different
        common_carriers = set_input_carrier & set_output_carrier
        assert not common_carriers, f"The conversion technology {name} has the same input and output carrier(s) ({list(common_carriers)})!"

    @staticmethod
    def check_duplicate_indices(df_input, file_name, folder_path):
        """
        Checks if df_input contains any duplicate indices and either removes them if they are of identical value or raises an error otherwise

        :param df_input: raw input dataframe
        :param folder_path: the path of the folder containing the selected file
        :param file_name: name of selected file
        :return: df_input without duplicate indices
        """
        unique_elements, counts = np.unique(df_input.index, return_counts=True)
        duplicates = unique_elements[counts > 1]

        if len(duplicates) != 0:
            for duplicate in duplicates:
                values = df_input.loc[duplicate]
                # check if all the duplicates are of the same value
                if values.nunique() == 1:
                    logging.warning(f"The input data file {file_name + '.csv'} at {folder_path} contains duplicate indices with identical values: {df_input.loc[duplicates]}.")
                else:
                    raise AssertionError(f"The input data file {file_name + '.csv'} at {folder_path} contains duplicate indices with different values: {df_input.loc[duplicates]}.")
            # remove duplicates
            duplicate_mask = df_input.index.duplicated(keep='first')
            df_input = df_input[~duplicate_mask]

        return df_input

    @staticmethod
    def read_system_file(config):
        """
        Reads the system file and returns the system dictionary

        :param config: config object
        """
        # check if system.json file exists
        if os.path.exists(os.path.join(config.analysis.dataset, "system.json")):
            with open(os.path.join(config.analysis.dataset, "system.json"), "r") as file:
                system = json.load(file)
        # otherwise read system.py file
        else:
            system_path = os.path.join(config.analysis.dataset, "system.py")
            spec = importlib.util.spec_from_file_location("module", system_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            system = module.system
        config.system.update(system)


class StringUtils:
    """
    This class handles some strings for logging and filenames to tidy up scripts
    """
    def __init__(self):
        """ Initializes the class """
        pass

    @classmethod
    def print_optimization_progress(cls, scenario, steps_horizon, step, system):
        """ prints the current optimization progress

        :param scenario: string of scenario name
        :param steps_horizon: all steps of horizon
        :param step: current step of horizon
        :param system: system of optimization
        """
        scenario_string = ScenarioUtils.scenario_string(scenario)
        if len(steps_horizon) == 1:
            logging.info(f"\n--- Conduct optimization for perfect foresight {scenario_string}--- \n")
        else:
            corresponding_year = system.reference_year + step * system.interval_between_years
            logging.info(
                f"\n--- Conduct optimization for rolling horizon step for {corresponding_year} ({steps_horizon.index(step) + 1} of {len(steps_horizon)}) {scenario_string}--- \n")

    @classmethod
    def generate_folder_path(cls, config, scenario, scenario_dict, steps_horizon, step):
        """ generates the folder path for the results

        :param config: config of optimization
        :param scenario: name of scenario
        :param scenario_dict: current scenario dict
        :param steps_horizon: all steps of horizon
        :param step: current step of horizon
        :return: scenario name in folder
        :return: subfolder in results file
        :return: mapping of parameters
        """
        subfolder = Path("")
        scenario_name = None
        param_map = None
        if config.system.conduct_scenario_analysis:
            # handle scenarios
            scenario_name = f"scenario_{scenario}"
            subfolder = Path(f"scenario_{scenario_dict['base_scenario']}")

            # set the scenarios
            if scenario_dict["sub_folder"] != "":
                # get the param map
                param_map = scenario_dict["param_map"]

                # get the output scenarios
                subfolder = subfolder.joinpath(f"scenario_{scenario_dict['sub_folder']}")
                scenario_name = f"scenario_{scenario_dict['sub_folder']}"

        # handle myopic foresight
        if len(steps_horizon) > 1:
            mf_f_string = f"MF_{step}"
            # handle combination of MF and scenario analysis
            if config.system.conduct_scenario_analysis:
                subfolder = Path(subfolder), Path(mf_f_string)
            else:
                subfolder = Path(mf_f_string)

        return scenario_name, subfolder, param_map

    @classmethod
    def setup_model_folder(cls, analysis, system):
        """return model name while conducting some tests

        :param analysis: analysis of optimization
        :param system: system of optimization
        :return: model name
        :return: output folder
        """
        model_name = os.path.basename(analysis.dataset)
        out_folder = cls.setup_output_folder(analysis, system)
        return model_name, out_folder

    @classmethod
    def setup_output_folder(cls, analysis, system):
        """return model name while conducting some tests

        :param analysis: analysis of optimization
        :param system: system of optimization
        :return: output folder
        """
        if not os.path.exists(analysis.folder_output):
            try:
                os.mkdir(analysis.folder_output)
            except FileExistsError:
                pass
        out_folder = cls.get_output_folder(analysis)
        if not os.path.exists(out_folder):
            try:
                os.mkdir(out_folder)
            except FileExistsError:
                pass
        else:
            logging.warning(f"The output folder '{out_folder}' already exists")
            if analysis.overwrite_output:
                logging.warning("Existing files will be overwritten!")
                if not system.conduct_scenario_analysis:
                    # TODO fix for scenario analysis, shared folder for all scenarios, so not robust for parallel process
                    for filename in os.listdir(out_folder):
                        file_path = os.path.join(out_folder, filename)
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
        return out_folder

    @staticmethod
    def get_output_folder(analysis):
        """return name of output folder

        :param analysis: analysis of optimization
        :return: output folder
        """
        model_name = os.path.basename(analysis.dataset)
        out_folder = os.path.join(analysis.folder_output, model_name)
        return out_folder


class ScenarioUtils:
    """
    This class handles some stuff for scenarios to tidy up scripts
    """

    def __init__(self):
        """ Initializes the class """
        pass

    @staticmethod
    def scenario_string(scenario):
        """ creates additional scenario string

        :param scenario: scenario name
        :return: scenario string """
        if scenario != "":
            scenario_string = f"for scenario {scenario} "
        else:
            scenario_string = ""
        return scenario_string

    @staticmethod
    def clean_scenario_folder(config, out_folder):
        """ cleans scenario dict when overwritten

        :param config: config of optimization
        :param out_folder: output folder"""
        # compare to existing sub-scenarios
        if config.system.conduct_scenario_analysis and config.system.clean_sub_scenarios:
            # collect all paths that are in the scenario dict
            folder_dict = defaultdict(list)
            for key, value in config.scenarios.items():
                if value["sub_folder"] != "":
                    folder_dict[f"scenario_{value['base_scenario']}"].append(f"scenario_{value['sub_folder']}")
                    folder_dict[f"scenario_{value['base_scenario']}"].append(
                        f"dict_all_sequence_time_steps_{value['sub_folder']}.h5")
            for scenario_name, sub_folders in folder_dict.items():
                scenario_path = os.path.join(out_folder, scenario_name)
                if os.path.exists(scenario_path) and os.path.isdir(scenario_path):
                    existing_sub_folder = os.listdir(scenario_path)
                    for sub_folder in existing_sub_folder:
                        # delete the scenario subfolder
                        sub_folder_path = os.path.join(scenario_path, sub_folder)
                        if os.path.isdir(sub_folder_path) and sub_folder not in sub_folders:
                            logging.info(f"Removing sub-scenario {sub_folder}")
                            shutil.rmtree(sub_folder_path, ignore_errors=True)
                        # the time steps dict
                        if sub_folder.startswith("dict_all_sequence_time_steps") and sub_folder not in sub_folders:
                            logging.info(f"Removing time steps dict {sub_folder}")
                            os.remove(sub_folder_path)

    @staticmethod
    def get_scenarios(config, job_index):
        """ retrieves and overwrites the scenario dicts

        :param config: config of optimization
        :param job_index: index of current job, if passed
        :return: scenarios of optimization
        :return: elements in scenario
        """
        if config.system.conduct_scenario_analysis:
            scenarios_path = os.path.abspath(os.path.join(config.analysis.dataset, "scenarios.json"))
            if os.path.exists(scenarios_path):
                with open(scenarios_path, "r") as file:
                    scenarios = json.load(file)
            else:
                scenarios_path = os.path.abspath(os.path.join(config.analysis.dataset, "scenarios.py"))
                if not os.path.exists(scenarios_path):
                    raise FileNotFoundError(f"Neither scenarios.json nor scenarios.py not found in dataset: {config.analysis.dataset}")
                spec = importlib.util.spec_from_file_location("module", scenarios_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                scenarios = module.scenarios
            config.scenarios.update(scenarios)
            # remove the default scenario if necessary
            if not config.system.run_default_scenario and "" in config.scenarios:
                del config.scenarios[""]

            # expand the scenarios
            config.scenarios = ScenarioDict.expand_lists(config.scenarios)

            # deal with the job array
            if job_index is not None:
                if isinstance(job_index, int):
                    job_index = [job_index]
                else:
                    job_index = list(job_index)
                logging.info(f"Running scenarios with job indices: {job_index}")
                # reduce the scenario and element to a single one
                scenarios = [list(config.scenarios.keys())[jx] for jx in job_index]
                elements = [list(config.scenarios.values())[jx] for jx in job_index]
            else:
                logging.info(f"Running all scenarios sequentially")
                scenarios = config.scenarios.keys()
                elements = config.scenarios.values()
        # Nothing to do with the scenarios
        else:
            scenarios = [""]
            elements = [{}]
        return scenarios, elements


class OptimizationError(RuntimeError):
    """
    Exception raised when the optimization problem is infeasible
    """

    def __init__(self, status="The optimization is infeasible or unbounded, or finished with an error"):
        """
        Initializes the class

        :param status: The message to display
        """
        self.message = f"The termination condition was {status}"
        super().__init__(self.message)
