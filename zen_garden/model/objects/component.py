"""===========================================================================================================================================================================
Title:          ZEN-GARDEN
Created:        July-2022
Authors:        Jacob Mannhardt (jmannhardt@ethz.ch)
Organization:   Laboratory of Reliability and Risk Engineering, ETH Zurich

Description:    Class containing parameters. This is a proxy for pyomo parameters, since the construction of parameters has a significant overhead.
==========================================================================================================================================================================="""
import copy
import logging
import numpy as np
import pandas as pd
import pyomo.environ as pe
import pyomo.gdp as pgdp
import itertools
from typing import Union


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
        if isinstance(index_list, tuple):
            index_values, index_names = index_list
        elif isinstance(index_list, pe.Set):
            index_values = copy.copy(index_list)
            index_names = [index_list.name]
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

        # attributes
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
        self.sets[name] = data
        self.docs[name] = doc
        if index_set is not None:
            self.index_sets[name] = index_set

    def is_indexed(self, name):
        """
        Checks if the set with the name is indexed
        :param name: The name of the set
        :return: True if indexed, False otherwise
        """

        return name in self.index_sets

    def get_index_name(self, name):
        """
        Returns the index name of an indexed set
        :param name: The name of the indexed set
        :return: The name of the index set
        """

        if not self.is_indexed(name=name):
            raise ValueError(f"Set {name} is not an indexed set!")
        return self.index_sets[name]

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
            # convert to dict
            data = self.convert_to_dict(data)
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
    def convert_to_dict(data):
        """ converts the data to a dict if pd.Series"""
        if isinstance(data, pd.Series):
            # if single entry in index
            if len(data.index[0]) == 1:
                data.index = pd.Index(sum(data.index.values, ()))
            data = data.to_dict()
        return data


class Variable(Component):
    def __init__(self):
        super().__init__()

    def add_variable(self, block_component: pe.ConcreteModel, name, index_sets, domain, bounds=(None, None), doc=""):
        """ initialization of a variable
        :param block_component: parent block component of variable, must be pe.ConcreteModel
        :param name: name of variable
        :param index_sets: indices and sets by which the variable is indexed
        :param domain: domain of variable
        :param bounds:  bounds of variable
        :param doc: docstring of variable """

        if name not in self.docs.keys():
            index_values, index_list = self.get_index_names_data(index_sets)
            var = pe.Var(index_values, domain=domain, bounds=bounds, doc=doc)
            block_component.add_component(name, var)
            # save variable doc
            self.docs[name] = self.compile_doc_string(doc, index_list, name, domain.name)
        else:
            logging.warning(f"Variable {name} already added. Can only be added once")


class Constraint(Component):
    def __init__(self):
        super().__init__()

    def add_constraint(self, block_component: Union[pe.Constraint, pgdp.Disjunct], name, index_sets, rule, doc="", constraint_class=pe.Constraint):
        """ initialization of a variable
        :param block_component: pe.Constraint or pgdp.Disjunct
        :param name: name of variable
        :param index_sets: indices and sets by which the variable is indexed
        :param rule: constraint rule
        :param doc: docstring of variable
        :param constraint_class: either pe.Constraint, pgdp.Disjunct,pgdp.Disjunction"""
        constraint_types = [pe.Constraint, pgdp.Disjunct, pgdp.Disjunction]
        assert constraint_class in constraint_types, f"Constraint type '{constraint_class.name}' unknown"

        if name not in self.docs.keys():
            index_values, index_list = self.get_index_names_data(index_sets)
            constraint = constraint_class(index_values, rule=rule, doc=doc)
            block_component.add_component(name, constraint)
            # save constraint doc
            self.docs[name] = self.compile_doc_string(doc, index_list, name)
        else:
            logging.warning(f"{constraint_class.name} {name} already added. Can only be added once")
