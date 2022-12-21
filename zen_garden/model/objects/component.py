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
        # set component object
        self.__class__.set_component_object(self)

    @classmethod
    def set_component_object(cls, component_object):
        """ sets component_object"""
        cls.component_object = component_object

    @classmethod
    def get_component_object(cls):
        """ returns component_object """
        if hasattr(cls, "component_object"):
            return cls.component_object
        else:
            raise AttributeError(f"The class {cls.__name__} has not yet been instantiated!")

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


class Parameter(Component):
    def __init__(self):
        """ initialization of the parameter object """
        super().__init__()
        self.min_parameter_value = {"name": None, "value": None}
        self.max_parameter_value = {"name": None, "value": None}

    @classmethod
    def add_parameter(cls, name, data, doc):
        """ initialization of a parameter
        :param name: name of parameter
        :param data: non default data of parameter and index_names
        :param doc: docstring of parameter """
        parameter_object = cls.get_component_object()
        if name not in parameter_object.docs.keys():
            data, index_list = cls.get_index_names_data(data)
            # save if highest or lowest value
            cls.save_min_max(parameter_object, data, name)
            # convert to dict
            data = cls.convert_to_dict(data)
            # set parameter
            setattr(parameter_object, name, data)
            # save additional parameters
            parameter_object.docs[name] = cls.compile_doc_string(doc, index_list, name)
        else:
            logging.warning(f"Parameter {name} already added. Can only be added once")

    @staticmethod
    def save_min_max(parameter_object, data, name):
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
        if not parameter_object.max_parameter_value["name"]:
            parameter_object.max_parameter_value["name"] = _idxmax
            parameter_object.max_parameter_value["value"] = _valmax
            parameter_object.min_parameter_value["name"] = _idxmin
            parameter_object.min_parameter_value["value"] = _valmin
        else:
            if _valmax > parameter_object.max_parameter_value["value"]:
                parameter_object.max_parameter_value["name"] = _idxmax
                parameter_object.max_parameter_value["value"] = _valmax
            if _valmin < parameter_object.min_parameter_value["value"]:
                parameter_object.min_parameter_value["name"] = _idxmin
                parameter_object.min_parameter_value["value"] = _valmin

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

    @classmethod
    def add_variable(cls, block_component: pe.ConcreteModel, name, index_sets, domain, bounds=(None, None), doc=""):
        """ initialization of a variable
        :param block_component: parent block component of variable, must be pe.ConcreteModel
        :param name: name of variable
        :param index_sets: indices and sets by which the variable is indexed
        :param domain: domain of variable
        :param bounds:  bounds of variable
        :param doc: docstring of variable """
        variable_object = cls.get_component_object()
        if name not in variable_object.docs.keys():
            index_values, index_list = cls.get_index_names_data(index_sets)
            var = pe.Var(index_values, domain=domain, bounds=bounds, doc=doc)
            block_component.add_component(name, var)
            # save variable doc
            variable_object.docs[name] = cls.compile_doc_string(doc, index_list, name, domain.name)
        else:
            logging.warning(f"Variable {name} already added. Can only be added once")


class Constraint(Component):
    def __init__(self):
        super().__init__()

    @classmethod
    def add_constraint(cls, block_component: Union[pe.Constraint, pgdp.Disjunct], name, index_sets, rule, doc="", constraint_class=pe.Constraint):
        """ initialization of a variable
        :param block_component: pe.Constraint or pgdp.Disjunct
        :param name: name of variable
        :param index_sets: indices and sets by which the variable is indexed
        :param rule: constraint rule
        :param doc: docstring of variable
        :param constraint_class: either pe.Constraint, pgdp.Disjunct,pgdp.Disjunction"""
        constraint_types = [pe.Constraint, pgdp.Disjunct, pgdp.Disjunction]
        assert constraint_class in constraint_types, f"Constraint type '{constraint_class.name}' unknown"
        constraint_object = cls.get_component_object()
        if name not in constraint_object.docs.keys():
            index_values, index_list = cls.get_index_names_data(index_sets)
            constraint = constraint_class(index_values, rule=rule, doc=doc)
            block_component.add_component(name, constraint)
            # save constraint doc
            constraint_object.docs[name] = cls.compile_doc_string(doc, index_list, name)
        else:
            logging.warning(f"{constraint_class.name} {name} already added. Can only be added once")
