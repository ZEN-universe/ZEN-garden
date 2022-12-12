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

class Component:

    def __init__(self):
        self.docs = {}
        # set component object
        self.__class__.setComponentObject(self)

    @classmethod
    def setComponentObject(cls,componentObject):
        """ sets componentObject"""
        cls.componentObject = componentObject

    @classmethod
    def get_component_object(cls):
        """ returns componentObject """
        if hasattr(cls,"componentObject"):
            return cls.componentObject
        else:
            raise AttributeError(f"The class {cls.__name__} has not yet been instantiated!")

    @staticmethod
    def compileDocString(doc,index_list,name,domain = None):
        """ compile docstring from doc and index_list"""
        assert type(doc)==str,f"Docstring {doc} has wrong format. Must be 'str' but is '{type(doc).__name__}'"
        # check for prohibited strings
        prohibitedStrings = [",",";",":","/","name","doc","dims","domain"]
        originalDoc = copy.copy(doc)
        for string in prohibitedStrings:
            if string in doc:
                logging.warning(f"Docstring '{originalDoc}' contains prohibited string '{string}'. Occurrences are dropped.")
                doc = doc.replace(string,"")
        # joined index names
        joinedIndex = ",".join(index_list)
        # complete doc string
        completeDoc = f"name:{name};doc:{doc};dims:{joinedIndex}"
        if domain:
            completeDoc += f";domain:{domain}"
        return completeDoc

    @staticmethod
    def getIndexNamesData(index_list):
        """ splits index_list in data and index names """
        if isinstance(index_list,tuple):
            indexValues,index_names = index_list
        elif isinstance(index_list,pe.Set):
            indexValues = copy.copy(index_list)
            index_names  = [index_list.name]
        else:
            raise TypeError(f"Type {type(index_list)} unknown to extract index names.")
        return indexValues,index_names

class Parameter(Component):
    def __init__(self):
        """ initialization of the parameter object """
        super().__init__()
        self.minParameterValue  = {"name":None,"value":None}
        self.maxParameterValue  = {"name":None,"value":None}

    @classmethod
    def add_parameter(cls,name, data, doc):
        """ initialization of a parameter
        :param name: name of parameter
        :param data: non default data of parameter and index_names
        :param doc: docstring of parameter """
        parameterObject = cls.get_component_object()
        if name not in parameterObject.docs.keys():
            data, index_list = cls.getIndexNamesData(data)
            # save if highest or lowest value
            cls.saveMinMax(parameterObject,data,name)
            # convert to dict
            data = cls.convertToDict(data)
            # set parameter
            setattr(parameterObject, name, data)
            # save additional parameters
            parameterObject.docs[name] = cls.compileDocString(doc,index_list,name)
        else:
            logging.warning(f"Parameter {name} already added. Can only be added once")

    @staticmethod
    def saveMinMax(parameterObject,data,name):
        """ stores min and max parameter """
        if isinstance(data,dict) and data:
            data = pd.Series(data)
        if isinstance(data,pd.Series):
            _abs = data.abs()
            _abs = _abs[(_abs != 0) & (_abs != np.inf)]
            if not _abs.empty:
                _idxmax = name + "_" + "_".join(map(str,_abs.index[_abs.argmax()]))
                _valmax = _abs.max()
                _idxmin = name + "_" + "_".join(map(str,_abs.index[_abs.argmin()]))
                _valmin = _abs.min()
            else:
                return
        else:
            if not data or (abs(data) == 0) or (abs(data) == np.inf):
                return
            _abs    = abs(data)
            _idxmax = name
            _valmax = _abs
            _idxmin = name
            _valmin = _abs
        if not parameterObject.maxParameterValue["name"]:
            parameterObject.maxParameterValue["name"]   = _idxmax
            parameterObject.maxParameterValue["value"]  = _valmax
            parameterObject.minParameterValue["name"]   = _idxmin
            parameterObject.minParameterValue["value"]  = _valmin
        else:
            if _valmax > parameterObject.maxParameterValue["value"]:
                parameterObject.maxParameterValue["name"]   = _idxmax
                parameterObject.maxParameterValue["value"]  = _valmax
            if _valmin < parameterObject.minParameterValue["value"]:
                parameterObject.minParameterValue["name"]   = _idxmin
                parameterObject.minParameterValue["value"]  = _valmin

    @staticmethod
    def convertToDict(data):
        """ converts the data to a dict if pd.Series"""
        if isinstance(data, pd.Series):
            # if single entry in index
            if len(data.index[0]) == 1:
                data.index = pd.Index(sum(data.index.values,()))
            data = data.to_dict()
        return data

class Variable(Component):
    def __init__(self):
        super().__init__()

    @classmethod
    def add_variable(cls, model:pe.ConcreteModel, name, index_sets, domain,bounds = (None,None), doc = ""):
        """ initialization of a variable
        :param model:       pe.ConcreteModel
        :param name:        name of variable
        :param index_sets:   indices and sets by which the variable is indexed
        :param domain:      domain of variable
        :param bounds:      bounds of variable
        :param doc:         docstring of variable """
        variableObject = cls.get_component_object()
        if name not in variableObject.docs.keys():
            indexValues,index_list = cls.getIndexNamesData(index_sets)
            var = pe.Var(
                indexValues,
                domain = domain,
                bounds = bounds,
                doc    = doc
            )
            model.add_component(name,var)
            # save variable doc
            variableObject.docs[name] = cls.compileDocString(doc,index_list,name,domain.name)
        else:
            logging.warning(f"Variable {name} already added. Can only be added once")

class Constraint(Component):
    def __init__(self):
        super().__init__()

    @classmethod
    def add_constraint(cls, model: pe.ConcreteModel, name, index_sets, rule,doc="",constraintType="Constraint"):
        """ initialization of a variable
        :param model:       pe.ConcreteModel
        :param name:        name of variable
        :param index_sets:   indices and sets by which the variable is indexed
        :param rule:        constraint rule
        :param doc:         docstring of variable
        :param constraintType: either 'Constraint', 'Disjunct','Disjunction'"""
        constraintTypes = ['Constraint', 'Disjunct','Disjunction']
        assert constraintType in constraintTypes,f"Constraint type '{constraintType}' unknown"
        constraintObject = cls.get_component_object()
        if name not in constraintObject.docs.keys():
            indexValues,index_list = cls.getIndexNamesData(index_sets)
            if constraintType == "Constraint":
                constraintClass = pe.Constraint
            elif constraintType == "Disjunct":
                constraintClass = pgdp.Disjunct
            else:
                constraintClass = pgdp.Disjunction
            constraint = constraintClass(
                indexValues,
                rule=rule,
                doc=doc
            )
            model.add_component(name, constraint)
            # save constraint doc
            constraintObject.docs[name] = cls.compileDocString(doc,index_list,name)
        else:
            logging.warning(f"{constraintType} {name} already added. Can only be added once")
