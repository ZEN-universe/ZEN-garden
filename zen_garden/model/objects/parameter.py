"""===========================================================================================================================================================================
Title:          ZEN-GARDEN
Created:        July-2022
Authors:        Jacob Mannhardt (jmannhardt@ethz.ch)
Organization:   Laboratory of Reliability and Risk Engineering, ETH Zurich

Description:    Class containing parameters. This is a proxy for pyomo parameters, since the construction of parameters has a significant overhead.
==========================================================================================================================================================================="""
import logging
import numpy as np
import pandas as pd


class Parameter:
    # initialize parameter object
    parameterObject = None

    def __init__(self):
        """ initialization of the parameter object """
        self.minParameterValue  = {"name":None,"value":None}
        self.maxParameterValue  = {"name":None,"value":None}
        self.parameterList      = []
        self.docs               = {}
        # set parameter object
        Parameter.setParameterObject(self)

    @classmethod
    def addParameter(cls,name, data, doc):
        """ initialization of a parameter
        :param name: name of parameter
        :param data: non default data of parameter
        :param doc: docstring of parameter """
        parameterObject = cls.getParameterObject()
        if not hasattr(parameterObject,name):
            # save if highest or lowest value
            cls.saveMinMax(parameterObject,data,name)
            # convert to dict
            data = cls.convertToDict(data)
            # set parameter
            setattr(parameterObject, name, data)
            # save additional parameters
            parameterObject.parameterList.append(name)
            parameterObject.docs[name] = doc
        else:
            logging.warning(f"Parameter {name} already added. Can only be added once")

    @classmethod
    def setParameterObject(cls,parameterObject):
        """ sets parameter object """
        cls.parameterObject = parameterObject

    @classmethod
    def getParameterObject(cls):
        """ returns parameter object """
        return cls.parameterObject

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
