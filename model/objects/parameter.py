"""===========================================================================================================================================================================
Title:          ZEN-GARDEN
Created:        July-2022
Authors:        Jacob Mannhardt (jmannhardt@ethz.ch)
Organization:   Laboratory of Reliability and Risk Engineering, ETH Zurich

Description:    Class containing parameters. This is a proxy for pyomo parameters, since the construction of parameters has a significant overhead.
==========================================================================================================================================================================="""
import logging
import numpy as np

class Parameter:
    # initialize parameter object
    parameterObject = None

    def __init__(self):
        """ initialization of the parameter object """
        if not Parameter.getParameterObject():
            self.minParameterValue  = {"name":None,"value":None}
            self.maxParameterValue  = {"name":None,"value":None}
            self.numParameter       = 0
            self.docs               = {}
            # set parameter object
            Parameter.setParameterObject(self)
        else:
            logging.warning("Parameter object already initialized. Skipped")

    @classmethod
    def addParameter(cls,name, data, doc):
        """ initialization of a parameter
        :param name: name of parameter
        :param data: non default data of parameter
        :param doc: docstring of parameter """
        parameterObject = cls.getParameterObject()
        if not hasattr(parameterObject,name):
            # set parameter
            setattr(parameterObject,name,data)
            # save if highest or lowest value
            _abs    = data.abs()
            _abs    = _abs[(_abs != 0) & (_abs != np.inf)]
            if not _abs.empty:
                _idxmax     = _abs.argmax()
                _valmax     = _abs.max()
                _idxmin     = _abs.argmin()
                _valmin     = _abs.min()
                if not parameterObject.maxParameterValue["name"]:
                    parameterObject.maxParameterValue["name"]   = name + "_" + str(_idxmax)
                    parameterObject.maxParameterValue["value"]  = _valmax
                    parameterObject.minParameterValue["name"]   = name + "_" + str(_idxmin)
                    parameterObject.minParameterValue["value"]  = _valmin
                else:
                    if _valmax > parameterObject.maxParameterValue["value"]:
                        parameterObject.maxParameterValue["name"]   = name + "_" + str(_idxmax)
                        parameterObject.maxParameterValue["value"]  = _valmax
                    if _valmin > parameterObject.minParameterValue["value"]:
                        parameterObject.minParameterValue["name"]   = name + "_" + str(_idxmin)
                        parameterObject.minParameterValue["value"]  = _valmin
            # save additional parameters
            parameterObject.numParameter        += 1
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
