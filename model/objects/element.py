"""===========================================================================================================================================================================
Title:          ENERGY-CARBON OPTIMIZATION PLATFORM
Created:        October-2021
Authors:        Alissa Ganter (aganter@ethz.ch)
                Jacob Mannhardt (jmannhardt@ethz.ch)
Organization:   Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:    Class defining a standard Element. Contains methods to add parameters, variables and constraints to the
                optimization problem. Parent class of the Carrier and Technology classes .The class takes the abstract
                optimization model as an input.
==========================================================================================================================================================================="""
import logging
import pyomo.environ as pe
import pandas as pd
import os
from preprocess.functions.extract_input_data import DataInput
from model.objects.energy_system import EnergySystem

class Element:
    # empty list of elements
    listOfElements = []

    def __init__(self,element):
        """ initialization of an element
        :param element: element that is added to the model"""
        # set attributes
        self.name = element
        # create DataInput object
        self.dataInput = DataInput(EnergySystem.getSystem(),EnergySystem.getAnalysis(),EnergySystem.getSolver(), EnergySystem.getEnergySystem())
        # add element to list
        Element.addElement(self)
        
    ### --- classmethods --- ###
    # setter/getter classmethods
    @classmethod
    def addElement(cls,element):
        """ add element to element list. Inherited by child classes.
        :param element: new element that is to be added to the list """
        cls.listOfElements.append(element)

    @classmethod
    def getAllElements(cls):
        """ get all elements in class. Inherited by child classes.
        :return cls.listOfElements: list of elements in this class """
        return cls.listOfElements

    @classmethod
    def getAllNamesOfElements(cls):
        """ get all names of elements in class. Inherited by child classes.
        :return namesOfElements: list of names of elements in this class """
        _elementsInClass = cls.getAllElements()
        namesOfElements = []
        for _element in _elementsInClass:
            namesOfElements.append(_element.name)
        return namesOfElements
        
    @classmethod
    def getElement(cls,name:str):
        """ get single element in class by name. Inherited by child classes.
        :param name: name of element
        :return element: return element whose name is matched """
        for _element in cls.listOfElements:
            if _element.name == name:
                return _element
        return None

    @classmethod
    def getAllSubclasses(cls):
        """ get all subclasses (child classes) of cls 
        :return subclasses: subclasses of cls """
        return cls.__subclasses__()

    @classmethod
    def getAttributeOfAllElements(cls,attributeName:str):
        """ get attribute values of all elements in this class 
        :param attributeName: str name of attribute
        :return dictOfAttributes: returns dict of attribute values """
        _classElements = cls.getAllElements()
        dictOfAttributes = {}
        for _element in _classElements:
            if hasattr(_element,attributeName):
                _attribute = getattr(_element,attributeName)
                if isinstance(_attribute,dict):
                    # if attribute is dict
                    for _key in _attribute:
                        if isinstance(_key,tuple):
                            dictOfAttributes[(_element.name,)+_key] = _attribute[_key]
                        else:
                            dictOfAttributes[(_element.name,_key)] = _attribute[_key]
                else:
                    dictOfAttributes[_element.name] = _attribute

        return dictOfAttributes

    ### --- classmethods to construct sets, parameters, variables, and constraints, that correspond to Element --- ###
    # Here, after defining EnergySystem-specific components, the components of the other classes are constructed
    @classmethod
    def constructModelComponents(cls):
        """ constructs the model components of the class <Element> """
        # construct pe.Sets
        cls.constructSets()
        # construct pe.Params
        cls.constructParams()
        # construct pe.Vars
        cls.constructVars()
        # construct pe.Constraints
        cls.constructConstraints()
        # construct pe.Objective
        EnergySystem.constructObjective()

    @classmethod
    def constructSets(cls):
        """ constructs the pe.Sets of the class <Element> """
        # construct pe.Sets of energy system
        EnergySystem.constructSets()
        # construct pe.Sets of the child classes
        for subclass in cls.getAllSubclasses():
            subclass.constructSets()

    @classmethod
    def constructParams(cls):
        """ constructs the pe.Params of the class <Element> """
        # construct pe.Params of energy system
        EnergySystem.constructParams()
        # construct pe.Params of the child classes
        for subclass in cls.getAllSubclasses():
            subclass.constructParams()

    @classmethod
    def constructVars(cls):
        """ constructs the pe.Vars of the class <Element> """
        # construct pe.Vars of energy system
        EnergySystem.constructVars()
        # construct pe.Vars of the child classes
        for subclass in cls.getAllSubclasses():
            subclass.constructVars()

    @classmethod
    def constructConstraints(cls):
        """ constructs the pe.Constraints of the class <Element> """
        # construct pe.Constraints of energy system
        EnergySystem.constructConstraints()
        # construct pe.Constraints of the child classes
        for subclass in cls.getAllSubclasses():
            subclass.constructConstraints()
