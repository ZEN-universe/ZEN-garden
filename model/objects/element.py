"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Class defining a standard element. Contains methods to add parameters, variables and constraints to the
              optimization problem. Parent class of the Carrier and Technology classes .The class takes the abstract
              optimization model as an input.
==========================================================================================================================================================================="""
import logging
import pyomo.environ as pe

class Element:
    # empty list of elements
    listOfElements = []

    subsets     = dict()
    params      = dict()
    variables   = dict()
    constraints = dict()

    def __init__(self,object,element):
        """ initialization of an element
        :param object: object of the abstract optimization model
        :param element: element that is added to the model"""

        self.model = object.model
        self.analysis = object.analysis
        self.system = object.system
        # set attributes
        self.name = element
        
        # add element to list
        Element.addElement(self)

    def getProperties(self, properties):
        """get properties (doc, dimensions, domain)
        :param  properties:      parameter, variable or constraint properties
        :return doc:             documentation
        :return dimensions:      dimensions of the parameter, variable, constraint
                                 (note: in case of constraints additional information can be passed with the keyword "Other")
        :return domain:          variable domain, empty for parameters and constraints"""

        doc        = properties
        dimensions = []
        domain     = []

        for propertyName in properties.split('.')[1:]:
            if 'Dimensions' in propertyName:
                dimensions = propertyName.split('.')[-1].split(':')[-1]
                if ',' in dimensions:
                    dimensions = dimensions.split(',')
                    dimensions = [getattr(self.model, dim.strip()) for dim in dimensions]
                else:
                    dimensions = [getattr(self.model, dimensions.strip())]
            elif 'Domain' in propertyName:
                domain = propertyName.split(':')[-1].strip()
                domain = getattr(pe, domain)

        return doc, dimensions, domain

    def addSets(self, subsets):
        """add sets or subsets to model
        :param sets: dictionary containing set names and properties"""

        for setName, setProperty in subsets.items():
            if 'Subset' in setProperty:
                subsetOf = setProperty.split(':')[-1].strip()
                peSet  = pe.Set(within= getattr(self.model, subsetOf), doc=setProperty)
                setattr(self.model, setName, peSet)
            if 'Alias' in setName:
                aliasOf   = setName.replace('Alias','')
                peSet = pe.SetOf(getattr(self.model, aliasOf))
            elif 'Index' in setName:
                index = setProperty.split(':').strip()[-1]
                peSet = pe.Set(index, doc=setProperty)
            elif 'Dimension' in setName:
                dimension = len(setProperty.split(':')[-1].split(','))
                peSet = pe.Set(dimen = dimension)
            elif 'Rule' in setName:
                rule = setProperty.split(':').strip()[-1]
                peSet = pe.Set(initialize=rule, doc=setProperty)
            else:
                peSet = pe.Set(doc=setProperty)

            setattr(self.model, setName, peSet)

    def addParams(self, params):
        """add parameters to model
        :param params: dictionary containing param names and properties"""

        for param, paramProperties in params.items():

            doc, dimensions, _ = self.getProperties(paramProperties)
            peParam            = pe.Param(*dimensions, doc=doc)
            
            setattr(self.model, param, peParam)

    def addVars(self, variables):
        """add variables to model
        :param variables: dictionary containing var names and properties"""

        for var, varProperties in variables.items():
            if not 'Dimensions' in varProperties:
                raise ValueError('Dimensions of variable {0} are undefined'.format(var))
            if not 'Domain' in varProperties:
                raise ValueError('Domain of variable {0} are undefined'.format(var))

            doc, dimensions, domain  = self.getProperties(varProperties)
            peVar                    = pe.Var(*dimensions, within=domain, doc=doc)

            setattr(self.model, var, peVar)

    def addConstr(self, constraints, replace = ['', ''], passValues = None):
        """add constraints to model
        :param constraints: dictionary containing var names and properties"""

        for constr, constrProperties in constraints.items():
            if not 'Dimensions' in constrProperties:
                raise ValueError('Dimensions of constraint {0} are undefined'.format(constr))

            doc, dimensions,_ = self.getProperties(constrProperties)
            if passValues:
                dimensions.insert(0, pe.Set(initialize=passValues))

            peConstr   = pe.Constraint(*dimensions, rule=getattr(self, f'constraint{constr.replace(*replace)}Rule'), doc=doc)

            setattr(self.model, f'constraint{constr}', peConstr)
                      
    # classmethods
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
    def getElement(cls,name:str):
        """ get single element in class by name. Inherited by child classes.
        :param name: name of element
        :return element: return element whose name is matched """
        for element in cls.listOfElements:
            if element.name == name:
                return element
        return None