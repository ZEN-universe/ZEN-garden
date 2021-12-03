"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Class defining a standard element. Contains methods to add parameters, variables and constraints to the
              optimization problem. Parent class of the Carrier and Technology classes .The class takes the abstract
              optimization model as an input.
==========================================================================================================================================================================="""
import pyomo.environ as pe

class Element:

    subsets     = dict()
    params      = dict()
    variables   = dict()
    constraints = dict()

    def __init__(self,object):
        """ initialization of an element
        :param model: object of the abstract optimization model"""

        self.model = object.model
        self.analysis = object.analysis
        self.system = object.system

    def getProperties(self, properties):
        """get properties (doc, dimensions, domain)
        :param  properties:      parameter, variable or constraint properties
        :return doc:             documentation
        :return dimensions:      dimensions of the parameter, variable, constraint
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
            if 'Domain' in propertyName:
                domain = propertyName.split(':')[-1].strip()
                domain = getattr(pe, domain)

        return doc, dimensions, domain

    def addSubsets(self, subsets):
        """
        add sets or subsets to model
        :param sets: dictionary containing set names and properties"""

        for setName, setProperty in subsets.items():
            if 'Subset' in setProperty:
                subsetOf = setProperty.split(':')[-1].strip()
                peSet  = pe.Set(within= getattr(self.model, subsetOf), doc=setProperty)
            if 'Alias' in setName:
                aliasOf   = setName.replace('Alias','')
                peSet = pe.SetOf(getattr(self.model, aliasOf))
            else:
                peSet = pe.Set(doc=setProperty)

            setattr(self.model, setName, peSet)

    def addParams(self, params):
        """add parameters to model
        :param params: dictionary containing param names and properties"""

        for param, paramProperties in params.items():
            if not 'Dimensions' in paramProperties:
                raise ValueError('Dimensions of parameter {0} are undefined'.format(param))

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

    def addConstr(self, constraints):
        """add constraints to model
        :param constraints: dictionary containing var names and properties"""

        for constr, constrProperties in constraints.items():
            if not 'Dimensions' in constrProperties:
                raise ValueError('Dimensions of constraint {0} are undefined'.format(constr))
                
            _,dimensions,_ = self.getProperties(constrProperties)

            peConstr   = pe.Constraint(*dimensions, rule=getattr(self, f'{constr}Rule')) 

            setattr(self.model, constr, peConstr)
                      
