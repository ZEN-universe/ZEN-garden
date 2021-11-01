"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Labratory of Risk and Reliability Engineering, ETH Zurich

Description:  Method to extract the properties of a parameter and add the parameter to the model.
              The method takes as inputs the model, the parameter name and parameter properties. It returns the model
              after the parameter was added.
==========================================================================================================================================================================="""
import pyomo.environ as pe

class Element:

    sets = dict()
    params = dict()
    vars = dict()
    constraints = dict()

    def __init__(self,model):
        """initialization of an element
        :param model: object of the abstract optimization mode"""

        self.model = model

    def getProperties(self, properties):
        """get properties (doc, dimensions, domain)
        :param  properties:      parameter, variable or constraint properties
        :return doc:             documentation
        :return dimensions:      dimensions of the parameter, variable, constraint
        :return domain:          variable domain, empty for parameters and constraints"""

        doc        = properties
        dimensions = []
        domain     = []

        for property, value in properties.split('.')[1:]:
            if property == 'Dimensions':
                dimensions = value.split('.')[-1].split(':')[-1]
                if ',' in dimensions:
                    dimensions = dimensions.split(',')
                dimensions = [getattr(self.model, dim) for dim in dimensions]

        if property == 'Domain':
            domain = value.split(':')[-1]

        return doc, dimensions, domain

    def addSets(self, sets, data):
        """add sets to model
        :param sets: dictionary containing set names and properties"""

        for set, setProperties in sets.items():
            if 'Subset' in setProperties:
                subsetOf = setProperties.split(':')[-1]
                peSet  = pe.Set(within= getattr(self.model, subsetOf), doc=setProperties)
            if 'Alias' in set:
                aliasOf   = set.remove('Alias')
                peSet = pe.SetOf(getattr(self.model, aliasOf))
            else:
                peSet = pe.Set(doc=setProperties)

            setattr(self.model, set, peSet)

    def addParam(self, params):
        """add parameter to model
        :param params: dictionary containing param names and properties"""

        for param, paramProperties in params.items():
            if not 'Dimensions' in paramProperties:
                raise ValueError('Dimensions of parameter {0} are undefined'.format(param))

            doc, dimensions, _ = self.getProperties(self, paramProperties)
            peParam            = pe.Param(*dimensions, doc=doc)

            setattr(self.model, param, peParam)

    def addVar(self, vars):
        """add variable to model
        :param vars: dictionary containing var names and properties"""

        for var, varProperties in vars.items():
            if not 'Dimensions' in varProperties:
                raise ValueError('Dimensions of variable {0} are undefined'.format(var))
            if not 'Domain' in varProperties:
                raise ValueError('Domain of variable {0} are undefined'.format(var))

            doc, dimensions, domain  = self.getProperties(self, varProperties)
            dimensions               = [getattr(self.model, dim) for dim in dimensions]
            peVar                    = pe.Var(*dimensions, within=getattr(pe, domain), doc=doc)

            setattr(self.model, var, peVar)

    def addConstr(self, constraints):
        """add constraint to model
        :param constraints: dictionary containing var names and properties"""

        for constr, constrProperties in constraints.items():
            if not 'Dimensions' in constrProperties:
                raise ValueError('Dimensions of constraint {0} are undefined'.format(constr))

            _,dimensions,_ = self.getProperties(self, constrProperties)
            peConstr   = pe.Constraint(*dimensions, rule=exec('{0}_rule'.format(constr)))

            setattr(self.model, constr, peConstr)

