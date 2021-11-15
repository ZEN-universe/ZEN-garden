"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Labratory of Risk and Reliability Engineering, ETH Zurich

Description:  Class defining a standard element. Contains methods to add parameters, variables and constraints to the
              optimization problem. Parent class of the Carrier and Technology classes .The class takes the abstract
              optimization model as an input.
==========================================================================================================================================================================="""
import pyomo.environ as pe

class Element:

    subsets     = dict()
    params      = dict()
    vars        = dict()
    constraints = dict()

    def __init__(self,object):
        """initialization of an element
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

        for property in properties.split('.')[1:]:
            if 'Dimensions' in property:
                dimensions = property.split('.')[-1].split(':')[-1]
                if ',' in dimensions:
                    dimensions = dimensions.split(',')
                    dimensions = [getattr(self.model, dim.strip()) for dim in dimensions]
            if 'Domain' in property:
                domain = property.split(':')[-1].strip()
                domain = getattr(pe, domain)

        return doc, dimensions, domain

    def addSubsets(self, subsets):
        """add sets to model
        :param sets: dictionary containing set names and properties"""

        for set, setProperties in subsets.items():
            if 'Subset' in setProperties:
                subsetOf = setProperties.split(':')[-1].strip()
                peSet  = pe.Set(within= getattr(self.model, subsetOf), doc=setProperties)
            if 'Alias' in set:
                aliasOf   = set.replace('Alias','')
                peSet = pe.SetOf(getattr(self.model, aliasOf))
            else:
                peSet = pe.Set(doc=setProperties)

            setattr(self.model, set, peSet)

    def addParams(self, params):
        """add parameter to model
        :param params: dictionary containing param names and properties"""

        for param, paramProperties in params.items():
            if not 'Dimensions' in paramProperties:
                raise ValueError('Dimensions of parameter {0} are undefined'.format(param))

            doc, dimensions, _ = self.getProperties(paramProperties)
            peParam            = pe.Param(*dimensions, doc=doc)

            setattr(self.model, param, peParam)

    def addVars(self, vars):
        """add variable to model
        :param vars: dictionary containing var names and properties"""

        for var, varProperties in vars.items():
            if not 'Dimensions' in varProperties:
                raise ValueError('Dimensions of variable {0} are undefined'.format(var))
            if not 'Domain' in varProperties:
                raise ValueError('Domain of variable {0} are undefined'.format(var))

            doc, dimensions, domain  = self.getProperties(varProperties)
            peVar                    = pe.Var(*dimensions, within=domain, doc=doc)

            setattr(self.model, var, peVar)

    def addConstr(self, constraints):
        """add constraint to model
        :param constraints: dictionary containing var names and properties"""

        for constr, constrProperties in constraints.items():
            if not 'Dimensions' in constrProperties:
                raise ValueError('Dimensions of constraint {0} are undefined'.format(constr))

            _,dimensions,_ = self.getProperties(constrProperties)
            peConstr   = pe.Constraint(*dimensions, rule=getattr(self, '{0}Rule'.format(constr))) #check if constraint can be added like that. Otherwise it might be necessary to pass the constraint in the function

            setattr(self.model, constr, peConstr)

    # def addMassBalance(self, system):
    #     """ """
    #
    #     massBalance = dict()
    #     setTechnologies = ['setProduction', 'setTransport', 'setStorage']
    #
    #     if system['setProduction']:
    #         massBalanceIn = massBalanceCarrier + 'model.inputProductionTech[carrier, node, time]'
    #         massBalanceOut = massBalanceCarrier + '- model.outputProductionTech[carrier, node, time]'
    #         massBalanceInOut = massBalanceCarrier +'model.inputProductionTech[carrier, node, time]' + '- model.outputProductionTech[carrier, node, time]'
    #     if system['setTransport']:
    #         massBalanceIn = massBalanceIn + 'sum(model.flowTransportTech[carrier, aliasNode, node, time] - model.flowTransportTech[carrier, node, aliasNode, time] for aliasNode in setAliasNodes)'
    #         massBalanceOut = massBalanceOut + 'sum(model.flowTransportTech[carrier, aliasNode, node, time] - model.flowTransportTech[carrier, node, aliasNode, time] for aliasNode in setAliasNodes)'
    #         massBalanceInOut = massBalanceInOut + 'sum(model.flowTransportTech[carrier, aliasNode, node, time] - model.flowTransportTech[carrier, node, aliasNode, time] for aliasNode in setAliasNodes)'
    #     if system['setStorage']:
    #         #TODO add storages
    #         pass
    #
    #
    #     # formulate different mass-balances
    #     functionHead = 'def constraintMassBalanceRule(carrier, node, time):'
    #     doc = 'nodal mass balance for each time step. Dimensions: setCarriers, setNodes, setTimeSteps'
    #     massBalanceInOut = f'if carrier in setCarrierIn and in setCarrierOut: return(massBalanceInOut)'
    #     massBalanceIn = f'elif carrier in setCarrierIn: return(massBalanceIn)'
    #     massBalanceOut = f'elif carrier in setCarrierIn: return(massBalanceOut)'
    #
    #     # assemble mass-balance
    #     massBalance = functionHead + doc + massBalanceInOut + massBalanceIn + massBalanceOut
    #     _, dimensions = self.getProperties(self, constrProperties)
    #     peConstr = pe.Constraint(*dimensions, rule=getattr(self, exec(massBalance))  # check if constraint can be added like that. Otherwise it might be necessary to pass the constraint in the function
    #
    #     setattr(self.model, 'ConstraintMassBalance', peConstr)