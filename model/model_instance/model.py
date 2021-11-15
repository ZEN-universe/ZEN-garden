"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Labratory of Risk and Reliability Engineering, ETH Zurich

Description:  Class defining the abstract optimization model.
              The class takes as inputs the properties of the optimization problem. The properties are saved in the
              dictionaries analysis and system which are passed to the class. After initializing the abstract model, the
              class adds carriers and technologies to the abstract model and returns the abstract optimization model.
              The class also includes a method to solve the optimization problem.
==========================================================================================================================================================================="""
import logging
import pyomo.environ as pe
from pyomo.opt import SolverStatus, TerminationCondition
from model.model_instance.objects.carrier import Carrier
from model.model_instance.objects.production_technology import ProductionTechnology
from model.model_instance.objects.transport_technology import TransportTechnology

class Model:

    def __init__(self, analysis, system):
        """
        create Pyomo Abstract Model
        :param analysis: dictionary defining the analysis framework
        :param system: dictionary defining the system
        """
        self.analysis = analysis
        self.system = system

        self.model = pe.AbstractModel()
        self.addSets()
        self.addElements()
        self.addObjectiveFunction()

    def addSets(self):
        """
        This method sets up the sets of the optimization problem.
        Some sets are initialized with default values, if the value is not specified by the input data
        """

        ## FYI
        #(1) model.ct = Set(model.t)  # model.ct is “dictionary” of sets, i.e., model.ct[i] = Set() for all i in model.t
        #(2) model.ct = Set(within=model.t)  # model.ct is a subset of model.t, Pyomo will do the verification of this
        #(3) model.i = Set(initialize=model.t)  # makes a copy of whatever is in model.t during the time of construction
        #(4) model.i = SetOf(model.t)  # references whatever is in model.t at runtime (alias)
        # 'setTransport': 'Set of all transport technologies. Subset: setTechnologies'
        # 'setProduction': 'Set of all production technologies. Subset: setTechnologies'

        sets = {'setCarriers':      'Set of carriers',     # Subsets: setInputCarriers, setOutputCarriers
                'setTechnologies':  'Set of technologies', # Subsets: setTransportTechnologies, setProductionTechnologies
                'setTimeSteps':     'Set of time-steps',
                'setNodes':         'Set of nodes'}

        for set, setProperties in sets.items():
            peSet = pe.Set(doc=setProperties)
            setattr(self.model, set, peSet)

    def addElements(self):
        """
        This method sets up the parameters, variables and constraints of the carriers of the optimization problem.
        :param analysis: dictionary defining the analysis framework
        :param system: dictionary defining the system
        """
        # TODO create list of carrier types, only add relevant types
        # TODO to create list of carrier tpyes (e.g. general, CO2,...) write a function getCarrierTypes

        # add carrier parameters, variables, and constraints
        Carrier(self)
        # add technology parameters, variables, and constraints
        if self.system['setProduction']:
            ProductionTechnology(self)
        if self.system['setTransport']:
            TransportTechnology(self)
        if self.system['setStorage']:
            print("Storage Technologies are not yet implemented")

        #TODO: decide if mass balance should be added here instead of wihtin Carrier??

    def addObjectiveFunction(self):
        """ add objective function to abstract optimization model"""

        objFunc  = self.analysis['objective']
        objSense = self.analysis['sense']
        objRule  = 'objective' + objFunc + 'Rule'
        peObj    = pe.Objective(rule =  getattr(self, objRule),
                                sense = getattr(pe,   objSense))
        setattr(self.model, objFunc, peObj)
    # %% CONSTRAINTS
    def objectiveTotalCostRule(model):
        """
        :return:
        """
        # carrier
        carrierCost = sum(sum(sum(model.importCarrier[carrier, node, time] * model.price[carrier, node, time]
                                for time in model.setTimeSteps)
                            for node in model.setNodes)
                        for carrier in model.setCarriersIn)

        # production and storage techs
        installCost = 0
        for techType in ['Production', 'Storage']:
            if hassattr(model, f'set{techType}'):
                installCost += sum(sum(sum(model.installProductionTech[tech, node, time]
                                           for time in model.setTimeSteps)
                                        for node in model.setNodes)
                                    for tech in getattr(model, f'set{techType}'))

        # transport techs
        if hassattr(model, 'setTransport'):
            installCost += sum(sum(sum(sum(model.installProductionTech[tech, node, aliasNode, time]
                                            for time in model.setTimeSteps)
                                        for node in model.setNodes)
                                    for aliasNode in model.setAliasNodes)
                                for tech in model.setTransport)

        return(carrierCost + installCost)

    def objectiveCarbonEmissionsRule(self):
        """
        :return:
        """
        # TODO implement objective functions for emissions

    def solve(self, solver, pyoDict):
        """
        create model instance by assigning parameter values and instantiating the sets
        :param data: dictionary containing the input data
        :return:
        """

        solverName = solver['name']
        del solver['name']
        solverOptions = solver

        logging.info("Create model instance")
        try:
            self.instance = self.model.create_instance(data=pyoDict)
        except:
            raise ValueError("Please provide pyoDict with input data.")

        logging.info(f"Solve model instance using {solverName}")
        self.opt = pe.SolverFactory(solverName, options=solverOptions)
        self.results = self.opt.solve(self.instance, tee=True)

        # TODO save and evaluate results

