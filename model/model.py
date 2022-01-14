"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Class defining the Concrete optimization model.
              The class takes as inputs the properties of the optimization problem. The properties are saved in the
              dictionaries analysis and system which are passed to the class. After initializing the Concrete model, the
              class adds carriers and technologies to the Concrete model and returns the Concrete optimization model.
              The class also includes a method to solve the optimization problem.
==========================================================================================================================================================================="""
import logging
import pyomo.environ as pe
from pyomo.opt import SolverStatus, TerminationCondition
from model.objects.element import Element
# the order of the following classes defines the order in which they are constructed. Keep this way
from model.objects.technology.conversion_technology import ConversionTechnology
from model.objects.technology.transport_technology import TransportTechnology
from model.objects.carrier import Carrier

class Model():

    def __init__(self, analysis, system, pyoDict):
        """create Pyomo Concrete Model
        :param analysis: dictionary defining the analysis framework
        :param system: dictionary defining the system
        :param pyoDict: input dictionary of optimization """
        self.analysis = analysis
        self.system = system
        self.pyoDict = pyoDict
        self.model = pe.ConcreteModel()
        # set optimization attributes (the three set above) to class <Element>
        Element.setOptimizationAttributes(analysis, system, pyoDict,self.model)
        # add Elements to optimization
        self.addElements()
        # define and construct components of self.model
        Element.defineModelComponents()

    def addElements(self):
        """This method sets up the parameters, variables and constraints of the carriers of the optimization problem.
        :param analysis: dictionary defining the analysis framework
        :param system: dictionary defining the system"""

        # add carrier parameters, variables, and constraints
        for carrier in self.system["setCarriers"]:
            Carrier(self,carrier)
        # add technology parameters, variables, and constraints
        for conversionTech in self.system['setConversionTechnologies']:
            ConversionTechnology(self, conversionTech)
        for transportTech in self.system['setTransportTechnologies']:
            TransportTechnology(self, transportTech)
        # TODO implement storage technologies
        # for storageTech in self.system['setStorageTechnologies']:
        #     print("Storage Technologies are not yet implemented")

    def solve(self, solver, pyoDict):
        """Create model instance by assigning parameter values and instantiating the sets
        :param solver: dictionary containing the solver settings
        :param pyoDict: dictionary containing the input data"""

        solverName = solver['name']
        solverOptions = solver.copy()
        solverOptions.pop('name')

        logging.info(f"Solve model instance using {solverName}")
        self.opt = pe.SolverFactory(solverName, options=solverOptions)
        self.results = self.opt.solve(self.model, tee=True, logfile=solver['logfile'])
        self.model.solutions.load_from(self.results)
        a=1



