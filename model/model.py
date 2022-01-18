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
import os
from pyomo.opt import SolverStatus, TerminationCondition
from model.objects.element import Element
# the order of the following classes defines the order in which they are constructed. Keep this way
from model.objects.technology.conversion_technology import ConversionTechnology
from model.objects.technology.transport_technology import TransportTechnology
from model.objects.carrier import Carrier

class Model():

    def __init__(self, analysis, system, paths):
        """create Pyomo Concrete Model
        :param analysis: dictionary defining the analysis framework
        :param system: dictionary defining the system"""
        self.analysis = analysis
        self.system = system
        self.paths = paths
        self.model = pe.ConcreteModel()
        # set optimization attributes (the three set above) to class <Element>
        Element.setOptimizationAttributes(analysis, system,paths,self.model)
        # add Elements to optimization
        self.addElements()
        # calculate and store input data
        self.storeInputDataInAllElements()
        # define and construct components of self.model
        Element.defineModelComponents()

    def addElements(self):
        """This method sets up the parameters, variables and constraints of the carriers of the optimization problem.
        :param analysis: dictionary defining the analysis framework
        :param system: dictionary defining the system"""

        # add element to define system
        Element("grid")
        # add carrier 
        for carrier in self.system["setCarriers"]:
            Carrier(carrier)
        # add technology 
        for conversionTech in self.system['setConversionTechnologies']:
            ConversionTechnology(conversionTech)
        for transportTech in self.system['setTransportTechnologies']:
            TransportTechnology(transportTech)
        # TODO implement storage technologies
        # for storageTech in self.system['setStorageTechnologies']:
        #     print("Storage Technologies are not yet implemented")
    
    def storeInputDataInAllElements(self):
        """This method iterates through all elements and retrieves the input data. The data is stored in the attributes of the class-specific elements """
        allElements = Element.getAllElements()
        for element in allElements:
            element.storeInputData()

    def solve(self, solver):
        """Create model instance by assigning parameter values and instantiating the sets
        :param solver: dictionary containing the solver settings """

        solverName = solver['name']
        solverOptions = solver.copy()
        solverOptions.pop('name')

        logging.info(f"Solve model instance using {solverName}")
        solver_parameters = f"ResultFile={os.path.dirname(solver['logfile'])}//model.ilp" # write an ILP file to print the IIS if infeasible (gives Warning: unable to write requested result file './/outputs//logs//model.ilp' if feasible)
        self.opt = pe.SolverFactory(solverName, options=solverOptions)
        self.opt.set_instance(self.model,symbolic_solver_labels =True)
        self.results = self.opt.solve(tee=True, logfile=solver['logfile'],options_string=solver_parameters)
        self.model.solutions.load_from(self.results)

        a=1



