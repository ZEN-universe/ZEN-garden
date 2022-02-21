"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch), Jacob Mannhardt (jmannhardt@ethz.ch)
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
import cProfile, pstats

from model.objects.element import Element
# the order of the following classes defines the order in which they are constructed. Keep this way
from model.objects.technology.conversion_technology import ConversionTechnology
from model.objects.technology.storage_technology import StorageTechnology
from model.objects.technology.transport_technology import TransportTechnology
from model.objects.carrier import Carrier
from model.objects.energy_system import EnergySystem
from preprocess.functions.time_series_aggregation import TimeSeriesAggregation

class OptimizationSetup():

    def __init__(self, analysis, system, paths,solver):
        """setup Pyomo Concrete Model
        :param analysis: dictionary defining the analysis framework
        :param system: dictionary defining the system
        :param paths: dictionary defining the paths of the model
        :param solver: dictionary defining the solver"""
        self.analysis   = analysis
        self.system     = system
        self.paths      = paths
        self.solver     = solver
        self.model      = pe.ConcreteModel()
        # set optimization attributes (the five set above) to class <EnergySystem>
        EnergySystem.setOptimizationAttributes(analysis, system, paths, solver, self.model)
        # add Elements to optimization
        self.addElements()
        # define and construct components of self.model
        # pr = cProfile.Profile()
        # pr.enable()
        Element.constructModelComponents()
        # pr.disable()
        # ps = pstats.Stats(pr).sort_stats("cumtime")
        # ps.print_stats()
        
        # add transformation factory so that disjuncts are solved
        pe.TransformationFactory("gdp.bigm").apply_to(self.model)  

    def addElements(self):
        """This method sets up the parameters, variables and constraints of the carriers of the optimization problem.
        :param analysis: dictionary defining the analysis framework
        :param system: dictionary defining the system"""
        logging.info("\n--- Add elements to model--- \n")
        # add energy system to define system
        EnergySystem("energySystem")
        # add technology 
        for conversionTech in self.system['setConversionTechnologies']:
            ConversionTechnology(conversionTech)
        for transportTech in self.system['setTransportTechnologies']:
            TransportTechnology(transportTech)
        for storageTech in self.system['setStorageTechnologies']:
            StorageTechnology(storageTech)
        # add carrier 
        for carrier in self.system["setCarriers"]:
            Carrier(carrier)
        # conduct  time series aggregation
        TimeSeriesAggregation.conductTimeSeriesAggregation()
        

    def solve(self, solver):
        """Create model instance by assigning parameter values and instantiating the sets
        :param solver: dictionary containing the solver settings """

        solverName = solver['name']
        solverOptions = solver["solverOptions"]

        logging.info(f"\n--- Solve model instance using {solverName} ---\n")
        # disable logger temporarily
        logging.disable(logging.WARNING)
        # write an ILP file to print the IIS if infeasible
        # (gives Warning: unable to write requested result file './/outputs//logs//model.ilp' if feasible)
        solver_parameters = f"ResultFile={os.path.dirname(solver['solverOptions']['logfile'])}//infeasibleModelIIS.ilp"
        self.opt = pe.SolverFactory(solverName, options=solverOptions)
        self.opt.set_instance(self.model,symbolic_solver_labels =True)
        self.results = self.opt.solve(tee=solver['verbosity'], logfile=solver["solverOptions"]["logfile"],options_string=solver_parameters)
        # enable logger 
        logging.disable(logging.NOTSET)
        self.model.solutions.load_from(self.results)