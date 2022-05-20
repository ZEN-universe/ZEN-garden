"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      October-2021
Authors:      Jacob Mannhardt (jmannhardt@ethz.ch)
              Alissa Ganter (aganter@ethz.ch)
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
import sys
import pandas as pd
import numpy as np
# import elements of the optimization problem
# technology and carrier classes from technology and carrier directory, respectively
from model.objects.element                              import Element
from model.objects.technology                           import *
from model.objects.carrier                              import *
from model.objects.energy_system                        import EnergySystem
from preprocess.functions.time_series_aggregation       import TimeSeriesAggregation

class OptimizationSetup():

    def __init__(self, analysis, prepare):
        """setup Pyomo Concrete Model
        :param analysis: dictionary defining the analysis framework
        :param system: dictionary defining the system
        :param paths: dictionary defining the paths of the model
        :param solver: dictionary defining the solver"""
        self.prepare    = prepare
        self.analysis   = analysis
        self.system     = prepare.system
        self.paths      = prepare.paths
        self.solver     = prepare.solver
        # step of optimization horizon
        self.stepHorizon = 0
        # set optimization attributes (the five set above) to class <EnergySystem>
        EnergySystem.setOptimizationAttributes(analysis, self.system, self.paths, self.solver)
        # add Elements to optimization
        self.addElements()

    def addElements(self):
        """This method sets up the parameters, variables and constraints of the carriers of the optimization problem.
        :param analysis: dictionary defining the analysis framework
        :param system: dictionary defining the system"""
        logging.info("\n--- Add elements to model--- \n")
        # add energy system to define system
        EnergySystem("energySystem")

        for elementName in EnergySystem.getElementList():
            elementClass = EnergySystem.dictElementClasses[elementName]
            elementName  = elementClass.label
            elementSet   = self.system[elementName]

            # before adding the carriers, get setCarriers and check if carrier data exists
            if elementName == "setCarriers":
                elementSet                 = EnergySystem.getAttribute("setCarriers")
                self.system["setCarriers"] = elementSet
                self.prepare.checkExistingCarrierData(self.system)

            # check if elementSet has a subset and remove subset from elementSet
            if elementName in self.analysis["subsets"].keys():
                elementSubset = []
                for subset in self.analysis["subsets"][elementName]:
                    elementSubset += [item for item in self.system[subset]]
                elementSet = list(set(elementSet) - set(elementSubset))

            # add element class
            for item in elementSet:
                elementClass(item)

        # conduct  time series aggregation
        TimeSeriesAggregation.conductTimeSeriesAggregation()

    def constructOptimizationProblem(self):
        """ constructs the optimization problem """
        # create empty ConcreteModel
        self.model = pe.ConcreteModel()
        EnergySystem.setConcreteModel(self.model)
        # define and construct components of self.model
        Element.constructModelComponents()
        logging.info("Apply Big-M GDP ")
        # add transformation factory so that disjuncts are solved
        pe.TransformationFactory("gdp.bigm").apply_to(self.model)

    def getOptimizationHorizon(self):
        """ returns list of optimization horizon steps """
        energySystem    = EnergySystem.getEnergySystem()
        # save "original" full setTimeStepsYearly
        self.setTimeStepsYearlyFull = energySystem.setTimeStepsYearly
        # if using rolling horizon
        if self.system["useRollingHorizon"]:
            self.yearsInHorizon = self.system["yearsInHorizon"]
            _timeStepsYearly    = energySystem.setTimeStepsYearly
            self.stepsHorizon   = {year: list(range(year,min(year + self.yearsInHorizon,max(_timeStepsYearly)+1))) for year in _timeStepsYearly}
        # if no rolling horizon
        else:
            self.yearsInHorizon = len(energySystem.setTimeStepsYearly)
            self.stepsHorizon   = {0: energySystem.setTimeStepsYearly}
        return list(self.stepsHorizon.keys())

    def overwriteParams(self, scenario, elements):
        """overwrite scenario dependent parameters
        :param scenario: scenario name
        :param elements:   dictionary of elements and scenario dependent parameters"""
        for elementName, params in elements.items():
            if elementName == "EnergySystem":
                element = EnergySystem.getEnergySystem()
                path    = self.paths["setScenarios"]["folder"] # adjust after merge and get input path
            else:
                element = Element.getElement(elementName)
                path    = element.inputPath
            for param in params:
                _oldParam   = getattr(element, param)
                if isinstance(_oldParam, pd.Series) or isinstance(element.carbonEmissionsLimit, pd.DataFrame):
                    _indexNames = _oldParam.index.names
                    _indexSets = [indexSet for indexSet, indexName in element.dataInput.indexNames.items() if indexName in _indexNames]
                    _timeSteps = []
                    if "time" in _indexNames:
                        _timeSteps = list(_oldParam.index.unique("time"))
                    _newParam = element.dataInput.extractInputData(path, param, indexSets=_indexSets, timeSteps=_timeSteps, scenario = f"_{scenario}")
                else:
                    _newParam = element.dataInput.extractAttributeData(path,param, fileName=f"{param}_{scenario}.csv", skipWarning=True)["value"]
                setattr(element, param, _newParam)

    def overwriteTimeIndices(self,stepHorizon):
        """ select subset of time indices, matching the step horizon
        :param stepHorizon: step of the rolling horizon """
        energySystem    = EnergySystem.getEnergySystem()

        if self.system["useRollingHorizon"]:
            _timeStepsYearlyHorizon = self.stepsHorizon[stepHorizon]
            _baseTimeStepsHorizon   = EnergySystem.decodeYearlyTimeSteps(_timeStepsYearlyHorizon)
            # overwrite time steps of each element
            for element in Element.getAllElements():
                element.overwriteTimeSteps(_baseTimeStepsHorizon)
            # overwrite base time steps and yearly base time steps
            energySystem.setBaseTimeSteps       = _baseTimeStepsHorizon.squeeze().tolist()
            energySystem.setTimeStepsYearly     = _timeStepsYearlyHorizon

    def solve(self, solver):
        """Create model instance by assigning parameter values and instantiating the sets
        :param solver: dictionary containing the solver settings """

        solverName          = solver["name"]
        # remove options that are None
        solverOptions       = {key:solver["solverOptions"][key] for key in solver["solverOptions"] if solver["solverOptions"][key] is not None}

        logging.info(f"\n--- Solve model instance using {solverName} ---\n")
        # disable logger temporarily
        logging.disable(logging.WARNING)
        # write an ILP file to print the IIS if infeasible
        #         # (gives Warning: unable to write requested result file ".//outputs//logs//model.ilp" if feasible)
        solver_parameters   = f"ResultFile={os.path.dirname(solver['solverOptions']['logfile'])}//infeasibleModelIIS.ilp"
        self.opt            = pe.SolverFactory(solverName, options=solverOptions)
        self.opt.set_instance(self.model,symbolic_solver_labels=True)
        self.results        = self.opt.solve(tee=solver["verbosity"], logfile=solver["solverOptions"]["logfile"],options_string=solver_parameters)
        # enable logger 
        logging.disable(logging.NOTSET)
        self.model.solutions.load_from(self.results)

    def addNewlyBuiltCapacity(self,stepHorizon):
        """ adds the newly built capacity to the existing capacity
        :param stepHorizon: step of the rolling horizon """
        if self.system["useRollingHorizon"]:
            _builtCapacity  = pd.Series(self.model.builtCapacity.extract_values())
            _capex          = pd.Series(self.model.capex.extract_values())
            _baseTimeSteps  = EnergySystem.decodeYearlyTimeSteps([stepHorizon])
            Technology      = getattr(sys.modules[__name__], "Technology")
            for tech in Technology.getAllElements():
                # new capacity
                _builtCapacityTech = _builtCapacity.loc[tech.name].unstack()
                _capexTech          = _capex.loc[tech.name].unstack()
                tech.addNewlyBuiltCapacityTech(_builtCapacityTech,_capexTech,_baseTimeSteps)