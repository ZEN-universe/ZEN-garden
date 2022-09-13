"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      October-2021
Authors:      Jacob Mannhardt (jmannhardt@ethz.ch)
              Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

Description:  Class defining the Concrete optimization model.
              The class takes as inputs the properties of the optimization problem. The properties are saved in the
              dictionaries analysis and system which are passed to the class. After initializing the Concrete model, the
              class adds carriers and technologies to the Concrete model and returns the Concrete optimization model.
              The class also includes a method to solve the optimization problem.
==========================================================================================================================================================================="""
import logging
import pyomo.environ as pe
from pyomo.core.expr.current import decompose_term
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

    baseScenario      = ""
    baseConfiguration = {}

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
                elementSet = EnergySystem.getAttribute("setCarriers")
                self.system["setCarriers"] = elementSet
                self.prepare.checkExistingCarrierData(self.system)

            # check if elementSet has a subset and remove subset from elementSet
            if elementName in self.analysis["subsets"].keys():
                elementSubset = []
                for subset in self.analysis["subsets"][elementName]:
                    elementSubset += [item for item in self.system[subset]]
                elementSet = list(set(elementSet)-set(elementSubset))

            # add element class
            for item in elementSet:
                elementClass(item)
        if EnergySystem.getSolver()["analyzeNumerics"]:
            EnergySystem.getUnitHandling().recommendBaseUnits(immutableUnit = EnergySystem.getSolver()["immutableUnit"],
                                                              unitExps = EnergySystem.getSolver()["rangeUnitExponents"])
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
        # find smallest and largest coefficient and RHS
        self.analyzeNumerics()

    def getOptimizationHorizon(self):
        """ returns list of optimization horizon steps """
        energySystem    = EnergySystem.getEnergySystem()
        # save "original" full setTimeStepsYearly
        self.setTimeStepsYearlyFull = energySystem.setTimeStepsYearly
        # if using rolling horizon
        if self.system["useRollingHorizon"]:
            self.yearsInHorizon = self.system["yearsInRollingHorizon"]
            _timeStepsYearly    = energySystem.setTimeStepsYearly
            self.stepsHorizon   = {year: list(range(year,min(year + self.yearsInHorizon,max(_timeStepsYearly)+1))) for year in _timeStepsYearly}
        # if no rolling horizon
        else:
            self.yearsInHorizon = len(energySystem.setTimeStepsYearly)
            self.stepsHorizon   = {0: energySystem.setTimeStepsYearly}
        return list(self.stepsHorizon.keys())

    def setBaseConfiguration(self, scenario, elements):
        """set base configuration
        :param scenario: name of base scenario
        :param elements: elements in base scenario """
        self.baseScenario      = scenario
        self.baseConfiguration = elements

    def restoreBaseConfiguration(self, scenario, elements):
        """restore default configuration
        :param scenario: scenario name
        :param elements: dictionary of scenario dependent elements and parameters"""
        if not scenario == self.baseScenario:
            # restore base configuration
            self.overwriteParams(self.baseScenario, self.baseConfiguration)
            # continuously update baseConfiguration so all parameters are reset to their base value after being changed
            for elementName, params in elements.items():
                if elementName not in self.baseConfiguration.keys():
                    self.baseConfiguration[elementName] = params
                else:
                    for param in params:
                        if param not in self.baseConfiguration[elementName]:
                            self.baseConfiguration[elementName].append(param)

    def overwriteParams(self, scenario, elements):
        """overwrite scenario dependent parameters
        :param scenario: scenario name
        :param elements: dictionary of scenario dependent elements and parameters"""
        if scenario == self.baseScenario:
            scenario = ""
        else:
            scenario = "_" + scenario
        # list of parameters with rawTimeSeries
        conductTimeSeriesAggregation = False
        # overwrite scenario dependent parameter values for all elements
        for elementName, params in elements.items():
            if elementName == "EnergySystem":
                element = EnergySystem.getEnergySystem()
            else:
                element = Element.getElement(elementName)
            # overwrite scenario dependent parameters
            for param in params:
                fileName = param
                if type(param) is tuple:
                    fileName, param = param
                # get old param value
                _oldParam   = getattr(element, param)
                _indexNames = _oldParam.index.names
                _indexSets  = [indexSet for indexSet, indexName in element.dataInput.indexNames.items() if indexName in _indexNames]
                _timeSteps  = None
                # set new parameter value
                if hasattr(element, "rawTimeSeries") and param in element.rawTimeSeries.keys():
                    conductTimeSeriesAggregation = True
                    _timeSteps                   = EnergySystem.getEnergySystem().setBaseTimeStepsYearly
                    element.rawTimeSeries[param] = element.dataInput.extractInputData(fileName, indexSets=_indexSets,column=param,timeSteps=_timeSteps,scenario=scenario)
                else:
                    if isinstance(_oldParam, pd.Series) or isinstance(_oldParam, pd.DataFrame):
                        if "time" in _indexNames:
                            _timeSteps = list(_oldParam.index.unique("time"))
                        _newParam  = element.dataInput.extractInputData(fileName,indexSets=_indexSets,timeSteps=_timeSteps,scenario=scenario)
                    #else: _newParam = element.dataInput.extractAttributeData(param,scenario=scenario,skipWarning=True)["value"]
                    setattr(element, param, _newParam)
        # if scenario contains timeSeries dependent params conduct timeSeriesAggregation
        if conductTimeSeriesAggregation:
            TimeSeriesAggregation.conductTimeSeriesAggregation()

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
            _newBaseTimeStepsHorizon            = _baseTimeStepsHorizon.squeeze().tolist()
            if not isinstance(_newBaseTimeStepsHorizon,list):
                _newBaseTimeStepsHorizon        = [_newBaseTimeStepsHorizon]
            energySystem.setBaseTimeSteps       = _newBaseTimeStepsHorizon
            energySystem.setTimeStepsYearly     = _timeStepsYearlyHorizon

    def analyzeNumerics(self):
        """ get largest and smallest matrix coefficients and RHS """
        if EnergySystem.getSolver()["analyzeNumerics"]:
            largestRHS      = [None,0]
            smallestRHS     = [None,np.inf]
            largestCoeff    = [None,0]
            smallestCoeff   = [None,np.inf]

            for cons in self.model.component_objects(pe.Constraint):
                for idx in cons:
                    decoLHS     = decompose_term(cons[idx].expr.args[0])
                    decoRHS     = decompose_term(cons[idx].expr.args[1])
                    decoComb    = []
                    if decoLHS[0]:
                        decoComb += decoLHS[1]
                    if decoRHS[0]:
                        decoComb += decoRHS[1]
                    _RHS = 0
                    for item in decoComb:
                        _abs = abs(item[0])
                        if _abs != 0:
                            if item[1] is not None:
                                if _abs > largestCoeff[1]:
                                    largestCoeff[0] = (cons[idx].name,item[1].name)
                                    largestCoeff[1] = _abs
                                if _abs < smallestCoeff[1]:
                                    smallestCoeff[0] = (cons[idx].name,item[1].name)
                                    smallestCoeff[1] = _abs
                            else:
                                _RHS += item[0]
                    _RHS = abs(_RHS)
                    if _RHS != 0:
                        if _RHS > largestRHS[1]:
                            largestRHS[0] = cons[idx].name
                            largestRHS[1] = _RHS
                        if _RHS < smallestRHS[1]:
                            smallestRHS[0] = cons[idx].name
                            smallestRHS[1] = _RHS
            logging.info(f"Numeric Range Statistics:\nLargest Matrix Coefficient: {largestCoeff[1]} in {largestCoeff[0]}\nSmallest Matrix Coefficient: {smallestCoeff[1]} in {smallestCoeff[0]}\nLargest RHS: {largestRHS[1]} in {largestRHS[0]}\nSmallest RHS: {smallestRHS[1]} in {smallestRHS[0]}")

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
        # (gives Warning: unable to write requested result file ".//outputs//logs//model.ilp" if feasible)
        solver_parameters   = f"ResultFile={os.path.dirname(solver['solverOptions']['logfile'])}//infeasibleModelIIS.ilp"

        if solverName == "gurobi_persistent":
            self.opt = pe.SolverFactory(solverName, options=solverOptions)
            self.opt.set_instance(self.model,symbolic_solver_labels=solver["useSymbolicLabels"])
            self.results    = self.opt.solve(tee=solver["verbosity"], logfile=solver["solverOptions"]["logfile"],options_string=solver_parameters)
        else:
            self.opt = pe.SolverFactory(solverName)
            self.results    = self.opt.solve(self.model,tee=solver["verbosity"], keepfiles=True, logfile=solver["solverOptions"]["logfile"])
        # enable logger
        logging.disable(logging.NOTSET)
        self.model.solutions.load_from(self.results)

    def addNewlyBuiltCapacity(self,stepHorizon):
        """ adds the newly built capacity to the existing capacity
        :param stepHorizon: step of the rolling horizon """
        if self.system["useRollingHorizon"]:
            _builtCapacity      = pd.Series(self.model.builtCapacity.extract_values())
            _investedCapacity   = pd.Series(self.model.investedCapacity.extract_values())
            _capex              = pd.Series(self.model.capex.extract_values())
            _roundingValue      = 10 ** (-EnergySystem.getSolver()["roundingDecimalPoints"])
            _builtCapacity[_builtCapacity <= _roundingValue]        = 0
            _investedCapacity[_investedCapacity <= _roundingValue]  = 0
            _capex[_capex <= _roundingValue]                        = 0
            _baseTimeSteps      = EnergySystem.decodeYearlyTimeSteps([stepHorizon])
            Technology          = getattr(sys.modules[__name__], "Technology")
            for tech in Technology.getAllElements():
                # new capacity
                _builtCapacityTech      = _builtCapacity.loc[tech.name].unstack()
                _investedCapacityTech   = _investedCapacity.loc[tech.name].unstack()
                _capexTech              = _capex.loc[tech.name].unstack()
                tech.addNewlyBuiltCapacityTech(_builtCapacityTech,_capexTech,_baseTimeSteps)
                tech.addNewlyInvestedCapacityTech(_investedCapacityTech,stepHorizon)

    def addCarbonEmissionsCumulative(self,stepHorizon):
        """ overwrite previous carbon emissions with cumulative carbon emissions
        :param stepHorizon: step of the rolling horizon """
        if self.system["useRollingHorizon"]:
            energySystem                            = EnergySystem.getEnergySystem()
            intervalBetweenYears                    = EnergySystem.getSystem()["intervalBetweenYears"]
            _carbonEmissionsCumulative              = self.model.carbonEmissionsCumulative.extract_values()[stepHorizon]
            _carbonEmissions                        = self.model.carbonEmissionsTotal.extract_values()[stepHorizon]
            energySystem.previousCarbonEmissions    = _carbonEmissionsCumulative + _carbonEmissions*(intervalBetweenYears-1)


