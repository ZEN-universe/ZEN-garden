"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

Description:  Default settings. Changes from the default values are specified in settings.py
==========================================================================================================================================================================="""

## Analysis - dictionary declaration
# This dictionary contains all the settings related to the typology of analysis given a specific system configuration.
# The dictionary also contains default settings related to the input data.
analysis = dict()

## Solver - dictionary declaration
# This dictionary contains all the settings related to the solver of the optimisation problem.
solver = dict()

## System - dictionary declaration
# This dictionary defines the configuration of the system by selecting the subset of technologies ot be included into the analysis.
system = dict()

## Solver - dictionary declaration
# This dictionary contains all the settings related to the solver of the optimisation problem.
solver = dict()

## Scenarios - dictionary declaration
# This dictionary defines the set of scenarios that is evaluated.
scenarios = dict()


## Analysis - Items assignment
# objective function definition
analysis["objective"] = "TotalCost"
# typology of optimisation: minimize or maximize
analysis["sense"]     = "minimize"
# discount rate
analysis["discountRate"] = 0.06
# transport distance (euclidean or actual)
analysis["transportDistance"] = "Euclidean"
# dictionary with subsets related to set
analysis["subsets"] = {"setCarriers": [],
                       "setTechnologies": ["setConversionTechnologies", "setTransportTechnologies","setStorageTechnologies"]}
# headers for the generation of input files
analysis["headerDataInputs"] =   {"setNodes": ["node", "x", "y"],
                                  "setEdges": ["edge"],
                                  "setScenarios":["scenario"],
                                  "setTimeSteps":["time"],
                                  "setTimeStepsYearly":["year"],
                                  "setCarriers":["demandCarrier", "availabilityCarrier", "exportPriceCarrier", "importPriceCarrier"],
                                  "setConversionTechnologies":["availability"],
                                  "setTransportTechnologies":["availability", "costPerDistance", "distanceEuclidean", "efficiencyPerDistance"],
                                  "setExistingTechnologies": ["existingTechnology"]}

# file format of input data
analysis["fileFormat"] = "csv"
# time series aggregation
analysis["timeSeriesAggregation"] = {
    "clusterMethod"         : "k_means",
    "solver"                : "gurobi",
    "hoursPerPeriod"        : 1,
    "extremePeriodMethod"   : "None",
    "rescaleClusterPeriods" : False,
    "representationMethod"  : "meanRepresentation",
    "resolution"            : 1,
    "segmentation"          : False,
    "noSegments"            : 12
}

analysis['headerDataOutputs']=   {'capexTotal': ['capacity[€]'],
                                'costCarrierTotal': ['capacity[€]'],
                                'opexTotal':['capacity[€]'],
                                'carbonEmissionsCarrierTotal':['capacity[GWh]'],
                                'carbonEmissionsTechnologyTotal':['capacity[GWh]'],
                                'carbonEmissionsTotal':['capacity[GWh]'],
                                'carbonEmissionsCarrier':['carrier','node','time','capacity[GWh]'],
                                'costCarrier': ['carrier','node','time','capacity[GWh]'],
                                'exportCarrierFlow':['carrier','node','time','capacity[GWh]'],
                                'importCarrierFlow':['carrier','node','time','capacity[GWh]'],
                                'carrierFlow':['transportationTechnology','edge','time','capacity[GWh]'],
                                'carrierLoss':['transportationTechnology','edge','time','capacity[GWh]'],
                                'dependentFlowApproximation':['conversionTechnology','carrier','node','time','capacity[GWh]'],
                                'inputFlow':['conversionTechnology','carrier','node','time','capacity[GWh]'],
                                'outputFlow':['conversionTechnology','carrier','node','time','capacity[GWh]'],
                                'referenceFlowApproximation':['conversionTechnology','carrier','node','time','capacity[GWh]'],
                                'installTechnology':['conversionTechnology','node','time','T/F'],
                                'builtCapacity':['conversionTechnology','node','time','capacity[GWh]'],
                                'capacity':['conversionTechnology','node','time','capacity[GWh]'],
                                'capacityApproximation':['conversionTechnology','node','time','capacity[GWh]'],
                                'capex':['conversionTechnology','node','time','capacity[GWh]'],
                                'capexApproximation':['conversionTechnology','node','time','capacity[GWh]'],
                                'carbonEmissionsTechnology':['conversionTechnology','node','time','capacity[GWh]'],
                                'opex':['conversionTechnology','node','time','capacity[GWh]'],
                                'carrierFlowCharge':['?','??','???','????'],
                                'carrierFlowDischarge':['?','??','???','????'],
                                'levelCharge':['?','??','???','????'],
                                }

analysis['postprocess'] = False

## System - Items assignment
# set of energy carriers
system["setCarriers"] = []
# set of conditioning carriers
system["setConditioningCarriers"] = []
# set of capacity types: power-rated or energy-rated
system["setCapacityTypes"] = ["power","energy"]
# set of conversion technologies
system["setConversionTechnologies"] = []
# set of conditioning technologies
system["setConditioningTechnologies"] = []
# set of storage technologies
system["setStorageTechnologies"] = []
# set of transport technologies
system["setTransportTechnologies"] = []
system['DoubleCapexTransport'] = False
system["setBidirectionalTransportTechnologies"] = []
# set of nodes
system["setNodes"] = []
# toggle to use timeSeriesAggregation
system["conductTimeSeriesAggregation"] = False
# toggle to perform analysis for multiple scenarios
system["conductScenarioAnalysis"] = False
# total hours per year
system["totalHoursPerYear"] = 8760
# unbounded market share for technology diffusion rate
system["unboundedMarketShare"] = 0.01
# rate at which the knowledge stock of existing capacities is depreciated annually
system["knowledgeDepreciationRate"] = 0.1
# spillover rate of knowledge stock to another
system["knowledgeSpilloverRate"] = 0.05
# social discount rate
system["socialDiscountRate"] = 0
# folder output
system["folderOutput"] = "outputs/results/"
# name of data folder for energy system specification
system["folderNameSystemSpecification"] = "systemSpecification"


## Solver - Items assignment
# solver selection (find more solver options for gurobi here: https://www.gurobi.com/documentation/9.1/refman/parameters.html)
solver["name"]      = "glpk"
# gurobi options
solver["solverOptions"] = {
    "logfile":      ".//outputs//logs//pyomoLogFile.log",
    "MIPGap":       None,
    "TimeLimit":    None,
    "Method":       None
}
# use symbolic labels, only sensible for debugging infeasible problems. Adds overhead
solver["useSymbolicLabels"] = False
# analyze numerics
solver["analyzeNumerics"]   = False
solver["immutableUnit"]     = []
solver["rangeUnitExponents"]    = {"min":-3,"max":3,"stepWidth":1}
# round down to number of decimal points, for new capacity and unit multipliers
solver["roundingDecimalPoints"]     = 5
# round down to number of decimal points, for time series after TSA
solver["roundingDecimalPointsTS"]   = 3
# verbosity
solver["verbosity"] = True
# typology of model solved: MILP or MINLP
solver["model"]      = "MILP"
# parameters of meta-heuristic algorithm
solver["parametersMetaheuristic"] = {
    "FEsMax":1e12, "kNumber":90, "mNumber":5, "q":0.05099, "xi":0.6795, "epsilon":1e-5, "MaxStagIter":650,
    "minVal":1e-6, "maxVal":1e6,"runsNumber":1
    }
# evaluation of convergence in meta-heuristic. conditionDelta: (i) relative, (ii) absolute
solver["convergenceCriterion"] = {"check": True, "conditionDelta":"relative", "restart":True}
# settings for performance check
solver["performanceCheck"] = {"printDeltaRun":1, "printDeltaIteration":1}
# settings for selection of x-y relationships, which are modeled as PWA, and which are modeled linearly:
# linear regression of x-y values: if relative intercept (intercept/slope) below threshold and rvalue above threshold, model linear with slope
solver["linearRegressionCheck"] = {"epsIntercept":0.1,"epsRvalue":1-(1E-5)}

## Scenarios - dictionary declaration
scenarios[""] = {}