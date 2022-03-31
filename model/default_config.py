"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

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

## Analysis - Items assignment
# objective function definition
analysis["objective"] = "TotalCost"
# typology of optimisation: minimize or maximize
analysis["sense"]     = "minimize"
# length of time horizon
analysis["timeHorizon"] = 25
# time resolution
analysis["timeResolution"] = "yearly"
# discount rate
analysis["discountRate"] = 0.06
# transport distance (euclidean or actual)
analysis["transportDistance"] = "Euclidean"
# dictionary with subsets related to set
analysis["subsets"] = {"setTechnologies": ["setConversionTechnologies", "setTransportTechnologies","setStorageTechnologies"]}
# settings for MINLP
analysis["variablesNonlinearModel"]                 = {"builtCapacity": []}
analysis["nonlinearTechnologyApproximation"]        = {"Capex": [], "ConverEfficiency":[]}
# headers for the generation of input files
analysis["headerDataInputs"] =   {"setNodes": ["node", "x", "y"],
                                  "setEdges": ["edge"],
                                  "setScenarios":["scenario"],
                                  "setTimeSteps":["time"],
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
    "extremePeriodMethod"   : "None",
    "resolution"            : 1
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
# set of conversion technologies
system["setConversionTechnologies"] = []
# set of storage technologies
system["setStorageTechnologies"] = []
# set of transport technologies
system["setTransportTechnologies"] = []
# set of nodes
system["setNodes"] = []
# time steps
system["referenceYear"]                             = 2020
system["timeStepsPerYear"]                          = 1
system["timeStepsYearly"]                           = 15
system["intervalYears"]                             = 1
system['setTimeStepsPerYear']                       = list(range(0,system["timeStepsPerYear"]))
system["numberTimeStepsPerYearDefault"]             = 1 # default number of operational time steps, only used in single-grid time series aggregation TODO number of time steps per period = 1
system["totalHoursPerYear"]                         = 8760

# folder output
system["folderOutput"] = "outputs/results/"

## Solver - Items assignment
# solver selection (find more solver options for gurobi here: https://www.gurobi.com/documentation/9.1/refman/parameters.html)
solver["name"]      = "gurobi_persistent"
# gurobi options
solver["solverOptions"] = {}
# optimality gap
solver["solverOptions"]["MIPgap"]    = 0.01
# time limit in seconds
solver["solverOptions"]["TimeLimit"] = 8760
# log file of results
solver["solverOptions"]["logfile"] = ".//outputs//logs//pyomoLogFile.log"
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
# rounding to number of decimal points
solver["roundingDecimalPoints"] = 10
