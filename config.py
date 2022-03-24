"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Model settings. Overwrite default values defined in default_config.py here.
==========================================================================================================================================================================="""
from model import default_config
from datetime import datetime

## Analysis - Defaul dictionary
analysis = default_config.analysis
## System - Defaul dictionary
system = default_config.system
## Solver - Defaul dictionary
solver = default_config.solver   

## Analysis - settings update compared to default values
analysis['dataset']                                 = 'NUTS0_electricity'
analysis['objective']                               = 'TotalCost' # choose from "TotalCost", "TotalCarbonEmissions", "Risk"
# definition of the approximation
analysis['variablesNonlinearModel']                 = {'builtCapacity': []}
analysis['nonlinearTechnologyApproximation']        = {'Capex': [], 'ConverEfficiency':[]}

## System - settings update compared to default values
system['setCarriers']                               = ['electricity','natural_gas',"hard_coal","uranium"]
system['setStorageTechnologies']                    = ["battery","pumped_hydro"]
system['setTransportTechnologies']                  = ['power_line']
system['setConversionTechnologies']                 = ["natural_gas_turbine","wind_onshore","hard_coal_plant","nuclear","photovoltaics"]
system['setNodes']                                  = ['CH','DE',"AT","IT"]#,"FR","ES","PT","CZ"]
# time steps
system["referenceYear"]                             = 2020
system["timeStepsPerYear"]                          = 2190
system["timeStepsYearly"]                           = 4
system["intervalYears"]                             = 10
system['setTimeStepsPerYear']                       = list(range(0,system["timeStepsPerYear"]))
system["numberTimeStepsPerYearDefault"]             = 50 # default number of operational time steps, only used in single-grid time series aggregation TODO number of time steps per period = 1

## Solver - settings update compared to default values
solver['gap']                                       = 0.01
solver['model']                                     = 'MILP'
solver['verbosity']                                 = True
solver['performanceCheck']['printDeltaIteration']   = 50