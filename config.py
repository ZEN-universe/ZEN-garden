"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Model settings. Overwrite default values defined in default_config.py here.
==========================================================================================================================================================================="""
from model import default_config

## Analysis - Defaul dictionary
analysis = default_config.analysis
## System - Defaul dictionary
system = default_config.system
## Solver - Defaul dictionary
solver = default_config.solver   

## Analysis - settings update compared to default values
analysis['spatialResolution']                       = 'NUTS0_Test_TSA'
analysis['modelFormulation']                        = 'HSC'
analysis['objective']                               = 'TotalCost' # choose from "TotalCost", "TotalCarbonEmissions", "Risk"
# definition of the approximation
analysis['variablesNonlinearModel']                 = {'builtCapacity': []}
analysis['nonlinearTechnologyApproximation']        = {'Capex': [], 'ConverEfficiency':[]}

## System - settings update compared to default values
system['setCarriers']                               = ['electricity','natural_gas']
system['setStorageTechnologies']                    = ["battery"]
system['setTransportTechnologies']                  = ['power_line']
system['setConversionTechnologies']                 = ['photovoltaics',"natural_gas_turbine","slow_natural_gas_turbine"]
system['setScenarios']                              = 'a'
system['setTimeSteps']                              = list(range(1,24+1))
system['setNodes']                                  = ['CH', 'DE'] 

## Solver - settings update compared to default values
solver['gap']                                       = 0.01
solver['model']                                     = 'MILP'
solver['verbosity']                                 = True
solver['performanceCheck']['printDeltaIteration']   = 50