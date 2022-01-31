"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Model settings. Overwrite default values defined in default_config.py here.
==========================================================================================================================================================================="""
from model import default_config
import string

## Analysis - Defaul dictionary
analysis = default_config.analysis
## System - Defaul dictionary
system = default_config.system
## Solver - Defaul dictionary
solver = default_config.solver   
analysis['sense']

## Analysis - settings update compared to default values
analysis['spatialResolution'] = 'NUTS0'
analysis['modelFormulation'] = 'HSC'
analysis['objective'] = 'TotalCost'
# definition of the approximation
analysis['variablesNonlinearModel'] = {'builtCapacity': ['electrolysis']}
analysis['nonlinearTechnologyApproximation'] = {'Capex': ['electrolysis'], 'ConverEfficiency':[]}
analysis['linearTechnologyApproximation'] = {'Capex': [], 'ConverEfficiency':['electrolysis']}

## System - settings update compared to default values
system['setCarriers'] = ['electricity','hydrogen','water', 'oxygen']
system['setStorageTechnologies'] = []
system['setTransportTechnologies'] = ['pipeline_hydrogen']
system['setConversionTechnologies'] = ['electrolysis']
system['setScenarios'] = 'a'
system['setTimeSteps'] = [0]
if analysis['spatialResolution'] == 'NUTS0':
    system['setNodes'] = ['DE', 'AT', 'CH'] 
elif analysis['spatialResolution'] == 'NUTS2':
    system['setNodes'] = ['BE21', 'BE23', 'BE24'] # for zero demand: 'BE10', 'BE24', 'BE31'

## Solver - settings update compared to default values
solver['gap'] = 0.01
solver['model'] = 'MILP'
solver['verbosity'] = True
solver['performanceCheck']['printDeltaIteration'] = 50