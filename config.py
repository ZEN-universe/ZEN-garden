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
analysis['timeHorizon'] = 1                                                      
analysis['spatialResolution'] = 'Test3'
analysis['modelFormulation'] = 'HSC'
analysis['objective'] = 'BasicTotalCost'

# TODO: MINLP-related
analysis['variablesNonlinearModel'] = {'capacity': ['electrolysis']}
analysis['nonlinearTechnologyApproximation'] = {'Capex': ['electrolysis'], 'ConverEfficiency':[]}
analysis['linearTechnologyApproximation'] = {'Capex': [], 'ConverEfficiency':['electrolysis']}
analysis['objective'] = 'TotalCost'

## System - settings update compared to default values
system['setImportCarriers'] = ['electricity']
system['setExportCarriers'] = ['hydrogen']
system['setStorageTechnologies'] = []
system['setTransportTechnologies'] = ['pipeline_hydrogen']
system['setConversionTechnologies'] = ['electrolysis']
system['setScenarios'] = 'a'
system['setTimeSteps'] = [0,1]
system['setNodes'] = list(string.ascii_uppercase[:3]) #TODO: define proper nomenclature for nodes

## Solver - settings update compared to default values
solver['gap'] = 0.01
solver['model'] = 'MINLP'