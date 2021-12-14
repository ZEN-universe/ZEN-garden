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

## Analysis - settings update compared to default values
analysis['timeHorizon'] = 1                                                      
analysis['spatialResolution'] = 'Test3'
analysis['modelFormulation'] = 'HSC'
analysis['technologyApproximationCapex'] = 'PWA'
analysis['technologyApproximationEfficiency'] = 'linear'
analysis['objective'] = 'BasicTotalCost'

## System - settings update compared to default values
system['setInputCarriers'] = ['electricity']
system['setOutputCarriers'] = ['hydrogen']
system['setStorageTechnologies'] = []
system['setTransportTechnologies'] = ['pipeline_hydrogen']
system['setProductionTechnologies'] = ['electrolysis']
system['setScenarios'] = 'a'
system['setTimeSteps'] = [0]
system['setNodes'] = list(string.ascii_uppercase[:3])

## Solver - settings update compared to default values
solver['gap'] = 0.01