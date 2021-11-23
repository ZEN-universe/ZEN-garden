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
analysis['timeHorizon'] = 1                                                            
analysis['spatialResolution'] = 'NUTS0'
analysis['modelFormulation'] = 'HSC'
analysis['technologyApproximation'] = 'linear'

## System - settings update compared to default values
system['setInputCarriers'] = ['electricity', 'dry_biomass']
system['setOutputCarriers'] = ['hydrogen']
system['setStorageTechnologies'] = ['carbon_storage']
system['setTransportTechnologies'] = ['pipeline_hydrogen', 'truck_hydrogen_gas', 'truck_hydrogen_liquid']
system['setProductionTechnologies'] = ['electrolysis']
system['setScenarios'] = 'a'
system['setTimeSteps'] = [1]
system['setNodes'] = ['Berlin', 'Rome']

## Solver - settings update compared to default values
solver['gap'] = 0.01