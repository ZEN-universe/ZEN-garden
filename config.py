"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Model settings. Overwrite default values defined in default_config.py here.
==========================================================================================================================================================================="""
from model import default_config

# ANALYSIS FRAMEWORK
analysis = default_config.analysis
analysis['timeHorizon'] = 1                                                            # length of time horizon in years
analysis['spatialResolution'] = 'NUTS0' # config
analysis['modelFormulation'] = 'HSC'

# TOPOLOGY OF THE VALUE CHAIN SYSTEM
system = default_config.system
system['setCarriersIn'] = ['electricity', 'dry_biomass']                                               # set of energy carriers
system['setCarriersOut'] = ['hydrogen']                                                 # set of energy carriers
system['setConversion'] = ['electrolysis']                                             # set of conversion technologies
system['setStorage'] = ['carbon_storage']                                                             # set of storage technologies
system['setTransport'] = ['pipeline_hydrogen', 'truck_hydrogen_gas', 'truck_hydrogen_liquid']                                                # set of transport technologies
system['setProduction'] = ['electrolysis']
system['setScenarios'] = ['a']                                                    # set of scenarios
system['setTimeSteps'] = [1]                                                       # set of time steps
system['setNodes'] = ['Berlin', 'Rome']

# SOLVER SETTINGS
solver = default_config.solver                                                         # solver options:
solver['name'] = 'gurobi',                                                              # solver name
solver['gap'] = 0.01                                                                    # gap to optimality
