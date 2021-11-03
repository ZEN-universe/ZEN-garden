# =====================================================================================================================
#                                   ENERGY-CARBON OPTIMIZATION PLATFORM
# =====================================================================================================================

#                                Institute of Energy and Process Engineering
#                                     Risk and Reliability Engineering
#                                        ETH Zurich, September 2021

# ======================================================================================================================
#                                               MODEL SETTINGS
# adjust model settings here
# ======================================================================================================================
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
