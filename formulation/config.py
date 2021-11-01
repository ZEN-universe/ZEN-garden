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
system['setInputCarriers'] = ['biomass']                                               # set of input carriers
system['setOutputCarriers'] = ['hydrogen']                                               # set of output carriers
system['setConversion'] = ['Electrolysis']                                             # set of conversion technologies
system['setStorage'] = []                                                              # set of storage technologies
system['setTransport'] = ['truck']                                                     # set of transport technologies

# SOLVER SETTINGS
solver = default_config.solver                                                         # solver options:
solver['name'] = 'gurobi',                                                              # solver name
solver['gap'] = 0.01                                                                    # gap to optimality
