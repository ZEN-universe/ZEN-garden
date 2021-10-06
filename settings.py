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
from data import defaultSettings

# ANALYSIS FRAMEWORK
analysis = defaultSettings.analysis
analysis['objective'] = 'minimum-cost'                                                 # objective function
analysis['technologyApproximation'] = 'linear'                                         # technology approximation
analysis['timeHorizon'] = 25                                                           # length of time horizon in years
analysis['discountRate'] = 0.06                                                        # discount rate

# TOPOLOGY OF THE VALUE CHAIN SYSTEM
system = defaultSettings.system
system['setCarriers'] = ['electricity', 'gas', 'hydrogen', 'biomass', 'CO2']           # set of energy carriers
system['setConversion'] = ['Electrolysis', 'SMR', 'b_SMR', 'b_Gasification']           # set of conversion technologies
system['setStorage'] = ['CO2_storage']                                                 # set of storage technologies

# SOLVER SETTINGS
solver = defaultSettings.solver                                                         # solver options:
solver['name'] = 'gurobi',                                                              # solver name
solver['gap'] = 0.01                                                                    # gap to optimality
