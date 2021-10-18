# =====================================================================================================================
#                                   ENERGY-CARBON OPTIMIZATION PLATFORM
# =====================================================================================================================

#                                Institute of Energy and Process Engineering
#                                     Risk and Reliability Engineering
#                                        ETH Zurich, September 2021

# ======================================================================================================================
#                                               DEFAULT SETTINGS
# default settings of the model. Do not change this script. Changes from the default values are specified in settings.py
# ======================================================================================================================


# ANALYSIS FRAMEWORK
analysis = dict()
analysis['objective'] = 'minimum-cost'                                                 # objective function
analysis['technologyApproximation'] = 'linear'                                         # technology approximation
analysis['timeHorizon'] = 25                                                           # length of time horizon in years
analysis['yearly'] = 'yearly'                                                          # time resolution
analysis['discountRate'] = 0.06                                                        # discount rate

# TOPOLOGY OF THE SYSTEM
system = dict()
system['setCarriers'] = ['electricity', 'gas', 'hydrogen', 'biomass', 'CO2']           # set of energy carriers
system['setProduction'] = ['Electrolysis', 'SMR', 'b_SMR', 'b_Gasification']           # set of production technologies
system['setStorage'] = ['CO2_storage']                                                 # set of storage technologies
system['setTransport'] = ['pipeline', 'truck', 'rail', 'ship']                         # set of transport technologies

# SOLVER SETTINGS
solver = dict()                                                                         # solver options:
solver['name'] = 'gurobi',                                                              # solver name
solver['gap'] = 0.01                                                                    # gap to optimality
