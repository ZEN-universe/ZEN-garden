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
analysis['timeResolution'] = 'yearly'                                                  # time resolution
analysis['discountRate'] = 0.06                                                        # discount rate

analysis['dataInputs'] = {                                                             # names used as headers or indexes in the files
    'nameScenarios':'scenario',
    'nameNodes':'node',
    'nameTimeSteps':'time',    
    }
analysis['carrierSubsets'] = ['setCarriersIn', 'setCarriersOut']                         # subsets of carriers
analysis['technologySubsets'] = ['setProduction', 'setStorage', 'setTransport']          # subsets of technologies
analysis['fileFormat'] = 'csv'

# TOPOLOGY OF THE SYSTEM
system = dict()
system['setCarriersIn'] = ['electricity', 'gas', 'hydrogen', 'biomass', 'CO2']           # set of energy carriers
system['setProduction'] = ['electrolysis', 'SMR', 'b_SMR', 'b_Gasification']           # set of production technologies
system['setStorage'] = ['CO2_storage']                                                 # set of storage technologies
system['setTransport'] = ['pipeline_hydrogen', 'truck_hydrogen_gas', 'truck_hydrogen_liquid']                         # set of transport technologies
system['setScenarios'] = ['a', 'b']                                                    # set of scenarios
system['setTimeSteps'] = [1,2,3]                                                       # set of time steps
system['setNodes'] = ['Berlin', 'Zurich', 'Rome']                                      # set of nodes


# SOLVER SETTINGS
solver = dict()                                                                         # solver options:
solver['name'] = 'gurobi',                                                              # solver name
solver['gap'] = 0.01                                                                    # gap to optimalityanalysis
