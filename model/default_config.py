"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Default settings. Changes from the default values are specified in config.py.
==========================================================================================================================================================================="""

#%% DICTIONARY DECLARATION - analysis: system topology, problem configuration and input data
analysis = dict()

# DICTIONARY DECLARATION - system: set of technologies and carriers
system = dict()

# DICTIONARY DECLARATION - solver: solver options
solver = dict()  


#%% ITEMS ASSIGNMENT - analysis
# objective function definition
analysis['objective'] = 'TotalCost'

# typology of optimisation: minimize or maximize
analysis['sense'] = 'minimize'

# technology approximation
analysis['technologyApproximation'] = 'linear'

# length of time horizon
analysis['timeHorizon'] = 25

# time resolution
analysis['timeResolution'] = 'yearly'

# discount rate
analysis['discountRate'] = 0.06

# dictionary with subsets related to set
analysis['subsets'] = {
    'setCarriers': ['setInputCarriers', 'setOutputCarriers'], 
    'setTechnologies': ['setProductionTechnologies', 'setStorageTechnologies', 'setTransportTechnologies']
    }

# headers in input files
analysis['dataInputs'] = {'nameScenarios':'scenario', 
                          'nameNodes':'node', 
                          'nameTimeSteps':'time', 
                          'nameConversionBalance':'energy', 
                          'nameCarrier':'carrier', 
                          'PWA':{'supportPoints':'sp', 'slope':'m', 'extreme0':'x0', 'extreme1':'x1', 'value0':'y0'}
                          }

# file format of input data
analysis['fileFormat'] = 'csv'

# ITEMS ASSIGNMENT - system
# set of energy carriers
system['setInputCarriers'] = ['electricity', 'gas', 'hydrogen', 'biomass', 'CO2']

# set of energy carriers for transport
system['setTransportCarriers'] = ['hydrogen']

# set of production technologies
system['setProductionTechnologies'] = ['electrolysis', 'SMR', 'b_SMR', 'b_Gasification']

# set of storage technologies
system['setStorageTechnologies'] = ['CO2_storage']

# set of transport technologies
system['setTransportTechnologies'] = ['pipeline_hydrogen', 'truck_hydrogen_gas', 'truck_hydrogen_liquid']

# set of scenarios
system['setScenarios'] = 'a'

# set of time steps
system['setTimeSteps'] = [1,2,3]

# set of nodes
system['setNodes'] = ['Berlin', 'Zurich', 'Rome']
# folder output
system['folderOutput'] = 'outputs/results/'

# ITEMS ASSIGNMENT - solver
# solver selection (for gurobi, finds more solver options here: https://www.gurobi.com/documentation/9.1/refman/parameters.html)
solver['name'] = 'gurobi'

# optimality gap
solver['MIPgap'] = 0.001

# time limit in seconds
solver['TimeLimit'] = 8760
# log file of results
solver['logfile'] = './/outputs//logs//pyomoLogFile.log'