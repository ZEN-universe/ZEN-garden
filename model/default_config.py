"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Default settings. Changes from the default values are specified in settings.py
==========================================================================================================================================================================="""

## Analysis - dictionary declaration
# This dictionary contains all the settings related to the typology of analysis given a specific system configuration.
# The dictionary also contains default settings related to the input data.
analysis = dict()

## Solver - dictionary declaration
# This dictionary contains all the settings related to the solver of the optimisation problem.
solver = dict()  

## System - dictionary declaration
# This dictionary defines the configuration of the system by selecting the subset of technologies ot be included into the analysis.
system = dict()

## Analysis - Items assignment
# objective function definition
analysis['objective'] = 'TotalCost'
# typology of optimisation: minimize or maximize
analysis['sense']     = 'minimize'
# length of time horizon
analysis['timeHorizon'] = 25
# time resolution
analysis['timeResolution'] = 'yearly'
# discount rate
analysis['discountRate'] = 0.06
# transport distance (euclidean or actual)
analysis['transportDistance'] = 'Euclidean'
# dictionary with subsets related to set
analysis['subsets'] = {
    'setTechnologies': ['setConversionTechnologies', 'setStorageTechnologies', 'setTransportTechnologies']
    }
# headers in input files
analysis['dataInputs'] = {'nameScenarios':'scenario', 'nameNodes':'node', 'nameTimeSteps':'time',
                          'nameConversionBalance':'energy', 'nameCarrier':'carrier',
                          'PWA':{'slope':'slope', 'intercept':'intercept', 'ubSegment':'ubSegment',
                                 'lbSegment':'lbSegment'}
                          }
# file format of input data
analysis['fileFormat'] = 'csv'

## System - Items assignment
# set of energy carriers
system['setCarriers'] = ['electricity', 'gas', 'hydrogen', 'biomass', 'CO2',"water","oxygen"]
# set of energy carriers for transport
system['setTransportCarriers'] = ['hydrogen']
# set of conversion technologies
system['setConversionTechnologies'] = ['electrolysis', 'SMR', 'b_SMR', 'b_Gasification']
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

## Solver - Items assignment
# solver selection (find more solver options for gurobi here: https://www.gurobi.com/documentation/9.1/refman/parameters.html)
solver['name']      = 'gurobi_persistent'
# optimality gap
solver['MIPgap']    = 0.01
# time limit in seconds
solver['TimeLimit'] = 8760
# verbosity
solver['verbosity'] = True
# log file of results
solver['logfile'] = './/outputs//logs//pyomoLogFile.log'
# typology of model solved: MILP or MINLP
solver['model']      = 'MILP'
# parameters of meta-heuristic algorithm
solver['parametersMetaheuristic'] = {
    'FEsMax':20, 'kNumber':2, 'mNumber':2, 'q':0.05099, 'xi':0.6795, 'epsilon':1e-5, 'MaxStagIter':2,
    'minVal':1e-6, 'maxVal':1e6,'runsNumber':1
    }
# evaluation of convergence in meta-heuristic. conditionDelta: (i) relative, (ii) absolute
solver['convergenceCriterion'] = {'check': True, 'conditionDelta':'relative', 'restart':True}
# settings for performance check
solver['performanceCheck'] = {'printDeltaRun':1, 'printDeltaIteration':1}