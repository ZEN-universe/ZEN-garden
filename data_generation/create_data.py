"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      January-2022
Authors:      Davide Tonelli (davidetonelli@outlook.com)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Class containing the methods for the generation of the input dataset respecting the platform's data structure.
==========================================================================================================================================================================="""
import numpy as np
import pandas as pd
import os, sys
import string
main_directory = os.path.abspath('..//..')
sys.path.insert(1, main_directory)
from config import analysis
from functions.create import Create
from functions.fill import Fill

# inputs for the creation of a the new data folder
numberScenarios = 1
numberTimeSteps = 1
 
data = dict()
data['mainFolder'] = 'NUTS2'
data['sourceData'] = pd.read_csv('{}.csv'.format(data['mainFolder']), header=0, index_col=None)
Create = Create(data, analysis)

###                                                                                                                  ###
# Data are added in the following ways:
# 1. source data file defined in dictionary data: the script looks for the items in headerInSource among the columns
#   of the source data file
# 2. input data in the dictionary <data>
# 3. Any other required input data from <analysis['headerDataInputs']> is assigned to zero
###                                                                                                                  ###

data['scenario'] = ['a']
data['time'] = [1]
# Nodes
headerInSource = {'node': 'ID', 'x': "('X', 'km')", 'y': "('Y', 'km')"}
Create.independentData('Nodes', headerInSource)
# Scenarios
headerInSource = {}
Create.independentData('Scenarios', headerInSource)
# TimeSteps
headerInSource = {}
Create.independentData('TimeSteps', headerInSource)

# compute Eucleadian matrix
Create.distanceMatrix('eucledian')

# ImportCarriers
headerInSource = {
    'electricity': {'availabilityCarrier':"('TotalGreen_potential', 'MWh')"},
    'water': {},
    'biomass': {'availabilityCarrier':"('Biomass_potential', 'MWh')"}
    }
data['importPriceCarrier_electricity'] = [1e2]
data['availabilityCarrier_water'] = [1e9]

Create.indexedData('ImportCarriers', headerInSource)
# ExportCarriers
headerInSource = {
    'hydrogen': {'demandCarrier':"('hydrogen_demand', 'MWh')"},
    'oxygen': {}
}
data['importPriceCarrier_hydrogen'] = [1e2]
Create.indexedData('ExportCarriers', headerInSource)

## ConversionTechnologies
# indexed data
headerInSource = {
    'electrolysis': {},
    'SMR':{}
}
data['availability_electrolysis'] = [1e6]
data['availability_SMR'] = [1e6]
Create.indexedData('ConversionTechnologies', headerInSource)
# empty datasets
newFiles = {
    'attributes':{
        'index':['minBuiltCapacity', 'maxBuiltCapacity', 'minLoad', 'maxLoad', 'lifetime',
                 'referenceCarrier', 'inputCarrier', 'outputCarrier'],
        'columns':['attibutes']
                  },
    'breakpointsPWACapex':{
        'index':[None],
        'columns':['capacity']
           },
    'breakpointsPWAConverEfficiency': {
        'index': [None],
        'columns': [None]
    },
    'nonlinearCapex': {
        'index': [None],
        'columns': ['capacity','capex']
    },
    'nonlinearConverEfficiency': {
        'index': [None],
        'columns': [None]
    }
}
Create.newFiles('ConversionTechnologies', headerInSource, newFiles)

## TransportTechnologies
headerInSource = {
    'pipeline_hydrogen': {},
    'truck_hydrogen':{}
}
data['availability_pipeline_hydrogen'] = [1e6]
data['availability_truck_hydrogen'] = [1e6]
Create.indexedData('TransportTechnologies', headerInSource)

# datasets based on nodes combination
headerInSource = {
    'pipeline_hydrogen': {},
    'truck_hydrogen': {}
}
data['distanceEuclidean_pipeline_hydrogen'] = Create.eucledian_distance
data['distanceEuclidean_truck_hydrogen'] = Create.eucledian_distance
data['efficiencyPerDistance_pipeline_hydrogen'] = [1]
data['efficiencyPerDistance_truck_hydrogen'] = [1]
data['costPerDistance_pipeline_hydrogen'] = [1]
data['costPerDistance_truck_hydrogen'] = [1]
Create.distanceData('TransportTechnologies', headerInSource)

# empty datasets
newFiles = {
    'attributes':{
        'index':['minBuiltCapacity', 'maxBuiltCapacity', 'minLoad', 'maxLoad', 'lifetime',
                 'referenceCarrier'],
        'columns':['attibutes']
                  },
}
Create.newFiles('TransportTechnologies', headerInSource, newFiles)