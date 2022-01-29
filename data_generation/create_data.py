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

# inputs for the creation of the new data folder
numberScenarios = 1
numberTimeSteps = 1
 
data = dict()
data['mainFolder'] = 'NUTS2'
data['sourceData'] = pd.read_csv('{}.csv'.format(data['mainFolder']), header=0, index_col=None)
data['PWAData'] = pd.read_csv('PWA.csv', header=0, index_col=None)
data['NLData'] = pd.read_csv('NL.csv', header=0, index_col=None)
Create = Create(data, analysis)

###                                                                                                                  ###
# Data are added in the following ways:
# 1. source data file defined in dictionary data: the script looks for the items in headerInSource among the columns
#   of the source data file
# 2. input data in the dictionary <data>
# 3. Any other required input data from <analysis['headerDataInputs']> is assigned to zero
###                                                                                                                  ###

data['scenario'] = ['a']
data['time'] = [0]
### Nodes
headerInSource = {'node': 'ID', 'x': "('X', 'km')", 'y': "('Y', 'km')"}
Create.independentData('Nodes', headerInSource)
### Scenarios
headerInSource = {}
Create.independentData('Scenarios', headerInSource)
### TimeSteps
headerInSource = {}
Create.independentData('TimeSteps', headerInSource)

# compute Euclidean matrix
Create.distanceMatrix('euclidean')

### Carriers
# Inputs from file
headerInSource = {
    'electricity': {'availabilityCarrier':"('TotalGreen_potential', 'MWh')"},
    'water': {},
    'biomass': {'availabilityCarrier':"('Biomass_potential', 'MWh')"},
    'natural_gas': {},
    'hydrogen': {'demandCarrier': "('hydrogen_demand', 'MWh')"},
    'oxygen': {}
    }
## Manual inputs
# biomass
data['importPriceCarrier_biomass'] = [0.07*1000] # EUR/MWh, value from Gabrielli et al. - dry biomass
# electricity
data['exportPriceCarrier_electricity'] = [0.05*1000] # EUR/MWh, value from Gabrielli et al.
data['importPriceCarrier_electricity'] = [0.12*1000] # EUR/MWh, value from Gabrielli et al.
# natural gas
data['importPriceCarrier_natural_gas'] = [0.06*1000] # EUR/MWh, value from Gabrielli et al.
# water
data['availabilityCarrier_water'] = [1e12] # unlimited availability
## create datasets
Create.nodalData('Carriers', headerInSource)

## attributes - customised dataframe
inputDataFrame = {
    'biomass':
        {'carbon_intensity': 13/1000   # tonCO2/MWh, value from Gabrielli et al. - dry biomass
         },
    'electricity':
        {'carbon_intensity': 127/1000  # tonCO2/MWh, value from Gabrielli et al.
         },
    'natural_gas':
        {'carbon_intensity': 237/1000  # tonCO2/MWh, value from Gabrielli et al.
         },
    'oxygen':
        {'carbon_intensity': 0
         },
    'hydrogen':
        {'carbon_intensity': 0
         },
}
## create datasets
Create.attributesDataFrame('Carriers', inputDataFrame)

## ConversionTechnologies
# nodal data
headerInSource = {
    'electrolysis': {},
    'SMR':{}
}
data['availability_electrolysis'] = [1e5]   # MW, value from Gabrielli et al
data['availability_SMR'] = [1e5]            # MW, value from Gabrielli et al
Create.nodalData('ConversionTechnologies', headerInSource)
# attributes
inputDataFrame = {
    'electrolysis': {
        'minBuiltCapacity':0,
        'maxBuiltCapacity':1,
        'minLoad':0.07,
        'lifetime':8760*10, # h, value from Gabrielli et al
        'costVariable':10*10**6,
        'referenceCarrier':'hydrogen',
        'inputCarrier':'electricity water',
        'outputCarrier':'hydrogen oxygen'},
    'SMR': {
        'minBuiltCapacity':1, # MW, value from Gabrielli et al
        'maxBuiltCapacity':1,
        'lifetime': 8760*20, # h, value from Gabrielli et al
        'minLoad':0.1,
        'maxLoad':1,
     'referenceCarrier':'hydrogen', 'inputCarrier':'natural_gas', 'outputCarrier':'hydrogen carbon_dioxide'},
}
Create.attributesDataFrame('ConversionTechnologies', inputDataFrame)

# files variable approximation
inputDataFrame = {
    'electrolysis': {
        'breakpointsPWACapex':{
            'columns':['capacity'],
            'values':data['PWAData']['breakpoints_capex_electrolysis'].dropna().values
        },
        'breakpointsPWAConverEfficiency':{
            'columns':['hydrogen'],
            'values':data['PWAData']['breakpoints_efficiency_electrolysis'].dropna().values
        },
        'nonlinearCapex':{
            'columns':['capacity', 'capex'],
            'values':data['NLData'][['capacity_capex_electrolysis', 'capex_capex_electrolysis']].dropna().values
        },
        'nonlinearConverEfficiency':{
            'columns':['hydrogen','oxygen','electricity', 'water'],
            'values':data['NLData'][['hydrogen_efficiency_electrolysis', 'hydrogen_efficiency_electrolysis',
                                     'electricity_efficiency_electrolysis', 'water_efficiency_electrolysis']].dropna().values
        }
    },
    'SMR': {
        'breakpointsPWACapex':{
            'columns':['capacity'],
            'values':data['PWAData']['breakpoints_capex_SMR'].dropna().values
        },
        'breakpointsPWAConverEfficiency':{
            'columns':['hydrogen'],
            'values':data['PWAData']['breakpoints_efficiency_SMR'].dropna().values
        },
        'nonlinearCapex':{
            'columns':['capacity', 'capex'],
            'values':data['NLData'][['capacity_capex_SMR', 'capex_capex_SMR']].dropna().values
        },
        'nonlinearConverEfficiency':{
            'columns':['hydrogen','natural_gas','carbon_dioxide',],
            'values':data['NLData'][['hydrogen_efficiency_SMR', 'natural_gas_efficiency_SMR',
                                     'carbon_dioxide_efficiency_SMR']].dropna().values
        }
    },
}
Create.generalDataFrame('ConversionTechnologies', inputDataFrame)

## TransportTechnologies
# datasets based on nodes combination
headerInSource = {
    'pipeline_hydrogen': {},
    'truck_hydrogen': {}
}
data['availability_pipeline_hydrogen'] = [85] # MW, value from Gabrielli et al.
data['availability_truck_hydrogen'] = [38] # MW, value from Gabrielli et al.
data['distanceEuclidean_pipeline_hydrogen'] = Create.eucledian_distance
data['distanceEuclidean_truck_hydrogen'] = Create.eucledian_distance
data['efficiencyPerDistance_pipeline_hydrogen'] = [1]
data['efficiencyPerDistance_truck_hydrogen'] = [1]
# cost per distance dependent on the way the total capex is computed: 1/2 factor to avoid accounting a connection twice
data['costPerDistance_pipeline_hydrogen'] = [(8.2*10**3)/2] # EUR/km/MW, value fixed cost from Gabrielli et al.
Create.edgesData('TransportTechnologies', headerInSource)

# attributes
inputDataFrame = {
    'pipeline_hydrogen': {
        'minBuiltCapacity':1.6, # MW, value from Gabrielli et al.
        'maxBuiltCapacity':1e12,
        'minLoad':0,
        'lifetime':8760*50, # h, value from Gabrielli et al.
        'costVariable':1.6*10**(-6), # EUR/MW, value from Gabrielli et al.
        'lossFlow':0.00012, # 1/km, value from Gabrielli et al.
        'referenceCarrier':'hydrogen'},
    'truck_hydrogen': {
        'minBuiltCapacity': 1.6,  # MW, value from Gabrielli et al.
        'maxBuiltCapacity': 1e12,
        'minLoad': 0,
        'lifetime': 8760*10,  # h, value from Gabrielli et al.
        'carbon_intensity': 4.2*10**(-6),  # tonCO2eq/km/MWh, value from Gabrielli et al.
        'costVariable':1.6*10**(-5),  # EUR/MW, value from Gabrielli et al.
        'costFixed': 13*10**3,  # EUR/MW, value from Gabrielli et al.
        'lossFlow': 0.00012,  # 1/km, value from Gabrielli et al.
        'referenceCarrier': 'hydrogen'},
}
Create.attributesDataFrame('TransportTechnologies', inputDataFrame)