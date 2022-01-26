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
data['mainFolder'] = 'NUTS0'
data['sourceData'] = pd.read_csv('NUTS0.csv', header=0, index_col=None)
Create = Create(data, analysis)

# headerInSource:  dictionary with the columns to be picked from input file if any
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

# ImportCarriers
headerInSource = {
    'electricity': {'availabilityCarrier':"('TotalGreen_potential', 'MWh')"},
    'water': {},
    'biomass': {'availabilityCarrier':"('Biomass_potential', 'MWh')"}
    }
data['importPriceCarrier_electricity'] = [1e2]
data['availabilityCarrier_water'] = [1e9]

Create.carrierDependentData('ImportCarriers', headerInSource)
# ExportCarriers
headerInSource = {
    'hydrogen': {'demandCarrier':"('hydrogen_demand', 'MWh')"},
    'oxygen': {}
}
data['importPriceCarrier_hydrogen'] = [1e2]
Create.carrierDependentData('ExportCarriers', headerInSource)