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
data['scenario'] = ['a']
data['time'] = ['1']
headerInSource = {'node': 'ID',
                  'x': "('X', 'km')",
                  'y': "('Y', 'km')",
                  }

Create = Create(data, analysis)

for name in ['Nodes', 'Scenarios', 'TimeSteps']:
    Create.columnIndepentData(name, headerInSource)




