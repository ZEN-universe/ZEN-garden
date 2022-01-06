"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      October-2021
Authors:      Davide Tonelli (davidetonelli@outlook.com)
Organization: Labratory of Risk and Reliability Engineering, ETH Zurich

Description:    This class contain auxiliary methods for the creation of a new folder of files which respect the standards of the code
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
data['mainFolder'] = 'Test'
data['setInputCarriers'] = ['electricity']
data['setOutputCarriers'] = ['hydrogen']
coords = np.arange(0,81, 40)
data['setNodes'] = {'Names': list(string.ascii_uppercase)[:9], 
                    'XCoord': list(np.tile(coords, (3,1)).flatten()), 
                    'YCoord': list(np.tile(coords, (3,1)).T.flatten())}
data['setConversionTechnologies'] = ['electrolysis']
data['setScenarios'] = list(string.ascii_lowercase)[:numberScenarios]
data['setStorageTechnologies'] = ['carbon_storage']
data['setTimeSteps'] = np.arange(numberTimeSteps, dtype=np.int)
data['setTransportTechnologies'] = ['pipeline_hydrogen', 'truck_hydrogen_liquid']

if True:
    Create = Create(data, analysis)
    Create.mainFolder(data['mainFolder'])
    Create.secondLevelFolders()
    Create.thirdLevelFolders()
    
    Create.carriersInFiles()
    Create.carriersOutFiles()
    Create.conversionFiles()
    Create.storageFiles()
    Create.transportFiles()
    
    Create.nodesFiles()
    Create.scenariosFiles()
    Create.timeStepsFiles()
    
    Fill = Fill(data)
    Fill.distanceMatrix()
