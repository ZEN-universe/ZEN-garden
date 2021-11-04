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
from formulation.config import analysis
from functions.create import Create
from functions.fill import Fill

# inputs for the creation of a the new data folder
numberScenarios = 10
numberTimeSteps = 3
 
data = dict()
data['mainFolder'] = 'NEW'
data['setCarriersIn'] = ['dry_biomass', 'electricity']
data['setCarriersOut'] = ['hydrogen']
data['setNodes'] = {'Names': ['Rome', 'Zurich', 'Berlin'], 'XCoord':[0.2, 1.2, 0.5], 'YCoord':[0.1, 0.4, 3.7]}
data['setProduction'] = ['electrolysis']
data['setScenarios'] = list(string.ascii_lowercase)[:numberScenarios]
data['setStorage'] = ['carbon_storage']
data['setTimeSteps'] = np.arange(numberTimeSteps, dtype=np.int)
data['setTransport'] = ['pipeline_hydrogen', 'truck_hydrogen_liquid']

Create = Create(data, analysis)
Create.mainFolder(data['mainFolder'])
Create.secondLevelFolders()
Create.thirdLevelFolders()

Create.carriersInFiles()
Create.carriersOutFiles()
Create.productionFiles()
Create.storageFiles()
Create.transportFiles()

Create.nodesFiles()
Create.scenariosFiles()
Create.timeStepsFiles()

Fill = Fill(data)
Fill.distanceMatrix()
