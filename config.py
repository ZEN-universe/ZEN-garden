"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Model settings. Overwrite default values defined in default_config.py.
==========================================================================================================================================================================="""

# IMPORT AND SETUP
import string

from model import default_config


#%% MODEL CONFIGURATION
# DEFAULT DICTIONARIES
analysis = default_config.analysis
system   = default_config.system
solver   = default_config.solver   

# SETTING UPDATE WITH RESPECT TO DEFAULT - analysis
analysis['timeHorizon']             = 1                                                      
analysis['spatialResolution']       = 'Test'
analysis['modelFormulation']        = 'HSC'
analysis['technologyApproximation'] = 'linear'

# SETTING UPDATE WITH RESPECT TO DEFAULT - system
system['setInputCarriers']          = ['electricity']
system['setOutputCarriers']         = ['hydrogen']
system['setStorageTechnologies']    = []
system['setTransportTechnologies']  = ['pipeline_hydrogen']
system['setProductionTechnologies'] = ['electrolysis']
system['setTimeSteps']              = [0]
system['setNodes']                  = list(string.ascii_uppercase[:9])

# SETTING UPDATE WITH RESPECT TO DEFAULT - solver
solver['MIPgap'] = 0.01