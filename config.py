"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Model settings. Overwrite default values defined in default_config.py here.
==========================================================================================================================================================================="""
from model import default_config
import importlib.util

## Analysis - Defaul dictionary
analysis = default_config.analysis
## Solver - Defaul dictionary


## Analysis - settings update compared to default values
analysis                                            = default_config.analysis         # get default settings from default config
analysis["dataset"]                                 = "NUTS0_HSC"                     # select a dataset
analysis["objective"]                               = "TotalCost"                     # choose from "TotalCost", "TotalCarbonEmissions", "Risk"


## Solver - settings update compared to default values
solver                                              = default_config.solver           # get default settings from default config
solver["gap"]                                       = 0.01
solver["model"]                                     = "MILP"
solver["verbosity"]                                 = True
solver["performanceCheck"]["printDeltaIteration"]   = 50


## System - load system configurations
spec    = importlib.util.spec_from_file_location("module", f"data/{analysis['dataset']}/system.py")
module  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
system  = module.system
