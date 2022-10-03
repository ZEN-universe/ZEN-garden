"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

Description:  Model settings. Overwrite default values defined in default_config.py here.
==========================================================================================================================================================================="""
from model import default_config
import importlib.util
import os

## Analysis - Default dictionary
analysis = default_config.analysis
## Solver - Default dictionary
solver = default_config.solver
## Scenarios - Default scenario dictionary
scenarios = default_config.scenarios

## Analysis - settings update compared to default values
analysis                                            = default_config.analysis         # get default settings from default config
analysis["dataset"] = "test_7a"
analysis["objective"]                               = "TotalCost"                     # choose from "TotalCost", "TotalCarbonEmissions", "Risk"
analysis["useExistingCapacities"]                   = True                           # use greenfield or brownfield approach
## Solver - settings update compared to default values
#
solver["name"] = "glpk" # free solver
# solver["solverOptions"]["Method"]       = 2
# solver["solverOptions"]["NodeMethod"]   = 2
solver["solverOptions"]["BarHomogeneous"]   = 1
# solver["solverOptions"]["Presolve"]     = -1
solver["solverOptions"]["Threads"]      = 46
# solver["solverOptions"]["CrossoverBasis"]   = 0
# solver["solverOptions"]["Crossover"]    = 0
solver["solverOptions"]["ScaleFlag"]    = 2

# This is only not needed for 6a this way is a bit hacky to make run_test.bat work easily
if analysis['dataset'] is not 'test_6a':
    solver["analyzeNumerics"]               = True
solver["immutableUnit"]                 = ["hour","km"]

## System - load system configurations
spec    = importlib.util.spec_from_file_location("module", f"data/{analysis['dataset']}/system.py")
module  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
system  = module.system
## overwrite default system and scenario dictionaries
if system["conductScenarioAnalysis"]:
    assert os.path.exists(f"data/{analysis['dataset']}/scenarios.py"), f"scenarios.py is missing from selected dataset '{analysis['dataset']}"
    spec        = importlib.util.spec_from_file_location("module", f"data/{analysis['dataset']}/scenarios.py")
    module      = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scenarios   = module.scenarios
